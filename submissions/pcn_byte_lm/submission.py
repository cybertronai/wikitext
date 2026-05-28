"""Predictive Coding Network (PCN) byte-LM — strictly local Hebbian updates.

Experiment N1 from research/gradfree_analysis.md. Spec:
research/non_nn_methods/spec_12_predictive_coding_local.md.
Design memo: experiments/gradient_free/experiment_N1_pcn_byte_lm.md.

Whittington & Bogacz 2017 ("An Approximation of the Error Backpropagation
Algorithm in a Predictive Coding Network with Local Hebbian Synaptic
Plasticity") and Millidge et al. 2020 (arXiv 2006.04182) prove that an
L-layer feedforward PCN, with the top layer clamped to a target and T
inner-loop activity updates, performs a strictly **local Hebbian**
weight update that approximates backprop:

    e_l = x_l - mu_l,    mu_l = g(W_l x_{l+1})
    inner:  x_l <- x_l - alpha * (e_l - g'(mu_l) * (W_{l+1}^T e_{l+1}))
    weight: Delta W_l = -eta * e_l * x_{l+1}^T

The update for W_l depends only on the activities of layers l and l+1 -
no chain rule through the rest of the network. We use `torch.no_grad()`
throughout training and never call `loss.backward()`; the update is
materialised explicitly as an outer product.

Architecture for byte LM:
  - K=64 byte context window
  - Frozen random-orthogonal byte embedding E: (256, d_emb=32)
  - Flatten K * d_emb = 2048 -> input layer x_3
  - 3 trainable PCN layers: 2048 -> 512 -> 512 -> 256 (vocab)
  - Top layer clamped to one-hot next byte target

Inference at eval: single forward pass (no relaxation needed) — the
softmax over the unclamped top layer is the next-byte distribution.

Properties that motivate this experiment:
  1. Update is driven by an *expectation* of error (e_l is signed/dense),
     not a one-shot WTA: escapes the stochasticity filter that killed
     SoftHebb / NBB on byte targets.
  2. Activities x_l adapt during inference: escapes the Paradigm-A
     <= 0.37 ceiling that caps frozen-feature + linear-readout methods.
  3. Repo has nothing in this family; closest neighbours are
     `mono_forward_v2` (layer-local SGD with backprop within each block)
     and `hebbian_fw_block_v2` (one Hebbian block among 4 SGD blocks).
     This is the first attempt at depth-wise local-rule learning beyond
     `mono_forward_v2`.
"""
from __future__ import annotations

__author__ = "@armin-claude-1m"

import math
import os
import time

import torch
import torch.nn.functional as F
from torch import Tensor

from wikitext import CharModel


# ---------------------------------------------------------------------------
# PCN core: trainable parameters held as plain tensors (no nn.Module / autograd)
# ---------------------------------------------------------------------------

class PCNState:
    """All trainable + frozen tensors for the PCN.

    Layers (top to bottom in the index used by the math, so layer 0 = top
    output, layer L = input):

        x_0      : top, clamped to one-hot target during training.
        x_1...x_{L-1}: hidden activities (relaxed during inference).
        x_L      : bottom (frozen — flattened K-byte embedding).

    Weights W[l] map x_{l+1} -> mu_l = W[l] @ x_{l+1} + b[l].
    g_l: nonlinearity used to compute mu_l. We use no nonlinearity at
    the top (linear readout, std PCN) and ReLU at hidden layers.
    """
    def __init__(
        self,
        layer_dims: list[int],     # [d_top, d_h1, d_h2, ..., d_input]
        device: torch.device,
        dtype: torch.dtype = torch.float32,
        seed: int = 0,
    ):
        self.layer_dims = layer_dims
        self.L = len(layer_dims) - 1   # number of weight layers
        self.device = device
        self.dtype = dtype

        gen = torch.Generator(device=device).manual_seed(seed)
        self.W: list[Tensor] = []
        self.b: list[Tensor] = []
        for l in range(self.L):
            d_out = layer_dims[l]       # mu_l size
            d_in = layer_dims[l + 1]    # x_{l+1} size
            # He-normal init for hidden layers (l > 0), small init for top (l=0).
            if l == 0:
                std = 0.01 / math.sqrt(d_in)
            else:
                std = math.sqrt(2.0 / d_in)
            W = torch.empty(d_out, d_in, device=device, dtype=dtype)
            W.normal_(generator=gen, std=std)
            b = torch.zeros(d_out, device=device, dtype=dtype)
            self.W.append(W)
            self.b.append(b)


# ---------------------------------------------------------------------------
# Frozen random byte embedding (gradient-free — fixed at init)
# ---------------------------------------------------------------------------

def _make_byte_embed(
    d_emb: int, device: torch.device, seed: int = 1234,
) -> Tensor:
    """Random Gaussian byte embedding, normalized per row.

    Not trained (this is the only frozen feature in the system — all
    other weights learn via PCN's local rule). Gives the input layer
    a unique, dense, position-distinguishable representation of each
    byte while leaving all learning to the PCN dynamics.
    """
    gen = torch.Generator(device=device).manual_seed(seed)
    E = torch.empty(256, d_emb, device=device, dtype=torch.float32)
    E.normal_(generator=gen)
    E = F.normalize(E, dim=-1)
    return E


def _input_features(
    byte_windows: Tensor,    # (B, K) int64 in [0, 256)
    E: Tensor,                # (256, d_emb)
) -> Tensor:
    """Embed each byte and flatten across the K positions.

    Returns (B, K*d_emb). All positions get the same lookup table — the
    flatten gives the network position-distinguishable features.
    """
    embedded = E[byte_windows]              # (B, K, d_emb)
    return embedded.reshape(byte_windows.size(0), -1)   # (B, K*d_emb)


# ---------------------------------------------------------------------------
# PCN forward init + inner-loop relaxation + local weight update
# ---------------------------------------------------------------------------

def _act(x: Tensor) -> Tensor:
    """Hidden activation: tanh.

    PCN propagates a backward signal as e_{l-1} * g'(mu_{l-1}). With
    ReLU, g'(.) is 0 on ~50% of units and gives a discontinuous, sparse
    signal. tanh's derivative is continuous and bounded away from 0 on
    a wide range, which produces much more reliable hidden-layer
    learning under the PCN local rule (Salvatori et al. 2022 §4).
    """
    return torch.tanh(x)


def _act_grad(mu: Tensor) -> Tensor:
    """Derivative of tanh evaluated at the pre-activation mu.

    g'(mu) = 1 - tanh(mu)^2 = 1 - g(mu)^2.
    """
    a = torch.tanh(mu)
    return 1.0 - a * a


@torch.no_grad()
def _pcn_forward_init(state: PCNState, x_bottom: Tensor) -> list[Tensor]:
    """Initialise activities by running a single feedforward pass:

        x[L] = x_bottom
        for l = L-1, ..., 1:  x[l] = relu(W[l] x[l+1] + b[l])
        x[0] = W[0] x[1] + b[0]      # linear top

    Returns list `xs` indexed by layer: xs[0] = top, xs[L] = bottom.
    """
    L = state.L
    xs: list[Tensor | None] = [None] * (L + 1)
    xs[L] = x_bottom
    for l in range(L - 1, 0, -1):
        mu = xs[l + 1] @ state.W[l].T + state.b[l]
        xs[l] = _act(mu)
    # Top layer (linear; will be clamped during training).
    xs[0] = xs[1] @ state.W[0].T + state.b[0]
    return [x for x in xs]      # type: ignore[return-value]


@torch.no_grad()
def _pcn_inference(
    state: PCNState,
    xs: list[Tensor],
    y_top: Tensor | None,
    T: int,
    alpha: float,
    log_norms: list | None = None,
) -> list[Tensor]:
    """Run T iterations of activity relaxation.

    Layers 0 (top) and L (bottom) are clamped:
      - bottom = input (always)
      - top    = target y_top if provided (training); otherwise also free
                 but we don't call this in that mode — at eval we just
                 read the forward-init top directly.

    Update for free hidden layer l in [1, L-1]:
      mu_l   = W_l x_{l+1} + b_l
      e_l    = x_l - mu_l           (negated by descent on F)
      back_l = (W_{l-1}^T e_{l-1}) * g'(mu_{l-1})   if l-1 is "next-up"
      x_l   <- x_l - alpha * (e_l - back_l)
    """
    L = state.L
    if y_top is not None:
        xs[0] = y_top

    for t in range(T):
        # Recompute mu for each layer (depends on current x_{l+1}).
        mus: list[Tensor] = []
        for l in range(L):
            mu = xs[l + 1] @ state.W[l].T + state.b[l]
            mus.append(mu)
        # Errors at all layers.
        es: list[Tensor] = []
        for l in range(L):
            if l == 0:
                e = xs[0] - mus[0]      # top (clamped or free)
            else:
                e = xs[l] - _act(mus[l])
            es.append(e)

        # Update free hidden layers only (1 .. L-1).
        for l in range(1, L):
            # dF / d x_l = e_l - W_{l-1}^T * (e_{l-1} * g'(mu_{l-1}))
            # Top layer (l-1 = 0) is linear, so g'(.) = 1 there.
            if l - 1 == 0:
                back = es[0] @ state.W[0]            # (B, d_l)
            else:
                back = (es[l - 1] * _act_grad(mus[l - 1])) @ state.W[l - 1]
            x_new = xs[l] - alpha * (es[l] - back)
            xs[l] = x_new

        if log_norms is not None and t == T - 1:
            row = [float(es[l].pow(2).mean().sqrt().item()) for l in range(L)]
            log_norms.append(row)

    return xs


@torch.no_grad()
def _pcn_weight_update(
    state: PCNState,
    xs: list[Tensor],
    eta_per_layer: list[float],
    grad_clip: float = 0.0,
) -> dict:
    """Apply local Hebbian updates:

        Delta W_l = + eta_l * (1/B) * (e_l * g'(mu_l))^T @ x_{l+1}
        Delta b_l = + eta_l * mean_B(e_l * g'(mu_l))

    Per-layer eta is needed because the top layer (linear, one-hot
    targets) has a structurally larger signal than hidden layers
    (small ReLU activations); a single global eta either explodes the
    top or starves the hidden.

    If `grad_clip > 0`, each layer's update is rescaled so its Frobenius
    norm does not exceed `grad_clip`. Returns a dict with the per-layer
    update Frobenius norms (for telemetry / diagnostics).
    """
    L = state.L
    B = xs[0].size(0)
    inv_B = 1.0 / B
    norms = []
    for l in range(L):
        mu = xs[l + 1] @ state.W[l].T + state.b[l]
        if l == 0:
            e = xs[0] - mu
        else:
            e = xs[l] - _act(mu)
        if l == 0:
            modulated = e
        else:
            modulated = e * _act_grad(mu)
        dW = modulated.T @ xs[l + 1] * inv_B
        db = modulated.mean(dim=0)
        if grad_clip > 0:
            n = dW.norm()
            if n > grad_clip:
                dW = dW * (grad_clip / (n + 1e-9))
        norms.append(float(dW.norm().item()))
        state.W[l].add_(dW, alpha=eta_per_layer[l])
        state.b[l].add_(db, alpha=eta_per_layer[l])
    return {"dW_norms": norms}


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def _train_pcn(
    state: PCNState,
    E: Tensor,
    train_bytes: Tensor,
    *,
    K: int,
    batch_size: int,
    T_inner: int,
    alpha: float,
    eta_schedule,        # callable(step) -> list[float] per layer
    grad_clip: float,
    n_steps: int,
    max_seconds: float,
    log_every: int,
) -> dict:
    device = state.device
    n = train_bytes.numel()
    L = state.L

    telemetry: dict = {
        "steps_completed": 0,
        "duration_s": 0.0,
        "e_norms_last": None,
        "loss_history": [],
    }

    t0 = time.monotonic()
    print(
        f"[pcn] begin  L={L} K={K} bs={batch_size} T_inner={T_inner} "
        f"alpha={alpha:.3f}  budget={max_seconds:.0f}s n_steps={n_steps}",
        flush=True,
    )

    for step in range(n_steps):
        if time.monotonic() - t0 > max_seconds:
            print(f"[pcn] budget exhausted at step {step}", flush=True)
            break

        # Sample B windows of length K+1: K bytes context, then target.
        idx = torch.randint(0, n - K - 1, (batch_size,), device=device)
        offsets = idx[:, None] + torch.arange(K + 1, device=device)[None, :]
        flat = train_bytes[offsets].long()
        x_ctx = flat[:, :K]          # (B, K) context
        target_byte = flat[:, K]      # (B,)   next byte

        # Embed context to bottom-layer features.
        x_bottom = _input_features(x_ctx, E)        # (B, K*d_emb)

        # Build one-hot target for top layer.
        y_top = F.one_hot(target_byte, num_classes=256).to(state.dtype)

        # Forward init activities.
        xs = _pcn_forward_init(state, x_bottom)

        # Capture pre-relax top error for loss logging (cheap proxy).
        with torch.no_grad():
            pre_mu_top = xs[1] @ state.W[0].T + state.b[0]
            preds = pre_mu_top.argmax(dim=-1)
            acc = (preds == target_byte).float().mean().item()
            # CE-equivalent: -log p_target under softmax(mu_top).
            log_probs = F.log_softmax(pre_mu_top, dim=-1)
            nll = -log_probs[torch.arange(batch_size), target_byte].mean().item()

        # Relax with top clamped to target.
        log_norms = [] if (log_every and step % log_every == 0) else None
        xs = _pcn_inference(
            state, xs, y_top=y_top, T=T_inner, alpha=alpha, log_norms=log_norms,
        )

        # Local Hebbian update.
        eta_pl = eta_schedule(step)
        upd_info = _pcn_weight_update(
            state, xs, eta_per_layer=eta_pl, grad_clip=grad_clip,
        )

        telemetry["loss_history"].append(nll)
        if log_every and (step % log_every == 0 or step == n_steps - 1):
            elapsed = time.monotonic() - t0
            e_str = "—"
            if log_norms:
                telemetry["e_norms_last"] = log_norms[-1]
                e_str = "[" + " ".join(f"{v:.3f}" for v in log_norms[-1]) + "]"
            dW_str = "[" + " ".join(f"{v:.3f}" for v in upd_info["dW_norms"]) + "]"
            eta_str = "[" + " ".join(f"{v:.4f}" for v in eta_pl) + "]"
            print(
                f"[pcn] step {step:5d}/{n_steps}  t={elapsed:5.1f}s  "
                f"eta={eta_str}  nll={nll:.3f}  acc={acc:.3f}  "
                f"||e_l||={e_str}  ||dW_l||={dW_str}",
                flush=True,
            )

    telemetry["steps_completed"] = step + 1
    telemetry["duration_s"] = time.monotonic() - t0
    print(
        f"[pcn] end    steps={step+1}  duration={telemetry['duration_s']:.1f}s",
        flush=True,
    )
    return telemetry


# ---------------------------------------------------------------------------
# Streaming CharModel wrapper
# ---------------------------------------------------------------------------

class PCNByteCharModel(CharModel):
    """Streaming wrapper. Holds a rolling K-byte context buffer and runs
    a single PCN forward pass per `predict()` call. No inner-loop
    relaxation needed at eval — the unclamped top mu IS the distribution.
    """
    def __init__(
        self,
        state: PCNState,
        E: Tensor,
        K: int,
        device: torch.device,
    ):
        self.state = state
        self.E = E
        self.K = K
        self.device = device
        self.d_emb = E.size(1)
        # CPU-side rolling byte buffer. We only push to GPU once per
        # predict() to avoid per-observe synchronization overhead.
        self._buf = bytearray(K)
        self._head = 0   # number of bytes observed so far (capped at K)

    def reset(self) -> None:
        for i in range(self.K):
            self._buf[i] = 0
        self._head = 0

    @torch.no_grad()
    def predict(self) -> dict[str, float]:
        # Materialize current K-byte window on the device as int64.
        ctx = torch.tensor(self._buf, dtype=torch.long, device=self.device)
        x_bottom = _input_features(ctx.unsqueeze(0), self.E)
        xs = _pcn_forward_init(self.state, x_bottom)
        logits = xs[0][0]                  # (256,)
        probs = F.softmax(logits.float(), dim=-1)

        out: dict[str, float] = {}
        for byte_id, p in enumerate(probs.tolist()):
            try:
                ch = bytes([byte_id]).decode("utf-8")
            except UnicodeDecodeError:
                continue
            out[ch] = p
        return out

    def observe(self, char: str) -> None:
        for byte in char.encode("utf-8"):
            # In-place rotate left by one byte, then append at end.
            # bytearray slicing is fast and stays on CPU.
            self._buf[:-1] = self._buf[1:]
            self._buf[-1] = byte
            if self._head < self.K:
                self._head += 1


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def train(train_text: str, valid_text: str | None = None) -> CharModel:
    seed_env = os.environ.get("SEED")
    if seed_env:
        seed = int(seed_env)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        print(f"[pcn_byte_lm] SEED={seed}", flush=True)
    else:
        seed = 0

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[pcn_byte_lm] device={device}", flush=True)

    # --- Hyperparameters --------------------------------------------------
    K = 64                 # context window length (bytes)
    d_emb = 32             # frozen byte-embedding dim
    d_hidden = 768         # PCN hidden width
    # Layer dims, top -> bottom (math indexing):
    #   x_0 = top   (256)
    #   x_1, x_2, x_3 = hidden (d_hidden)
    #   x_4 = input (K * d_emb)
    layer_dims = [256, d_hidden, d_hidden, d_hidden, K * d_emb]

    batch_size = 256
    T_inner = 16              # more inner iters -> better fixed point on real data
    alpha = 0.1               # activity-relaxation step size (small = stable)
    # Per-layer LR. The top sees one-hot targets (large signal); hidden
    # layers see small tanh activations. Without per-layer eta, the top
    # explodes and hiddens starve. l = 0 is top.
    eta_top = 0.01
    eta_hidden = 0.15
    eta_warmup_steps = 300
    grad_clip = 0.5
    n_steps = 200_000         # ceiling well above achievable — wall-clock bounds
    max_seconds = 240.0       # leave ~60s slack for cold-start
    log_every = 500

    # --- Data -------------------------------------------------------------
    raw = train_text.encode("utf-8")
    train_bytes = torch.frombuffer(bytearray(raw), dtype=torch.uint8).to(device)
    n = train_bytes.numel()
    if n < K + 2:
        raise ValueError(f"need at least {K+2} bytes; got {n}")

    # --- Init -------------------------------------------------------------
    state = PCNState(layer_dims=layer_dims, device=device, seed=seed)
    E = _make_byte_embed(d_emb=d_emb, device=device, seed=seed + 7)

    n_params = sum(W.numel() + b.numel() for W, b in zip(state.W, state.b))
    print(
        f"[pcn_byte_lm] {n_params/1e6:.2f}M params  K={K} d_emb={d_emb} "
        f"d_hidden={d_hidden} layer_dims={layer_dims}",
        flush=True,
    )

    # Per-layer eta schedule: linear warmup then constant.
    # Wall-clock-bounded training: don't bake in step-based cooldown
    # because we don't know in advance how many steps complete.
    def eta_schedule(step: int) -> list[float]:
        wu = min(1.0, (step + 1) / eta_warmup_steps)
        # l=0 is top, l>=1 are hidden.
        out = [eta_top * wu]
        for _ in range(1, state.L):
            out.append(eta_hidden * wu)
        return out

    # --- Train ------------------------------------------------------------
    telemetry = _train_pcn(
        state, E, train_bytes,
        K=K,
        batch_size=batch_size,
        T_inner=T_inner,
        alpha=alpha,
        eta_schedule=eta_schedule,
        grad_clip=grad_clip,
        n_steps=n_steps,
        max_seconds=max_seconds,
        log_every=log_every,
    )

    print(
        f"[pcn_byte_lm] final  "
        f"steps={telemetry['steps_completed']}  "
        f"final_nll={telemetry['loss_history'][-1] if telemetry['loss_history'] else 'n/a'}  "
        f"final_e_norms={telemetry['e_norms_last']}",
        flush=True,
    )

    return PCNByteCharModel(state, E, K=K, device=device)
