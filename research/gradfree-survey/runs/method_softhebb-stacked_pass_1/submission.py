"""SoftHebb Stacked Conv-1D + Ridge Readout (pass 1).

4-layer causal 1-D conv stack trained layer-wise by the SoftHebb soft-WTA
Hebbian rule on raw bytes, with a closed-form ridge readout to next-byte.

No backprop, no autograd anywhere in the conv stack — purely local Hebbian
updates with Oja anti-Hebbian normalization. Readout is closed-form ridge.

Spec: .survey/designs/method_softhebb-stacked_pass_1.md
"""
from __future__ import annotations

__author__ = "@survey-softhebb"

import os
import time

import torch
import torch.nn.functional as F
from torch import Tensor

from wikitext import CharModel


# Spec hyperparameters (do not vary)
CHANNELS = [384, 384, 512, 512]      # layer 1..4 output channels
KERNEL = 5
DILATIONS = [1, 2, 4, 8]
TAU = 1.0
ETAS = [0.02, 0.01, 0.01, 0.005]
M_CHARS_PER_LAYER = 100_000_000
BATCH_SIZE = 32
SEQ_LEN = 8192
READOUT_LAYERS = [1, 2, 3]            # layers 2,3,4 (0-indexed)
FEATURE_SUBSAMPLE_STRIDE = 8
N_FEATURE_CHARS = 40_000_000
RIDGE_LAMBDA_SCALE = 1e-2
C_IN = 256


# ---------------------------------------------------------------------------
# Helpers: causal conv + SoftHebb update
# ---------------------------------------------------------------------------

def causal_conv1d(x: Tensor, W: Tensor, dilation: int) -> Tensor:
    """Apply a causal 1-D conv. x: (B, C_in, L). W: (C_out, C_in, K).
    Output: (B, C_out, L) — same length, left-padded by (K-1)*dilation zeros.
    """
    K = W.shape[2]
    pad = (K - 1) * dilation
    x_pad = F.pad(x, (pad, 0))
    return F.conv1d(x_pad, W, dilation=dilation)


def normalize_features(z: Tensor, eps: float = 1e-6) -> Tensor:
    """L2-normalize across the channel axis (dim=1) for x of shape (B, C, L)."""
    n = z.norm(dim=1, keepdim=True).clamp_min(eps)
    return z / n


def init_softhebb_weights(c_out: int, c_in: int, K: int, device, generator) -> Tensor:
    """Initialize per-output-channel filter with unit L2 norm."""
    W = torch.randn(c_out, c_in, K, device=device, dtype=torch.float32, generator=generator)
    # Normalize each output filter to unit norm (over flatten of c_in*K).
    flat = W.view(c_out, -1)
    flat = flat / flat.norm(dim=1, keepdim=True).clamp_min(1e-6)
    return flat.view(c_out, c_in, K)


@torch.no_grad()
def softhebb_layer_train(
    layer_idx: int,
    W: Tensor,
    encode_input_fn,
    train_bytes: Tensor,
    eta: float,
    n_chars: int,
    batch_size: int,
    seq_len: int,
    dilation: int,
    tau: float,
    device,
    log_every: int = 50,
) -> Tensor:
    """Run a single SoftHebb sweep over the byte corpus for one layer.

    encode_input_fn(window_uint8) -> (B, C_in, L) input to this layer
    (raw byte one-hot for layer 0, or features from previous layers).
    """
    K = W.shape[2]
    c_out, c_in, _ = W.shape
    n_total = train_bytes.numel()

    # Window size: pull SEQ_LEN bytes; encode_input_fn turns them into
    # (B, C_in, SEQ_LEN) (the previous layers already pad causally so the
    # output is the same length as input).
    chars_per_step = batch_size * seq_len
    n_steps = max(1, n_chars // chars_per_step)
    t0 = time.monotonic()

    # Pre-pick all starting indices for this layer (deterministic, contiguous chunks).
    # Use sequential non-overlapping windows for simplicity; if we run out, wrap.
    for step in range(n_steps):
        # Pick batch_size random starts so we cover diverse regions.
        starts = torch.randint(
            0, max(1, n_total - seq_len - 1), (batch_size,), device=device,
        )
        offsets = starts[:, None] + torch.arange(seq_len, device=device)[None, :]
        window = train_bytes[offsets]  # (B, L) uint8

        x = encode_input_fn(window)    # (B, C_in, L) fp32

        u = causal_conv1d(x, W, dilation=dilation)  # (B, c_out, L)
        # SoftWTA across channels at each time
        y = torch.softmax(u / tau, dim=1)            # (B, c_out, L)

        # Hebbian update: dW[c,:,:] = sum_{b,t} y[b,c,t] * (x_win[b,t,:,:] - u[b,c,t] * W[c,:,:])
        # where x_win[b,t,:,:] is the (c_in, K) patch ending at time t (causal).
        #
        # Vectorized: extract patches via F.unfold-style indexing on padded x.
        pad = (K - 1) * dilation
        x_pad = F.pad(x, (pad, 0))                   # (B, c_in, L+pad)
        # Build patches: for each t in [0, L), we want
        # x_pad[:, :, t : t + (K-1)*dilation + 1 : dilation]  → (B, c_in, K)
        # F.unfold needs a 4-D input; reshape to (B, c_in, 1, L+pad).
        B, _, L_pad = x_pad.shape
        L = x.shape[2]
        patches = x_pad.unfold(2, (K - 1) * dilation + 1, 1)  # (B, c_in, L, eff_kernel)
        # Pick every `dilation`-th tap: index 0, dilation, 2*dilation, ...
        patches = patches[..., ::dilation]                    # (B, c_in, L, K)
        # patches[b, c_in, t, k] = input at b, c_in, t - (K-1-k)*dilation? Verify.
        # unfold gives windows of length eff_kernel starting at each position;
        # for causal padded x_pad with pad=(K-1)*dil zeros on left, the window
        # starting at t covers original positions [t-(K-1)*dil ... t], which
        # ends at original t — matches causal conv. Picking every `dilation`-th
        # tap gives dilation-spaced taps ending at t.

        # Hebbian term: sum_{b,t} y[b,c,t] * x_patch[b,t,:,:]
        # y: (B, c_out, L), patches: (B, c_in, L, K)
        # => (c_out, c_in, K) by einsum
        hebb = torch.einsum("bcl,bilk->cik", y, patches)

        # Oja anti-Hebb term: sum_{b,t} y[b,c,t] * u[b,c,t] * W[c,:,:]
        # scalar per (c) summed: s[c] = sum_{b,t} y[b,c,t] * u[b,c,t]
        s = (y * u).sum(dim=(0, 2))  # (c_out,)
        oja = s[:, None, None] * W

        dW = (hebb - oja) / (B * L)

        # Anneal eta linearly to 0
        cur_eta = eta * (1 - step / max(1, n_steps))
        W.add_(dW, alpha=cur_eta)

        # Periodic projection to unit norm to prevent drift.
        if step % 50 == 0 or step == n_steps - 1:
            flat = W.view(c_out, -1)
            norms = flat.norm(dim=1, keepdim=True).clamp_min(1e-6)
            flat.div_(norms)
            W.copy_(flat.view(c_out, c_in, K))

        if step % log_every == 0 or step == n_steps - 1:
            entropy = -(y.mean(dim=(0, 2)) * (y.mean(dim=(0, 2)).clamp_min(1e-12).log())).sum().item()
            elapsed = time.monotonic() - t0
            print(
                f"[softhebb L{layer_idx+1}] step {step:4d}/{n_steps}  "
                f"eta={cur_eta:.5f}  ch_entropy={entropy:.3f}/{torch.log(torch.tensor(c_out, dtype=torch.float32)).item():.3f}  "
                f"elapsed={elapsed:.1f}s",
                flush=True,
            )

    return W


# ---------------------------------------------------------------------------
# Stack of frozen conv layers — used for both training higher layers and
# for feature collection / streaming inference.
# ---------------------------------------------------------------------------

class SoftHebbStack:
    def __init__(self, weights: list[Tensor], dilations: list[int]):
        self.weights = weights
        self.dilations = dilations

    @torch.no_grad()
    def forward(self, x_bytes: Tensor) -> list[Tensor]:
        """x_bytes: (B, L) uint8. Returns list of layer activations
        (B, C_l, L), one per layer. Layer 0 input is byte one-hot."""
        x = byte_onehot(x_bytes)  # (B, 256, L)
        acts = []
        for W, dil in zip(self.weights, self.dilations):
            u = causal_conv1d(x, W, dilation=dil)
            y = torch.softmax(u / TAU, dim=1)
            # Normalize between layers (L2 over channels) so magnitudes don't
            # inherit unbounded growth.
            y_n = normalize_features(y)
            acts.append(y_n)
            x = y_n
        return acts


def byte_onehot(x_bytes: Tensor) -> Tensor:
    """(B, L) uint8 -> (B, 256, L) fp32 one-hot."""
    B, L = x_bytes.shape
    x = x_bytes.long()
    out = torch.zeros(B, C_IN, L, device=x_bytes.device, dtype=torch.float32)
    out.scatter_(1, x.unsqueeze(1), 1.0)
    return out


# ---------------------------------------------------------------------------
# Layer-wise SoftHebb training
# ---------------------------------------------------------------------------

@torch.no_grad()
def train_stack(train_bytes: Tensor, device, generator) -> SoftHebbStack:
    """Train 4 layers sequentially. Each layer trained on M_CHARS_PER_LAYER
    samples with its own eta, then frozen."""
    weights: list[Tensor] = []
    dilations = DILATIONS

    for li, (c_out, dil, eta) in enumerate(zip(CHANNELS, dilations, ETAS)):
        c_in = C_IN if li == 0 else CHANNELS[li - 1]
        W = init_softhebb_weights(c_out, c_in, KERNEL, device, generator)
        print(f"[softhebb] training layer {li+1}/4  c_in={c_in} c_out={c_out} dil={dil} eta={eta}", flush=True)

        # Build an encoder fn that runs frozen lower layers and returns
        # the input that layer li sees.
        frozen_weights = list(weights)  # copy
        frozen_dils = dilations[:li]

        def encode_fn(window_uint8: Tensor,
                      _weights=frozen_weights,
                      _dils=frozen_dils) -> Tensor:
            x = byte_onehot(window_uint8)  # (B, 256, L)
            for Wf, df in zip(_weights, _dils):
                u = causal_conv1d(x, Wf, dilation=df)
                y = torch.softmax(u / TAU, dim=1)
                x = normalize_features(y)
            return x

        W = softhebb_layer_train(
            li, W, encode_fn, train_bytes,
            eta=eta,
            n_chars=M_CHARS_PER_LAYER,
            batch_size=BATCH_SIZE,
            seq_len=SEQ_LEN,
            dilation=dil,
            tau=TAU,
            device=device,
        )
        weights.append(W)

    return SoftHebbStack(weights, dilations)


# ---------------------------------------------------------------------------
# Ridge readout
# ---------------------------------------------------------------------------

@torch.no_grad()
def fit_ridge_readout(
    stack: SoftHebbStack,
    train_bytes: Tensor,
    device,
    n_chars: int = N_FEATURE_CHARS,
    stride: int = FEATURE_SUBSAMPLE_STRIDE,
) -> tuple[Tensor, Tensor]:
    """Stream through stack, collect Phi (concat readout layers) and Y
    (next-byte one-hot). Solve ridge: W_out, b such that logits = W_out @ phi + b.

    Streaming: accumulate Phi^T Phi (D x D), Phi^T Y (D x 256), and the
    column sum of Phi (D,) and Y (256,) for the intercept.
    Returns: (W_out (D, 256), b (256,)).
    """
    D = sum(CHANNELS[li] for li in READOUT_LAYERS)
    print(f"[softhebb] fitting readout, D={D} features, target {n_chars} chars (stride {stride})", flush=True)

    PtP = torch.zeros(D, D, device=device, dtype=torch.float32)
    PtY = torch.zeros(D, C_IN, device=device, dtype=torch.float32)
    sum_P = torch.zeros(D, device=device, dtype=torch.float32)
    sum_Y = torch.zeros(C_IN, device=device, dtype=torch.float32)
    n_rows = 0

    n_total = train_bytes.numel()
    t0 = time.monotonic()
    chars_consumed = 0
    pos = 0
    step = 0
    while chars_consumed < n_chars and pos + BATCH_SIZE * SEQ_LEN + 1 < n_total:
        starts = pos + torch.arange(BATCH_SIZE, device=device) * SEQ_LEN
        offsets = starts[:, None] + torch.arange(SEQ_LEN + 1, device=device)[None, :]
        block = train_bytes[offsets]  # (B, L+1)
        window = block[:, :-1]        # (B, L)
        target_bytes = block[:, 1:]   # (B, L)

        acts = stack.forward(window)  # list of (B, C_l, L)
        feats = torch.cat([acts[li] for li in READOUT_LAYERS], dim=1)  # (B, D, L)

        # Subsample every `stride`-th timestep
        feats_sub = feats[:, :, ::stride]                # (B, D, L/stride)
        tgt_sub = target_bytes[:, ::stride]              # (B, L/stride)

        B, _, Ls = feats_sub.shape
        # Reshape to (B*Ls, D)
        Phi = feats_sub.permute(0, 2, 1).reshape(-1, D)  # (N_chunk, D)
        y_idx = tgt_sub.reshape(-1).long()               # (N_chunk,)

        # Build one-hot target (sparse-style accumulation)
        # Y is (N, 256); we don't want to materialize full Y, but
        # for PtY we can do: PtY[:, c] += sum over rows where y_idx==c of Phi.
        # Equivalently, scatter-add: PtY.index_add_(1, y_idx, Phi.T) but
        # index_add wants matching shape. Use scatter on a temporary:
        #   one_hot = F.one_hot(y_idx, num_classes=C_IN).float()  # (N, 256)
        #   PtY += Phi.T @ one_hot
        one_hot = F.one_hot(y_idx, num_classes=C_IN).to(torch.float32)
        PtP.addmm_(Phi.T, Phi)
        PtY.addmm_(Phi.T, one_hot)
        sum_P.add_(Phi.sum(dim=0))
        sum_Y.add_(one_hot.sum(dim=0))
        n_rows += Phi.shape[0]

        chars_consumed += B * SEQ_LEN
        pos += BATCH_SIZE * SEQ_LEN
        step += 1
        if step % 20 == 0:
            elapsed = time.monotonic() - t0
            print(
                f"[softhebb readout] {chars_consumed:,}/{n_chars:,} chars, "
                f"N_rows={n_rows:,}, elapsed={elapsed:.1f}s", flush=True,
            )

    # Center for intercept: solve as augmented linear system.
    # logits = W^T phi + b   where W has shape (D, 256).
    # Closed-form ridge with intercept: compute mean_phi, mean_y; then
    # W^T = (Phi_c^T Phi_c + lambda I)^-1 Phi_c^T Y_c
    # where Phi_c = Phi - mean_phi, Y_c = Y - mean_y.
    # In streaming form: PtP_c = PtP - N * mean_phi mean_phi^T
    #                    PtY_c = PtY - N * mean_phi mean_y^T
    N = float(n_rows)
    mean_phi = sum_P / N             # (D,)
    mean_y = sum_Y / N               # (256,)
    PtP_c = PtP - N * torch.outer(mean_phi, mean_phi)
    PtY_c = PtY - N * torch.outer(mean_phi, mean_y)

    lam = RIDGE_LAMBDA_SCALE * (torch.trace(PtP_c).item() / D)
    print(f"[softhebb readout] N={int(N):,}  D={D}  lambda={lam:.4e}", flush=True)
    PtP_c.diagonal().add_(lam)

    # Cholesky solve for W (D, 256). PtP_c @ W = PtY_c. With softmax-based
    # features the columns of Phi sum to a constant per timestep so the Gram
    # is rank-deficient up to numerical precision; if Cholesky fails, add a
    # small numerical jitter (proportional to trace) and retry once, then
    # fall back to lstsq.
    try:
        L_chol = torch.linalg.cholesky(PtP_c)
        W = torch.cholesky_solve(PtY_c, L_chol)         # (D, 256)
    except torch._C._LinAlgError:
        jitter = 1e-4 * (torch.trace(PtP_c).item() / D)
        print(f"[softhebb readout] Cholesky failed; adding jitter={jitter:.4e}", flush=True)
        PtP_c.diagonal().add_(jitter)
        try:
            L_chol = torch.linalg.cholesky(PtP_c)
            W = torch.cholesky_solve(PtY_c, L_chol)
        except torch._C._LinAlgError:
            print("[softhebb readout] Cholesky still failed; falling back to lstsq", flush=True)
            sol = torch.linalg.lstsq(PtP_c, PtY_c)
            W = sol.solution
    b = mean_y - W.T @ mean_phi                     # (256,)
    return W, b


# ---------------------------------------------------------------------------
# Streaming CharModel wrapper
# ---------------------------------------------------------------------------

class SoftHebbCharModel(CharModel):
    def __init__(
        self,
        stack: SoftHebbStack,
        W_out: Tensor,
        b_out: Tensor,
        device: torch.device,
    ):
        self.stack = stack
        self.W_out = W_out      # (D, 256)
        self.b_out = b_out      # (256,)
        self.device = device
        # Receptive field = 1 + sum_l (K-1)*dilation_l
        self.recep = 1 + sum((KERNEL - 1) * d for d in DILATIONS)
        self._buf: list[int] = []
        self._next_logits: Tensor | None = None

    @torch.no_grad()
    def reset(self) -> None:
        self._buf = []
        self._refresh_logits()

    @torch.no_grad()
    def _refresh_logits(self) -> None:
        # Build a uint8 tensor from the buffer (left-padded with zeros if short).
        if len(self._buf) == 0:
            window = torch.zeros(1, 1, device=self.device, dtype=torch.uint8)
        else:
            arr = torch.tensor(self._buf[-self.recep:], device=self.device, dtype=torch.uint8)
            window = arr.unsqueeze(0)  # (1, L)
        acts = self.stack.forward(window)  # list of (1, C_l, L)
        feats = torch.cat([acts[li][0, :, -1] for li in READOUT_LAYERS], dim=0)  # (D,)
        logits = self.W_out.T @ feats + self.b_out
        self._next_logits = logits

    @torch.no_grad()
    def predict(self) -> dict[str, float]:
        if self._next_logits is None:
            raise RuntimeError("predict() called before reset()")
        probs = F.softmax(self._next_logits.float(), dim=-1)
        out: dict[str, float] = {}
        for byte_id, p in enumerate(probs.tolist()):
            try:
                ch = bytes([byte_id]).decode("utf-8")
            except UnicodeDecodeError:
                continue
            out[ch] = p
        return out

    @torch.no_grad()
    def observe(self, char: str) -> None:
        for byte in char.encode("utf-8"):
            self._buf.append(int(byte))
            # Trim buffer to receptive field size for memory.
            if len(self._buf) > self.recep * 2:
                self._buf = self._buf[-self.recep:]
        self._refresh_logits()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def train(train_text: str, valid_text: str | None = None) -> CharModel:
    seed_env = os.environ.get("SEED")
    if seed_env:
        seed = int(seed_env)
    else:
        seed = 0
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    print(f"[softhebb] SEED={seed}", flush=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    generator = torch.Generator(device=device).manual_seed(seed)

    # Encode train text -> uint8 on GPU. Slice to first 100M chars per spec
    # (M_CHARS_PER_LAYER governs the per-layer sweep; we also use this slice
    # for readout collection).
    raw = train_text.encode("utf-8")
    train_bytes = torch.frombuffer(bytearray(raw), dtype=torch.uint8).to(device)
    n = train_bytes.numel()
    print(f"[softhebb] train bytes available: {n:,}", flush=True)

    # Train the 4-layer stack via SoftHebb local rule.
    stack = train_stack(train_bytes, device, generator)

    # Fit ridge readout.
    W_out, b_out = fit_ridge_readout(stack, train_bytes, device)

    print(
        f"[softhebb] done. receptive field = "
        f"{1 + sum((KERNEL - 1) * d for d in DILATIONS)} bytes",
        flush=True,
    )
    return SoftHebbCharModel(stack, W_out, b_out, device)
