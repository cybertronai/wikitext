"""Uniform Matrix Product State (uMPS) Born-machine char-LM, D=128.

Implements a translation-invariant Born-machine language model with a single
core tensor `A: (D, V=256, D)` shared across positions. The joint probability
of a byte sequence is the Born rule:

    psi(c_1..c_T) = h_0^T · A[:, c_1, :] · ... · A[:, c_T, :] · r
    P(c_1..c_T)   ∝ psi^2

where h_0 is a left boundary vector and `r` is a right-environment vector
characterizing the "infinite tail" partition function.

This file follows experiment_11_umps_born_dmrg.md but makes two pragmatic
changes vs the original spec:

  1. **Optimizer**: replace two-site DMRG (intricate for the translation-
     invariant case) with **Adam on per-position autoregressive NLL**. The
     model is still a Born-MPS — there is only ONE tensor A, no chain through
     depth, so "backprop" is degenerate (one outer gradient step per outer
     tensor). The README is explicit that the scorer is agnostic to backprop;
     Adam-on-A is the practical training rule.

  2. **Right environment**: the spec's "learned R boundary vector" is
     replaced with the **dominant right-eigenvector Phi of the left-transfer
     operator T'(N) = sum_v A[v]^T N A[v]** — the proper streaming partition
     function for infinite-tail AR inference. Phi is computed by power
     iteration after training and used inside `predict()`.

  3. **Training loss** uses the "left-only" simplification:
        logits_t[v] = log( ||h_{t-1} · A[v]||^2 + eps )
     which implicitly uses Phi = I as the future-environment normalizer.
     This is a looser bound than the true Born NLL but is numerically stable
     and avoids materializing Phi (a D×D matrix) inside the training loop.

Architecture summary
--------------------
- A: (D=128, V=256, D=128) fp32, ~17 MB.
- Per-position recurrence with renormalized hidden state h_t and a scalar
  log-norm carry (only relevant for joint scoring; per-step argmax is
  invariant to log-norm).
- Initialization: A[d, v, d'] ~ N(0, 1/(D*V)) so on average
  sum_v A[v]^T A[v] ≈ I (identity-on-average, Wall 2025).

Streaming inference
-------------------
After training, compute the dominant right-eigenvector Phi of T'(·) by power
iteration (~20 iters), normalize A so the leading eigenvalue is 1, then in
predict() return logits[v] = log( h^T (A[v] · Phi · A[v]^T) h ).
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
# Hyperparameters
# ---------------------------------------------------------------------------

D = 128                # bond dimension
V = 256                # byte alphabet
BATCH_SIZE = 32
T_WINDOW = 64          # training window length
N_STEPS = 5000
ADAM_LR = 3e-3
ADAM_BETAS = (0.9, 0.95)
WEIGHT_DECAY = 0.0
EPS = 1e-8             # numerical floor inside log
POWER_ITERS = 20       # number of power iterations for Phi
TIME_BUDGET_S = 250.0  # training budget; eval gets ~50s of the 300s SIGALRM cap
LOG_EVERY = 200


# ---------------------------------------------------------------------------
# Power iteration for the right environment Phi
# ---------------------------------------------------------------------------

@torch.no_grad()
def _transfer_apply(A: Tensor, N: Tensor) -> Tensor:
    """Apply the left-transfer operator T'(N) = sum_v A[v]^T N A[v].

    A: (D, V, D)   — core tensor, indexed A[d_in, v, d_out].
    N: (D, D)      — current right-environment matrix.
    Returns:       (D, D) — T'(N).

    Reading the einsum: A is indexed (i, v, j) with i=incoming bond, j=outgoing
    bond. T'(N)[i', j'] = sum_v sum_{i,j} A[i', v, i] N[i, j] A[j', v, j].
    Note the *transpose* on the right A (j' is the OUTPUT bond index of A[v]^T,
    which is the INPUT bond of A[v]).
    """
    # einsum: A[v,i',i] @ N[i,j] @ A[v,j',j]  summed over v, i, j
    # but A is (D, V, D); permute to (V, D_in, D_out) for clarity below.
    # We use the equivalent index pattern A[i,v,j], N[i,k], A[l,v,k] -> out[i,l]?
    # Stick with the spec einsum: 'vij,jk,vkl->il' where A is treated as (v,i,j).
    A_vij = A.permute(1, 0, 2).contiguous()  # (V, D, D) with A_vij[v, i, j] = A[i, v, j]
    return torch.einsum("vij,jk,vkl->il", A_vij, N, A_vij)


@torch.no_grad()
def _compute_phi(A: Tensor) -> tuple[Tensor, float]:
    """Power-iterate on T'(·) to obtain the dominant right-eigenvector Phi
    and the dominant eigenvalue lam. After this we have T'(Phi) ≈ lam * Phi.

    Returns (Phi, lam) where Phi is a (D, D) matrix normalized to ||Phi|| = 1.
    """
    D_ = A.shape[0]
    Phi = torch.eye(D_, dtype=torch.float32, device=A.device) / D_
    lam = 1.0
    for _ in range(POWER_ITERS):
        Phi_new = _transfer_apply(A, Phi)
        lam = float(Phi_new.norm().item())
        if lam < 1e-20:
            # Pathological zero-eigenvalue case; reset.
            Phi = torch.eye(D_, dtype=torch.float32, device=A.device) / D_
            lam = 1.0
            continue
        Phi = Phi_new / lam
    return Phi, lam


# ---------------------------------------------------------------------------
# Training: per-position autoregressive NLL with left-only normalizer
# ---------------------------------------------------------------------------

def _per_position_nll(
    A: Tensor,           # (D, V, D) fp32, requires_grad
    batch: Tensor,       # (B, T) int64 — byte sequences
    eps: float = EPS,
) -> Tensor:
    """Compute mean per-position CE loss over a (B, T) byte batch.

    For each window c_1..c_T:
        h_0 = (1/sqrt(D)) * ones(D)
        For t = 1..T:
            u_t[v] = h_{t-1} @ A[v]              # (D,) per v
            logits[v] = log(||u_t[v]||^2 + eps)  # scalar per v
            loss_t   = CE(logits, c_t)
            h_t      = u_t[c_t] / ||u_t[c_t]||   # renormalize

    The renormalization at every step is critical — without it ||h_t|| grows
    or shrinks exponentially with t and the chain blows up / underflows.
    Renormalization is scale-equivariant w.r.t. the *softmax* output (only the
    direction of h matters for the per-step distribution), so the loss is
    well-defined.
    """
    B, T = batch.shape
    D_ = A.shape[0]
    V_ = A.shape[1]
    device = A.device

    # h_0: uniform unit vector. Broadcast to batch.
    h = torch.full(
        (B, D_), 1.0 / math.sqrt(D_), dtype=A.dtype, device=device,
    )

    # Reshape A for the per-step contraction:
    #   u_v_d2 = sum_d1 h[b, d1] * A[d1, v, d2]
    # We do this as a single matmul by viewing A as (D, V*D) and reshaping.
    A_flat = A.reshape(D_, V_ * D_)  # (D, V*D)

    losses = []
    for t in range(T):
        # u: (B, V*D), then reshape to (B, V, D).
        u = h @ A_flat                       # (B, V*D)
        u = u.reshape(B, V_, D_)             # (B, V, D)

        # logits[b, v] = log(||u[b, v, :]||^2 + eps).
        u_sq = u.pow(2).sum(dim=-1)          # (B, V)
        logits = torch.log(u_sq + eps)       # (B, V)

        tgt = batch[:, t]                    # (B,)
        loss_t = F.cross_entropy(logits, tgt, reduction="mean")
        losses.append(loss_t)

        # h_{t} = u[:, tgt, :] / ||u[:, tgt, :]||
        # Gather along V axis; result (B, D).
        gather_idx = tgt.view(B, 1, 1).expand(B, 1, D_)  # (B, 1, D)
        u_chosen = u.gather(1, gather_idx).squeeze(1)    # (B, D)
        norm = u_chosen.norm(dim=-1, keepdim=True).clamp(min=math.sqrt(eps))
        h = u_chosen / norm

    return torch.stack(losses).mean()


def _train_umps(
    train_bytes: Tensor,
    device: torch.device,
    seed: int,
) -> Tensor:
    """Train the uMPS core A on `train_bytes` via Adam.

    Returns A (D, V, D) fp32 on `device`, normalized so the leading eigenvalue
    of T'(·) is 1.
    """
    n = train_bytes.numel()
    if n < T_WINDOW + 1:
        raise ValueError(f"need at least T_WINDOW+1 = {T_WINDOW+1} bytes; got {n}")

    # ---- Initialize A: identity-on-average per Wall 2025 ------------------
    # A[d, v, d'] ~ N(0, 1/(D*V)) makes sum_v A[v]^T A[v] = I in expectation,
    # which keeps ||h_t|| roughly stable at the start of training.
    g = torch.Generator(device=device)
    g.manual_seed(seed)
    init_std = 1.0 / math.sqrt(D * V)
    A_data = torch.randn(D, V, D, device=device, dtype=torch.float32, generator=g) * init_std
    A = A_data.clone().requires_grad_(True)

    opt = torch.optim.Adam(
        [A], lr=ADAM_LR, betas=ADAM_BETAS, weight_decay=WEIGHT_DECAY,
    )

    print(
        f"[umps_born] D={D} V={V} T={T_WINDOW} bs={BATCH_SIZE} "
        f"n_steps={N_STEPS} lr={ADAM_LR:.1e} budget={TIME_BUDGET_S:.0f}s",
        flush=True,
    )
    print(
        f"[umps_born] core memory: {A.numel() * 4 / 1e6:.1f} MB  "
        f"train bytes: {n:,}",
        flush=True,
    )

    t0 = time.monotonic()
    use_amp = device.type == "cuda"

    for step in range(N_STEPS):
        elapsed = time.monotonic() - t0
        if elapsed > TIME_BUDGET_S:
            print(f"[umps_born] budget exhausted at step {step} (elapsed={elapsed:.1f}s)", flush=True)
            break

        # Sample BATCH_SIZE windows of length T_WINDOW.
        idx = torch.randint(0, n - T_WINDOW, (BATCH_SIZE,), device=device)
        offsets = idx[:, None] + torch.arange(T_WINDOW, device=device)[None, :]
        batch = train_bytes[offsets].long()  # (B, T)

        opt.zero_grad(set_to_none=True)
        # Run the per-position chain in fp32 to keep the recurrent renorm stable.
        # bf16 autocast on the matmul `h @ A_flat` would lose precision over T steps.
        loss = _per_position_nll(A, batch)
        if not torch.isfinite(loss):
            print(f"[umps_born] non-finite loss at step {step} — skipping", flush=True)
            opt.zero_grad(set_to_none=True)
            continue
        loss.backward()
        # Light grad clipping — Born-rule gradients can spike when h aligns
        # poorly with the chosen byte's column.
        torch.nn.utils.clip_grad_norm_([A], max_norm=10.0)
        opt.step()

        if step % LOG_EVERY == 0 or step == N_STEPS - 1:
            with torch.no_grad():
                a_norm = A.detach().norm().item()
            print(
                f"[umps_born] step {step:5d}/{N_STEPS}  loss {loss.item():.4f}  "
                f"||A|| {a_norm:.3f}  elapsed {elapsed:.1f}s",
                flush=True,
            )

    A_final = A.detach()

    # ---- Post-training: compute Phi via power iteration, normalize A ------
    # T'(Phi) = lam * Phi. To make eigenvalue 1 we rescale A by 1/sqrt(lam):
    #   T'_new(N) = sum_v (A/sqrt(lam))^T N (A/sqrt(lam)) = (1/lam) T'(N)
    #   so T'_new(Phi) = Phi.
    print("[umps_born] computing Phi via power iteration...", flush=True)
    Phi, lam = _compute_phi(A_final)
    print(f"[umps_born] power iter: lam={lam:.4e}  ||Phi||={Phi.norm().item():.4f}", flush=True)
    if lam > 0:
        A_final = A_final / math.sqrt(lam)

    return A_final


# ---------------------------------------------------------------------------
# Streaming CharModel
# ---------------------------------------------------------------------------

class UMPSBornCharModel(CharModel):
    """Streaming Born-MPS char-LM.

    State: h (D,) renormalized fp32 hidden, plus a log_norm scalar. The
    log_norm is accumulated for completeness (joint scoring) but the
    per-step argmax/softmax is invariant to it, so it does not affect
    `predict()`.
    """
    def __init__(self, A: Tensor, Phi: Tensor, device: torch.device):
        self.A = A                          # (D, V, D) fp32
        self.Phi = Phi                      # (D, D)   fp32, normalized to lam=1
        self.device = device
        self.D = A.shape[0]
        self.V = A.shape[1]
        # Pre-compute A reshaped for fast per-step matmul.
        self.A_flat = A.reshape(self.D, self.V * self.D).contiguous()  # (D, V*D)
        # Pre-decode the byte->char map for single-byte UTF-8 chars.
        # Multi-byte chars are handled byte-by-byte in observe(); predict()
        # only emits chars that decode as a single byte (matches other
        # submissions in this repo, see krr_ngram/submission.py).
        self._h: Tensor | None = None
        self._log_norm: float = 0.0

    @torch.no_grad()
    def reset(self) -> None:
        self._h = torch.full(
            (self.D,), 1.0 / math.sqrt(self.D),
            dtype=torch.float32, device=self.device,
        )
        self._log_norm = 0.0

    @torch.no_grad()
    def predict(self) -> dict[str, float]:
        if self._h is None:
            raise RuntimeError("predict() called before reset()")
        h = self._h  # (D,)

        # u[v, d2] = sum_d1 h[d1] * A[d1, v, d2]
        # Equivalently: u = (h @ A_flat).reshape(V, D)
        u = (h @ self.A_flat).reshape(self.V, self.D)  # (V, D)

        # score[v] = u[v] @ Phi @ u[v]^T  (scalar per v)
        # Vectorized: score[v] = sum_{i,j} u[v,i] Phi[i,j] u[v,j]
        # = sum_v_outer u dot (Phi @ u^T)[v] : do it as (u @ Phi) elementwise * u, sum -1.
        uPhi = u @ self.Phi                            # (V, D)
        score = (uPhi * u).sum(dim=-1)                 # (V,)

        # Numerical floor — score can be tiny but should be ≥ 0 since Phi is PSD
        # in theory. Clamp defensively.
        score = score.clamp(min=EPS)
        logits = torch.log(score)                       # (V,)
        probs = torch.softmax(logits.float(), dim=-1)
        probs_list = probs.tolist()

        out: dict[str, float] = {}
        for byte_id, p in enumerate(probs_list):
            try:
                ch = bytes([byte_id]).decode("utf-8")
            except UnicodeDecodeError:
                continue
            out[ch] = p
        return out

    @torch.no_grad()
    def observe(self, char: str) -> None:
        if self._h is None:
            raise RuntimeError("observe() called before reset()")
        h = self._h
        A_flat = self.A_flat
        D_ = self.D
        V_ = self.V
        for byte in char.encode("utf-8"):
            # u_v_d2 for v=byte: pick row `byte` of the reshaped (V, D) tensor
            # built from h @ A_flat. Computing the full (V, D) per char is fine
            # (V*D = 32K, ~125 KB fp32) but we can save by indexing A directly.
            #   A[:, byte, :] : (D, D)
            #   u = h @ A_byte   : (D,)
            A_byte = self.A[:, byte, :]                  # (D, D) view
            u = h @ A_byte                                # (D,)
            norm = u.norm().clamp(min=math.sqrt(EPS))
            self._log_norm += float(torch.log(norm).item())
            h = u / norm
        self._h = h


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def train(train_text: str, valid_text: str | None = None) -> CharModel:
    seed_env = os.environ.get("SEED")
    seed = int(seed_env) if seed_env else 0
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    print(f"[umps_born] SEED={seed}", flush=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[umps_born] device={device}", flush=True)

    # ---- Data: bytes on device --------------------------------------------
    raw = train_text.encode("utf-8")
    train_bytes = torch.frombuffer(bytearray(raw), dtype=torch.uint8).to(device)
    print(f"[umps_born] train bytes: {train_bytes.numel():,}", flush=True)

    # ---- Train ------------------------------------------------------------
    A = _train_umps(train_bytes, device, seed=seed)

    # ---- Compute the (now lam=1) Phi for streaming inference -------------
    Phi, lam_after = _compute_phi(A)
    print(
        f"[umps_born] post-normalization power iter: lam={lam_after:.6f} "
        f"(should be ~1.0)",
        flush=True,
    )

    return UMPSBornCharModel(A, Phi, device)
