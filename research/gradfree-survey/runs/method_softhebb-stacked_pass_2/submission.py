"""Shallow Wide Hebbian Patch Bank + Ridge — pass 2.

Direction B from the pass-2 design: one Hebbian patch-bank layer at
H=8192 width, soft-WTA, with a closed-form ridge readout for next-byte
prediction. Side-by-side frozen-Gaussian random-projection control on
the same ridge pipeline; both reported, Hebbian shipped.

Spec: .survey/designs/method_softhebb-stacked_pass_2.md
"""
from __future__ import annotations

__author__ = "@survey-softhebb-p2"

import math
import os
import time

import torch
import torch.nn.functional as F
from torch import Tensor

from wikitext import CharModel


# ---------------------------------------------------------------------------
# Spec hyperparameters (do not vary)
# ---------------------------------------------------------------------------
H = 8192                           # bank width
K = 16                             # patch length (bytes)
C_IN = 256                         # byte alphabet
D_IN = C_IN * K                    # 4096
TAU = 0.5                          # soft-WTA temperature
ETA = 0.01                         # Hebbian learning rate (annealed to 0)
M_TRAIN_CHARS = 200_000_000        # Hebbian sweep budget
BATCH_SIZE = 32                    # locked plan (spec sec 5)
SEQ_LEN = 4096                     # window length
STRIDE_TRAIN = 8                   # subsample stride during Hebb training
N_FEATURE_CHARS = 30_000_000       # ridge feature collection
STRIDE_FEAT = 8                    # subsample stride for ridge features
RIDGE_LAMBDA_SCALE = 1e-2          # lambda = 1e-2 * trace(Gram) / H
RENORM_EVERY = 50                  # row-renormalize W every N steps
ENTROPY_CHECK_STEP = 100           # one-shot entropy diagnostic step
ENTROPY_THRESHOLD = 0.3            # fraction of log(H)
TAU_BUMP = 1.5                     # one-time tau multiplier if entropy collapses


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def byte_onehot_patches(bytes_2d: Tensor, stride: int = 1) -> Tensor:
    """(B, L) uint8 -> (B, ceil((L-K+1)/stride), K * 256) fp32 patches.

    Each row is the flattened one-hot of K consecutive bytes. We index
    bytes first (uint8) and one-hot afterwards to keep the intermediate
    tensor small — materializing a (B, L, 256) one-hot before unfold
    would cost ~128 MB per batch and the unfolded reshape would be ~2 GB.
    """
    B, L = bytes_2d.shape
    # Build patches of uint8 byte indices: (B, n_patches, K).
    byte_patches = bytes_2d.unfold(1, K, 1)  # (B, L-K+1, K) -- view
    if stride > 1:
        byte_patches = byte_patches[:, ::stride, :].contiguous()
    n_patches = byte_patches.shape[1]
    # One-hot of byte indices and flatten K dim into the channel axis.
    oh = F.one_hot(byte_patches.long(), num_classes=C_IN).to(torch.float32)
    # oh: (B, n_patches, K, 256)
    return oh.reshape(B, n_patches, K * C_IN)


@torch.no_grad()
def softhebb_train(
    train_bytes: Tensor,
    device: torch.device,
    generator: torch.Generator,
) -> tuple[Tensor, float]:
    """Train the H x D_IN Hebbian bank on byte patches.

    Returns (W, tau_final) where tau_final is TAU possibly bumped once
    if an entropy-collapse was detected at the diagnostic step.
    """
    W = torch.randn(H, D_IN, device=device, dtype=torch.float32, generator=generator)
    W = W / W.norm(dim=1, keepdim=True).clamp_min(1e-6)

    n_total = train_bytes.numel()
    chars_per_step = BATCH_SIZE * SEQ_LEN
    n_steps = max(1, M_TRAIN_CHARS // chars_per_step)
    tau = TAU
    tau_bumped = False

    t0 = time.monotonic()
    for step in range(n_steps):
        starts = torch.randint(
            0, max(1, n_total - SEQ_LEN - 1), (BATCH_SIZE,),
            device=device, generator=generator,
        )
        offsets = starts[:, None] + torch.arange(SEQ_LEN, device=device)[None, :]
        window = train_bytes[offsets]  # (B, L)

        patches = byte_onehot_patches(window, stride=STRIDE_TRAIN)  # (B, Ns, D_IN)
        x = patches.reshape(-1, D_IN)                                # (N, D_IN)

        u = x @ W.T                                    # (N, H)
        y = torch.softmax(u / tau, dim=1)              # (N, H)

        # SoftHebb / Oja update: dW = (y^T @ (x - y@W)) / N
        N_rows = x.shape[0]
        residual = x - y @ W                           # (N, D_IN)
        dW = (y.T @ residual) / N_rows                 # (H, D_IN)

        cur_eta = ETA * (1.0 - step / max(1, n_steps))
        W.add_(dW, alpha=cur_eta)

        if step % RENORM_EVERY == 0 or step == n_steps - 1:
            norms = W.norm(dim=1, keepdim=True).clamp_min(1e-6)
            W.div_(norms)

        if step == ENTROPY_CHECK_STEP and not tau_bumped:
            ch_mean = y.mean(dim=0)
            ch_p = ch_mean / ch_mean.sum().clamp_min(1e-12)
            entropy = -(ch_p * ch_p.clamp_min(1e-12).log()).sum().item()
            log_H = math.log(H)
            if entropy < ENTROPY_THRESHOLD * log_H:
                tau *= TAU_BUMP
                tau_bumped = True
                print(
                    f"[hebb] entropy collapse at step {step}: {entropy:.3f} < "
                    f"{ENTROPY_THRESHOLD * log_H:.3f}; bumping tau -> {tau:.3f}",
                    flush=True,
                )

        if step % 50 == 0 or step == n_steps - 1:
            ch_mean = y.mean(dim=0)
            ch_p = ch_mean / ch_mean.sum().clamp_min(1e-12)
            entropy = -(ch_p * ch_p.clamp_min(1e-12).log()).sum().item()
            elapsed = time.monotonic() - t0
            print(
                f"[hebb] step {step:4d}/{n_steps}  eta={cur_eta:.5f}  tau={tau:.3f}  "
                f"ch_entropy={entropy:.3f}/{math.log(H):.3f}  N_rows={N_rows}  "
                f"elapsed={elapsed:.1f}s",
                flush=True,
            )

    return W, tau


@torch.no_grad()
def fit_ridge(
    W: Tensor,
    tau_used: float,
    train_bytes: Tensor,
    device: torch.device,
    n_chars: int = N_FEATURE_CHARS,
    stride: int = STRIDE_FEAT,
    tag: str = "ridge",
) -> tuple[Tensor, Tensor]:
    """Stream byte patches through the bank, accumulate Phi^T Phi and
    Phi^T Y in streaming form, Cholesky-solve for the readout W_out
    (H, 256) and bias b (256,). Returns (W_out, b).

    Mapping convention: a patch covering window-bytes [t-K+1, t]
    predicts the byte at position t+1.
    """
    PtP = torch.zeros(H, H, device=device, dtype=torch.float32)
    PtY = torch.zeros(H, C_IN, device=device, dtype=torch.float32)
    sum_P = torch.zeros(H, device=device, dtype=torch.float32)
    sum_Y = torch.zeros(C_IN, device=device, dtype=torch.float32)
    n_rows = 0

    n_total = train_bytes.numel()
    chars_per_step = BATCH_SIZE * SEQ_LEN
    chars_consumed = 0
    pos = 0
    step = 0
    t0 = time.monotonic()

    while chars_consumed < n_chars and pos + chars_per_step + 1 < n_total:
        starts = pos + torch.arange(BATCH_SIZE, device=device) * SEQ_LEN
        offsets = starts[:, None] + torch.arange(SEQ_LEN + 1, device=device)[None, :]
        block = train_bytes[offsets]   # (B, L+1)
        window = block[:, :-1]         # (B, L)
        # Patches end at positions K-1..L-1 inside `window`; predict block[:, K..L].
        target_bytes = block[:, K:]    # (B, L - K + 1)

        patches_sub = byte_onehot_patches(window, stride=stride)  # (B, Ls, D_IN)
        tgt_sub = target_bytes[:, ::stride]

        B, Ls, _ = patches_sub.shape
        x = patches_sub.reshape(-1, D_IN)
        u = x @ W.T
        phi = torch.softmax(u / tau_used, dim=1)   # (N, H)

        y_idx = tgt_sub.reshape(-1).long()
        one_hot = F.one_hot(y_idx, num_classes=C_IN).to(torch.float32)

        PtP.addmm_(phi.T, phi)
        PtY.addmm_(phi.T, one_hot)
        sum_P.add_(phi.sum(dim=0))
        sum_Y.add_(one_hot.sum(dim=0))
        n_rows += phi.shape[0]

        chars_consumed += B * SEQ_LEN
        pos += BATCH_SIZE * SEQ_LEN
        step += 1
        if step % 10 == 0:
            elapsed = time.monotonic() - t0
            print(
                f"[{tag}] {chars_consumed:,}/{n_chars:,} chars, "
                f"N_rows={n_rows:,}, elapsed={elapsed:.1f}s", flush=True,
            )

    N = float(n_rows)
    mean_phi = sum_P / N
    mean_y = sum_Y / N
    PtP_c = PtP - N * torch.outer(mean_phi, mean_phi)
    PtY_c = PtY - N * torch.outer(mean_phi, mean_y)

    trace_per_dim = torch.trace(PtP_c).item() / H
    lam = RIDGE_LAMBDA_SCALE * trace_per_dim
    print(f"[{tag}] N={int(N):,}  H={H}  lambda={lam:.4e}", flush=True)
    PtP_c.diagonal().add_(lam)

    try:
        L_chol = torch.linalg.cholesky(PtP_c)
        W_out = torch.cholesky_solve(PtY_c, L_chol)
    except Exception as e:
        jitter = 1e-4 * trace_per_dim
        print(f"[{tag}] cholesky failed ({e!r}); adding jitter={jitter:.4e}", flush=True)
        PtP_c.diagonal().add_(jitter)
        try:
            L_chol = torch.linalg.cholesky(PtP_c)
            W_out = torch.cholesky_solve(PtY_c, L_chol)
        except Exception:
            print(f"[{tag}] cholesky still failed; falling back to lstsq", flush=True)
            sol = torch.linalg.lstsq(PtP_c, PtY_c)
            W_out = sol.solution

    b = mean_y - W_out.T @ mean_phi
    return W_out, b


@torch.no_grad()
def quick_eval(
    W: Tensor,
    tau_used: float,
    W_out: Tensor,
    b: Tensor,
    valid_bytes: Tensor,
    device: torch.device,
    max_chars: int = 60_000,
    tag: str = "eval",
) -> float:
    """Greedy-argmax char accuracy on a prefix of validation bytes.

    For each position t (t >= K), form the patch over bytes[t-K..t-1] and
    predict bytes[t]. Counts only positions that have a full patch.
    """
    n = min(valid_bytes.numel(), max_chars + K)
    bs = valid_bytes[:n]
    if n <= K:
        return 0.0
    oh = F.one_hot(bs.long(), num_classes=C_IN).to(torch.float32)  # (n, 256)
    patches = oh.unfold(0, K, 1)                                    # (n-K+1, 256, K)
    patches = patches.reshape(-1, C_IN * K)                          # (n-K+1, D_IN)
    # Patch index i ends at position i+K-1, predicts byte at i+K.
    # Drop the last row (no target).
    n_pred = patches.shape[0] - 1
    if n_pred <= 0:
        return 0.0
    x = patches[:n_pred]
    targets = bs[K : K + n_pred].long()

    correct = 0
    total = 0
    chunk = 16384
    for i in range(0, n_pred, chunk):
        xb = x[i : i + chunk]
        u = xb @ W.T
        phi = torch.softmax(u / tau_used, dim=1)
        logits = phi @ W_out + b
        pred = logits.argmax(dim=1)
        correct += (pred == targets[i : i + chunk]).sum().item()
        total += xb.shape[0]
    acc = correct / max(1, total)
    print(f"[{tag}] quick-eval char-acc on {total} chars: {acc:.4f}", flush=True)
    return acc


# ---------------------------------------------------------------------------
# Streaming CharModel
# ---------------------------------------------------------------------------

class ShallowHebbCharModel(CharModel):
    def __init__(
        self,
        W: Tensor,
        tau_used: float,
        W_out: Tensor,
        b: Tensor,
        device: torch.device,
    ):
        self.W = W
        self.tau_used = tau_used
        self.W_out = W_out
        self.b = b
        self.device = device
        self._buf: list[int] = []

    @torch.no_grad()
    def reset(self) -> None:
        self._buf = []

    @torch.no_grad()
    def _build_patch(self) -> Tensor:
        if len(self._buf) >= K:
            arr = self._buf[-K:]
        else:
            arr = [0] * (K - len(self._buf)) + self._buf
        idx = torch.tensor(arr, device=self.device, dtype=torch.long)
        oh = F.one_hot(idx, num_classes=C_IN).to(torch.float32)
        return oh.reshape(K * C_IN)

    @torch.no_grad()
    def predict(self) -> dict[str, float]:
        x = self._build_patch()
        u = x @ self.W.T                          # (H,)
        phi = torch.softmax(u / self.tau_used, dim=0)
        logits = self.W_out.T @ phi + self.b      # (256,)
        probs = F.softmax(logits.float(), dim=-1)
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
            if len(self._buf) > 4 * K:
                self._buf = self._buf[-K:]


# ---------------------------------------------------------------------------
# train()
# ---------------------------------------------------------------------------

def train(train_text: str, valid_text: str | None = None) -> CharModel:
    t_start = time.monotonic()
    seed_env = os.environ.get("SEED")
    seed = int(seed_env) if seed_env else 0
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    print(f"[hebb-p2] SEED={seed}", flush=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    generator = torch.Generator(device=device).manual_seed(seed)

    raw = train_text.encode("utf-8")
    train_bytes = torch.frombuffer(bytearray(raw), dtype=torch.uint8).to(device)
    n = train_bytes.numel()
    print(f"[hebb-p2] train bytes available: {n:,}", flush=True)

    valid_bytes = None
    if valid_text is not None:
        vraw = valid_text.encode("utf-8")
        valid_bytes = torch.frombuffer(bytearray(vraw), dtype=torch.uint8).to(device)

    # ---- Hebbian variant ----
    print("[hebb-p2] === Hebbian variant ===", flush=True)
    t0 = time.monotonic()
    W_hebb, tau_hebb = softhebb_train(train_bytes, device, generator)
    print(f"[hebb-p2] hebbian train done in {time.monotonic() - t0:.1f}s (tau={tau_hebb})", flush=True)

    t0 = time.monotonic()
    W_out_hebb, b_hebb = fit_ridge(W_hebb, tau_hebb, train_bytes, device, tag="hebb-ridge")
    print(f"[hebb-p2] hebbian ridge done in {time.monotonic() - t0:.1f}s", flush=True)

    acc_hebb = -1.0
    if valid_bytes is not None:
        acc_hebb = quick_eval(W_hebb, tau_hebb, W_out_hebb, b_hebb, valid_bytes, device, tag="hebb-eval")

    # ---- Random-projection control ----
    print("[hebb-p2] === Random-projection control ===", flush=True)
    t0 = time.monotonic()
    W_rand = torch.randn(H, D_IN, device=device, dtype=torch.float32, generator=generator) / math.sqrt(D_IN)
    tau_rand = TAU
    W_out_rand, b_rand = fit_ridge(W_rand, tau_rand, train_bytes, device, tag="rand-ridge")
    print(f"[hebb-p2] random-control ridge done in {time.monotonic() - t0:.1f}s", flush=True)

    acc_rand = -1.0
    if valid_bytes is not None:
        acc_rand = quick_eval(W_rand, tau_rand, W_out_rand, b_rand, valid_bytes, device, tag="rand-eval")

    delta = acc_hebb - acc_rand
    elapsed = time.monotonic() - t_start
    print(
        f"[hebb-p2] SUMMARY  hebb_val_acc={acc_hebb:.4f}  "
        f"rand_val_acc={acc_rand:.4f}  delta={delta:+.4f}  "
        f"train_time={elapsed:.1f}s",
        flush=True,
    )

    # Ship Hebbian (primary variant per spec).
    return ShallowHebbCharModel(W_hebb, tau_hebb, W_out_hebb, b_hebb, device)
