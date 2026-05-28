"""Nystrom KRR hybrid: NN backbone (phase 1, gradient-based) + closed-form
Nystrom kernel ridge regression head (phase 2).

Per experiments/kernel_methods/experiment_07_falkon_krr_learned_embedding.md.

Phase 1 (~120s wall, gradient-based): small transformer encoder 4L/256d with a
temporary linear LM head trained with AdamW + cross-entropy on next-byte. The
KV-cached forward signature is copied verbatim from submissions/modded_nanogpt.

Phase 2 (<30s, closed-form): sample N=100K context windows, encode the last
token's residual with the frozen phase-1 backbone, L2-normalize -> e_i in R^256.
Build one-hot Y in R^(N x 256). Hand-rolled Nystrom KRR with cosine kernel
(linear on normalized inputs), M=1024 random landmarks, penalty 1e-3. The
phase-1 LM head is discarded — at inference time predict() does
encoder forward -> L2-normalize -> K_query @ alpha -> softmax.
"""
from __future__ import annotations

__author__ = "@ab-10"

import os
import time

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.optim import AdamW

from wikitext import CharModel


# ---------------------------------------------------------------------------
# Architecture — small encoder (4L/256d), KV-cached, RoPE offset support.
# Copied from submissions/modded_nanogpt/submission.py, smaller config.
# ---------------------------------------------------------------------------

class RMSNorm(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.gains = nn.Parameter(torch.ones(dim))

    def forward(self, x: Tensor) -> Tensor:
        return F.rms_norm(x, (x.size(-1),), weight=self.gains.type_as(x))


class Linear(nn.Linear):
    def __init__(self, in_features: int, out_features: int):
        super().__init__(in_features, out_features, bias=True)

    def forward(self, x: Tensor) -> Tensor:
        return F.linear(x, self.weight.type_as(x), self.bias.type_as(x))


class Rotary(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        angular_freq = (1 / 1024) ** torch.linspace(0, 1, steps=dim // 4, dtype=torch.float32)
        self.register_buffer(
            "angular_freq",
            torch.cat([angular_freq, angular_freq.new_zeros(dim // 4)]),
        )

    def forward(self, x_BTHD: Tensor, offset: int = 0) -> Tensor:
        T = x_BTHD.size(1)
        pos = torch.arange(T, dtype=torch.float32, device=x_BTHD.device) + offset
        theta = torch.outer(pos, self.angular_freq)[None, :, None, :]
        cos, sin = theta.cos(), theta.sin()
        x1, x2 = x_BTHD.to(dtype=torch.float32).chunk(2, dim=-1)
        y1 = x1 * cos + x2 * sin
        y2 = x1 * (-sin) + x2 * cos
        return torch.cat((y1, y2), 3).type_as(x_BTHD)


class CausalSelfAttention(nn.Module):
    def __init__(self, dim: int, head_dim: int = 64):
        super().__init__()
        self.num_heads = dim // head_dim
        self.head_dim = head_dim
        hdim = self.num_heads * self.head_dim
        self.q = Linear(dim, hdim)
        self.k = Linear(dim, hdim)
        self.v = Linear(dim, hdim)
        self.proj = Linear(hdim, dim)
        self.rotary = Rotary(head_dim)

    def forward(
        self,
        x: Tensor,
        kv_cache: tuple[Tensor, Tensor] | None = None,
        offset: int = 0,
    ) -> tuple[Tensor, tuple[Tensor, Tensor]]:
        B, T = x.size(0), x.size(1)
        q = self.q(x).view(B, T, self.num_heads, self.head_dim)
        k = self.k(x).view(B, T, self.num_heads, self.head_dim)
        v = self.v(x).view(B, T, self.num_heads, self.head_dim)
        q = F.rms_norm(q, (q.size(-1),))
        k = F.rms_norm(k, (k.size(-1),))
        q = self.rotary(q, offset=offset)
        k = self.rotary(k, offset=offset)

        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        if kv_cache is not None:
            k_cache, v_cache = kv_cache
            k = torch.cat([k_cache, k], dim=2)
            v = torch.cat([v_cache, v], dim=2)

        is_causal = (kv_cache is None) and T > 1
        y = F.scaled_dot_product_attention(q, k, v, scale=0.12, is_causal=is_causal)
        y = y.transpose(1, 2).contiguous().view(B, T, self.num_heads * self.head_dim)
        return self.proj(y), (k, v)


class MLP(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        hdim = 4 * dim
        self.fc = Linear(dim, hdim)
        self.proj = Linear(hdim, dim)

    def forward(self, x: Tensor) -> Tensor:
        x = self.fc(x)
        x = x.relu().square()
        x = self.proj(x)
        return x


class Block(nn.Module):
    def __init__(self, dim: int, head_dim: int):
        super().__init__()
        self.attn = CausalSelfAttention(dim, head_dim=head_dim)
        self.mlp = MLP(dim)
        self.norm1 = RMSNorm(dim)
        self.norm2 = RMSNorm(dim)

    def forward(
        self,
        x: Tensor,
        kv_cache: tuple[Tensor, Tensor] | None = None,
        offset: int = 0,
    ) -> tuple[Tensor, tuple[Tensor, Tensor]]:
        h, new_kv = self.attn(self.norm1(x), kv_cache, offset=offset)
        x = x + h
        x = x + self.mlp(self.norm2(x))
        return x, new_kv


class Encoder(nn.Module):
    """Phase-1 encoder. Exposes both `encode` (returns the post-norm residual,
    used for KRR features) and `forward_logits` (encoder + temporary LM head,
    used during phase-1 supervised training)."""

    def __init__(
        self,
        vocab_size: int,
        num_layers: int,
        model_dim: int,
        head_dim: int = 64,
        max_len: int = 512,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.max_len = max_len
        self.model_dim = model_dim
        self.embed = nn.Embedding(vocab_size, model_dim).bfloat16()
        self.blocks = nn.ModuleList(
            [Block(model_dim, head_dim=head_dim) for _ in range(num_layers)]
        )
        self.norm1 = RMSNorm(model_dim)
        self.norm2 = RMSNorm(model_dim)
        # Temporary phase-1 LM head; discarded after phase 1.
        self.proj = Linear(model_dim, vocab_size)

    def encode(
        self,
        inputs: Tensor,
        kv_caches: list[tuple[Tensor, Tensor]] | None = None,
        offset: int = 0,
    ) -> tuple[Tensor, list[tuple[Tensor, Tensor]]]:
        """Return post-norm residual stream e in R^(B, T, d) and KV caches.
        No LM head applied — this is what we feed into KRR."""
        x = self.norm1(self.embed(inputs))
        new_caches: list[tuple[Tensor, Tensor]] = []
        for i, block in enumerate(self.blocks):
            kv = kv_caches[i] if kv_caches is not None else None
            x, new_kv = block(x, kv, offset=offset)
            new_caches.append(new_kv)
        e = self.norm2(x)
        return e, new_caches

    def forward_logits(
        self,
        inputs: Tensor,
        kv_caches: list[tuple[Tensor, Tensor]] | None = None,
        offset: int = 0,
    ) -> tuple[Tensor, list[tuple[Tensor, Tensor]]]:
        e, new_caches = self.encode(inputs, kv_caches, offset=offset)
        logits = self.proj(e).float()
        logits = 15 * logits * (logits.square() + 15 ** 2).rsqrt()
        return logits, new_caches


# ---------------------------------------------------------------------------
# Init scheme (mirrors modded-nanogpt simple).
# ---------------------------------------------------------------------------

def _init_modded(model: Encoder) -> None:
    for name, p in model.named_parameters():
        w = p.data
        if name.endswith("weight"):
            if "proj" in name:
                w.zero_()
            elif "embed" in name:
                w.normal_()
            else:
                w.normal_(std=0.33 ** 0.5 / w.size(-1) ** 0.5)
        elif name.endswith("bias"):
            w.zero_()
        elif name.endswith("gains"):
            w.normal_(mean=1, std=0)
        else:
            raise RuntimeError(f"Uninitialized parameter: {name}")


# ---------------------------------------------------------------------------
# Hand-rolled Nystrom KRR (verbatim from experiment_07 spec).
# ---------------------------------------------------------------------------

def nystrom_krr_fit(
    E: Tensor,             # (N, d), L2-normalized
    Y: Tensor,             # (N, C), one-hot or smoothed
    M: int = 1024,
    penalty: float = 1e-3,
    landmark_idx: Tensor | None = None,
) -> tuple[Tensor, Tensor]:
    """Closed-form Nystrom KRR with cosine kernel (inputs assumed L2-
    normalized, so cosine kernel == linear kernel). Solve
        (K_MM + (lambda / N) K_MN K_MN.T) alpha = (1/N) K_MN Y.
    Returns (landmarks Z in R^(M x d), alpha in R^(M x C))."""
    N, _ = E.shape
    if landmark_idx is None:
        landmark_idx = torch.randperm(N, device=E.device)[:M]
    Z = E[landmark_idx]
    Z32 = Z.float()
    E32 = E.float()
    K_MM = Z32 @ Z32.t()                                 # (M, M)
    K_MN = Z32 @ E32.t()                                 # (M, N)
    A = K_MM + (penalty / N) * (K_MN @ K_MN.t())         # (M, M) PSD
    A.diagonal().add_(1e-6)                              # numerical jitter
    rhs = (K_MN @ Y.float()) / N                         # (M, C)
    try:
        L = torch.linalg.cholesky(A)
        alpha = torch.cholesky_solve(rhs, L)
    except Exception as e:
        # Fall back to LU on the rare case Cholesky fails.
        print(f"[krr] cholesky failed ({e}); falling back to solve", flush=True)
        A.diagonal().add_(1e-4)
        alpha = torch.linalg.solve(A, rhs)
    return Z, alpha


def nystrom_krr_predict(
    e_query: Tensor,       # (B, d), L2-normalized
    Z: Tensor,             # (M, d), L2-normalized
    alpha: Tensor,         # (M, C)
) -> Tensor:
    """Return logits in R^(B, C). Caller applies softmax."""
    K_query = e_query.float() @ Z.t().float()            # (B, M)
    return K_query @ alpha                                # (B, C)


# ---------------------------------------------------------------------------
# Phase 1: time-budgeted gradient training of the encoder + temp LM head.
# ---------------------------------------------------------------------------

class TrainConfig:
    """Phase-1 training config. Smaller than modded-nanogpt (4L/256d vs 6L/384d)
    to leave wall-time for phase-2 KRR fit + eval."""

    def __init__(
        self,
        model_dim: int = 256,
        num_layers: int = 4,
        head_dim: int = 64,
        max_len: int = 512,
        batch_size: int = 64,
        phase1_seconds: float = 120.0,
        cooldown_frac: float = 0.4,
        embed_lr: float = 0.3,
        head_lr: float = 1.0 / 320,
        scalar_lr: float = 0.01,
        block_lr: float = 3e-3,
        block_wd: float = 0.01,
        log_every: int = 50,
    ):
        self.model_dim = model_dim
        self.num_layers = num_layers
        self.head_dim = head_dim
        self.max_len = max_len
        self.batch_size = batch_size
        self.phase1_seconds = phase1_seconds
        self.cooldown_frac = cooldown_frac
        self.embed_lr = embed_lr
        self.head_lr = head_lr
        self.scalar_lr = scalar_lr
        self.block_lr = block_lr
        self.block_wd = block_wd
        self.log_every = log_every

    def __repr__(self):
        return (f"TrainConfig(d={self.model_dim} L={self.num_layers} "
                f"H={self.model_dim // self.head_dim} bs={self.batch_size} "
                f"T={self.max_len} phase1={self.phase1_seconds:.0f}s)")


def _train_phase1(
    train_bytes: Tensor,
    cfg: TrainConfig,
    device: torch.device,
) -> Encoder:
    n = train_bytes.numel()
    model = Encoder(
        vocab_size=256,
        num_layers=cfg.num_layers,
        model_dim=cfg.model_dim,
        head_dim=cfg.head_dim,
        max_len=cfg.max_len,
    ).to(device)
    _init_modded(model)

    # AdamW for everything (no Muon — keeps phase-1 simple and reliable).
    # Two param groups: embeddings/head/scalars at one lr, block weights at another.
    embed_head = [model.embed.weight, model.proj.weight]
    scalars = [p for p in model.parameters() if p.ndim < 2]
    block_2d = [p for p in model.blocks.parameters() if p.ndim >= 2]

    optimizer = AdamW(
        [
            dict(params=[model.embed.weight], lr=cfg.embed_lr, weight_decay=0.0),
            dict(params=[model.proj.weight], lr=cfg.head_lr, weight_decay=0.0),
            dict(params=scalars, lr=cfg.scalar_lr, weight_decay=0.0),
            dict(params=block_2d, lr=cfg.block_lr, weight_decay=cfg.block_wd),
        ],
        betas=(0.9, 0.95),
        eps=1e-10,
        fused=(device.type == "cuda"),
    )
    for g in optimizer.param_groups:
        g["initial_lr"] = g["lr"]

    n_params = sum(p.numel() for p in model.parameters())
    print(f"[nystrom_krr] phase1 {n_params/1e6:.2f}M params  cfg={cfg}", flush=True)

    # Heuristic: assume ~25 ms/step at this size on A100 -> ~4800 steps in 120s.
    # We don't actually need a precise n_steps; cosine LR uses elapsed/budget.
    model.train()
    use_amp = device.type == "cuda"
    t0 = time.monotonic()
    step = 0
    while True:
        elapsed = time.monotonic() - t0
        progress = elapsed / cfg.phase1_seconds
        if progress >= 1.0:
            break
        # Linear warmup-then-cooldown (same shape as modded-nanogpt simple).
        if progress < 1 - cfg.cooldown_frac:
            eta = 1.0
        else:
            eta = max(0.0, (1 - progress) / cfg.cooldown_frac)
        for g in optimizer.param_groups:
            g["lr"] = g["initial_lr"] * eta

        idx = torch.randint(0, n - cfg.max_len - 1, (cfg.batch_size,), device=device)
        offsets = idx[:, None] + torch.arange(cfg.max_len + 1, device=device)[None, :]
        flat = train_bytes[offsets].long()
        x = flat[:, :-1]
        y = flat[:, 1:]

        optimizer.zero_grad(set_to_none=True)
        if use_amp:
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                logits, _ = model.forward_logits(x)
                loss = F.cross_entropy(logits.reshape(-1, 256), y.reshape(-1))
        else:
            logits, _ = model.forward_logits(x)
            loss = F.cross_entropy(logits.reshape(-1, 256), y.reshape(-1))
        loss.backward()
        optimizer.step()

        if cfg.log_every and (step % cfg.log_every == 0):
            print(
                f"[nystrom_krr] phase1 step {step:5d}  "
                f"loss {loss.item():.4f}  elapsed {elapsed:.1f}s  lr_eta {eta:.3f}",
                flush=True,
            )
        step += 1
    print(f"[nystrom_krr] phase1 done: {step} steps in {time.monotonic()-t0:.1f}s", flush=True)
    return model


# ---------------------------------------------------------------------------
# Phase 2: collect features, run hand-rolled Nystrom KRR.
# ---------------------------------------------------------------------------

@torch.no_grad()
def _build_features_and_targets(
    encoder: Encoder,
    train_bytes: Tensor,
    n_samples: int,
    seq_len: int,
    batch_size: int,
    device: torch.device,
) -> tuple[Tensor, Tensor]:
    """Sample n_samples windows of length (seq_len+1). For each window, encode
    the first seq_len bytes; return the last-token residual (L2-normalized) as
    e in R^(N, d) and the (seq_len)-th byte as the target y in [0, 256).
    Returns (E, Y_onehot) on `device`, fp32."""
    n = train_bytes.numel()
    assert n > seq_len + 1
    encoder.eval()
    d = encoder.model_dim

    E = torch.empty(n_samples, d, dtype=torch.float32, device=device)
    Y_idx = torch.empty(n_samples, dtype=torch.long, device=device)

    t0 = time.monotonic()
    use_amp = device.type == "cuda"
    cursor = 0
    while cursor < n_samples:
        bs = min(batch_size, n_samples - cursor)
        idx = torch.randint(0, n - seq_len - 1, (bs,), device=device)
        offsets = idx[:, None] + torch.arange(seq_len + 1, device=device)[None, :]
        flat = train_bytes[offsets].long()
        x = flat[:, :-1]                                 # (bs, seq_len)
        y = flat[:, -1]                                  # (bs,)
        if use_amp:
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                e_BTd, _ = encoder.encode(x)
        else:
            e_BTd, _ = encoder.encode(x)
        e_last = e_BTd[:, -1, :].float()                 # (bs, d)
        # L2-normalize.
        e_last = e_last / (e_last.norm(dim=-1, keepdim=True) + 1e-8)
        E[cursor:cursor + bs] = e_last
        Y_idx[cursor:cursor + bs] = y
        cursor += bs

    Y = F.one_hot(Y_idx, num_classes=256).float()
    print(f"[nystrom_krr] features: N={n_samples} d={d} in {time.monotonic()-t0:.1f}s",
          flush=True)
    return E, Y


# ---------------------------------------------------------------------------
# CharModel wrapper: KV-cached encoder + Nystrom KRR readout.
# ---------------------------------------------------------------------------

class NystromKRRCharModel(CharModel):
    def __init__(
        self,
        encoder: Encoder,
        Z: Tensor,
        alpha: Tensor,
        device: torch.device | None = None,
    ):
        self.model = encoder
        self.device = device or next(encoder.parameters()).device
        self.model.eval()
        self.Z = Z          # (M, d)
        self.alpha = alpha  # (M, C=256)
        self._kv: list[tuple[Tensor, Tensor]] | None = None
        self._next_logits: Tensor | None = None
        self._pos: int = 0

    @torch.no_grad()
    def _logits_from_residual(self, e: Tensor) -> Tensor:
        """e: (d,) -> logits: (256,). Cosine kernel via L2-normalization."""
        e = e.float()
        e = e / (e.norm() + 1e-8)
        K_q = e.unsqueeze(0) @ self.Z.t().float()        # (1, M)
        logits = K_q @ self.alpha                        # (1, 256)
        return logits[0]

    @torch.no_grad()
    def reset(self) -> None:
        self._kv = None
        self._pos = 0
        # Seed with a single zero byte (same stream-start sentinel as the
        # reference modded-nanogpt wrapper).
        x = torch.zeros(1, 1, dtype=torch.long, device=self.device)
        e_BTd, self._kv = self.model.encode(x, None, offset=self._pos)
        self._next_logits = self._logits_from_residual(e_BTd[0, -1])
        self._pos = 1

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
        if self._kv is None:
            raise RuntimeError("observe() called before reset()")
        for byte in char.encode("utf-8"):
            self._maybe_trim_cache()
            x = torch.tensor([[byte]], dtype=torch.long, device=self.device)
            e_BTd, self._kv = self.model.encode(x, self._kv, offset=self._pos)
            self._next_logits = self._logits_from_residual(e_BTd[0, -1])
            self._pos += 1

    def _maybe_trim_cache(self) -> None:
        if self._kv is None:
            return
        cur = self._kv[0][0].shape[2]
        if cur < self.model.max_len:
            return
        keep = self.model.max_len - 1
        self._kv = [(k[:, :, -keep:], v[:, :, -keep:]) for k, v in self._kv]


# ---------------------------------------------------------------------------
# Entry point — `submit.py` looks for this signature.
# ---------------------------------------------------------------------------

def train(train_text: str, valid_text: str | None = None) -> CharModel:
    seed_env = os.environ.get("SEED")
    if seed_env:
        seed = int(seed_env)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        print(f"[nystrom_krr] SEED={seed}", flush=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Hold the full corpus on GPU as uint8; cast windows to long at sample time.
    raw = train_text.encode("utf-8")
    train_bytes = torch.frombuffer(bytearray(raw), dtype=torch.uint8).to(device)

    cfg = TrainConfig()

    # ---- Phase 1 ---------------------------------------------------------
    t_phase1_start = time.monotonic()
    encoder = _train_phase1(train_bytes, cfg, device)
    phase1_wall = time.monotonic() - t_phase1_start
    print(f"[nystrom_krr] PHASE1 WALL: {phase1_wall:.1f}s", flush=True)

    # ---- Phase 2: collect features ---------------------------------------
    t_phase2_start = time.monotonic()
    N = int(os.environ.get("NYSTROM_N", "100000"))
    M = int(os.environ.get("NYSTROM_M", "1024"))
    penalty = float(os.environ.get("NYSTROM_PENALTY", "1e-3"))
    feat_batch = int(os.environ.get("NYSTROM_FEAT_BS", "256"))

    E, Y = _build_features_and_targets(
        encoder, train_bytes,
        n_samples=N, seq_len=cfg.max_len,
        batch_size=feat_batch, device=device,
    )
    print(f"[nystrom_krr] E: {tuple(E.shape)}  Y: {tuple(Y.shape)}  "
          f"E mean-norm={(E.norm(dim=-1).mean().item()):.4f}", flush=True)

    # ---- Phase 2: solve --------------------------------------------------
    t_solve_start = time.monotonic()
    Z, alpha = nystrom_krr_fit(E, Y, M=M, penalty=penalty)
    # Free large feature matrices before exiting (eval will need GPU memory).
    del E, Y
    torch.cuda.empty_cache() if device.type == "cuda" else None
    print(f"[nystrom_krr] solve: Z={tuple(Z.shape)} alpha={tuple(alpha.shape)} "
          f"in {time.monotonic()-t_solve_start:.2f}s", flush=True)
    phase2_wall = time.monotonic() - t_phase2_start
    print(f"[nystrom_krr] PHASE2 WALL: {phase2_wall:.1f}s", flush=True)
    print(f"[nystrom_krr] TOTAL TRAIN WALL: "
          f"{(phase1_wall + phase2_wall):.1f}s "
          f"(phase1={phase1_wall:.1f}s, phase2={phase2_wall:.1f}s)", flush=True)

    # Discard the phase-1 LM head — replaced by the KRR readout. Saves a bit
    # of host memory and makes accidental misuse impossible.
    encoder.proj = None  # type: ignore[assignment]

    return NystromKRRCharModel(encoder, Z, alpha, device=device)
