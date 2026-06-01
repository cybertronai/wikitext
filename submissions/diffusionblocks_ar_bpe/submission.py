"""DiffusionBlocks AR for WikiText-103 — BPE-token variant.

Adapts the best-performing byte-level diffusion submission
(`diffusionblocks_ar_v4`, total_J ≈ 62,430, val acc ≈ 0.235, DQ) to the
paper's original token-level recipe (Shing, Koyama, Akiba — ICLR 2026,
arxiv:2506.14202, §5.4): a 6-layer Llama-2-style transformer with token-
level BPE, B=3 blocks. Per spec_17 the paper used Llama-2 BPE; we use
GPT-2 BPE via tiktoken — both are deterministic merge tables over byte
streams (rules §"Internal representations") and GPT-2 is what the
in-image registry already ships.

Why the change. The byte-level v1..v4 runs all plateaued at val char-acc
≈ 0.22–0.24 with per-block CE landing at b0≈0.25, b1≈0.6, b2≈1.6 —
the high-σ block stayed at ~ln(256)·0.3, i.e. it never moved past
"slightly better than uniform over 256 bytes" even after the v3 σ-invariant
CE-head fix and the v4 σ_max=5 schedule shrink. That suggests the EDM
denoising signal is not the bottleneck — the bottleneck is what the
denoiser is being asked to predict. Bytes have ~1 bit of structure per
position (the next byte is mostly determined by the last ~4 bytes for
English text); a continuous-embedding denoiser is mis-matched to that
near-deterministic target. The paper's AR experiment trained on BPE
tokens, where each target carries ~5–6 nats of next-token uncertainty,
and the continuous denoising signal has something to actually reduce.

Internal-BPE → char-acc translation. Per README §"Internal representations":
predict() returns `argmax_c P(next_char | observed_chars)`, computed by
marginalising over BPE tokens consistent with the current byte buffer.
For an active partial buffer `p` and candidate next byte `c`:

  P(c | p) ∝ Σ_t P(t | committed) · 1[bytes(t) starts with p+c]

then argmax over bytes and decode to a UTF-8 char (return "" if the
argmax byte is a non-ASCII continuation, treated as abstain).

What stays identical to v4 (the diffusion machinery itself is unchanged):
  - B=3 blocks of 2 cross-attention layers, AdaLN(c_noise) modulation
  - EDM preconditioning (c_skip, c_out, c_in, c_noise = 0.25·log σ)
  - Equi-probability noise partitioning over LogNormal(P_mean=-1.2, P_std=1.2)
  - σ_max=5 (v4's narrowed schedule — keeps inference Euler inside training
    distribution's support; see v4 module docstring)
  - L2 + λ_ce·CE objective with σ-invariant normalised CE head
  - Per-block AdamW + shared optimizer for embed/E_out/cond_embed
  - B Euler steps at inference, each routed to its block's KV-cache

What changes:
  - Tokenizer: tiktoken "gpt2" (50_257 vocab). Held in the image (submit.py
    pre-installs tiktoken==0.7.0).
  - Training corpus: train_text → token-id stream via encode_ordinary, then
    we sample (max_len+1)-token windows for next-token prediction.
  - Sequence/batch: max_len=512 tokens (≈ 2,000 chars per row; matches
    v4's effective per-row char budget at much lower memory because the
    CE head over 50K classes dominates activation memory). batch_size=32.
  - n_steps stays B·baseline=6450 from v4 (paper Appendix D.1 fair-comparison
    rule). Per-step wall-clock is similar — the active block forward sees
    the same B·T·d activations; the embedding-table and E_out matmuls
    are larger but still fast for V=50_257 on A100.
  - Loss/CE head over 50_257 classes. ln 50_257 ≈ 10.8, so the v4 CE
    ceiling story (saturation at ln 256 = 5.5 over a 256-vocab uniform)
    doesn't constrain us here — the normalised CE head still gives
    σ-invariant logit magnitudes.
  - CharModel: BPE marginalisation in predict(); incremental token commit
    in observe() (mirrors submissions/bpe_internal_nn_v2's contract-correct
    streaming pattern, ported to the new `predict() -> str` API).
"""
from __future__ import annotations

__author__ = "@ab-10"

import concurrent.futures
import math
import os
import random
import time
from contextlib import contextmanager

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.optim import AdamW

from wikitext import CharModel


# ---------------------------------------------------------------------------
# Noise schedule (verbatim from v4)
# ---------------------------------------------------------------------------

def _norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _norm_ppf(p: float) -> float:
    if p <= 0.0:
        return -float("inf")
    if p >= 1.0:
        return float("inf")
    a = (-3.969683028665376e+01, 2.209460984245205e+02,
         -2.759285104469687e+02, 1.383577518672690e+02,
         -3.066479806614716e+01, 2.506628277459239e+00)
    b = (-5.447609879822406e+01, 1.615858368580409e+02,
         -1.556989798598866e+02, 6.680131188771972e+01,
         -1.328068155288572e+01)
    c = (-7.784894002430293e-03, -3.223964580411365e-01,
         -2.400758277161838e+00, -2.549732539343734e+00,
         4.374664141464968e+00, 2.938163982698783e+00)
    d = (7.784695709041462e-03, 3.224671290700398e-01,
         2.445134137142996e+00, 3.754408661907416e+00)
    p_low = 0.02425
    p_high = 1.0 - p_low
    def _num_c(q: float) -> float:
        return (((((c[0]*q + c[1])*q + c[2])*q + c[3])*q + c[4])*q + c[5])
    def _den_d(q: float) -> float:
        return ((((d[0]*q + d[1])*q + d[2])*q + d[3])*q + 1.0)
    if p < p_low:
        q = math.sqrt(-2.0 * math.log(p))
        return _num_c(q) / _den_d(q)
    if p <= p_high:
        q = p - 0.5
        r = q * q
        num = (((((a[0]*r + a[1])*r + a[2])*r + a[3])*r + a[4])*r + a[5]) * q
        den = (((((b[0]*r + b[1])*r + b[2])*r + b[3])*r + b[4])*r + 1.0)
        return num / den
    q = math.sqrt(-2.0 * math.log(1.0 - p))
    return -_num_c(q) / _den_d(q)


def get_block_sigmas(B: int, sigma_min: float = 0.002, sigma_max: float = 80.0,
                     P_mean: float = -1.2, P_std: float = 1.2) -> list[float]:
    cdf_min = _norm_cdf((math.log(sigma_min) - P_mean) / P_std)
    cdf_max = _norm_cdf((math.log(sigma_max) - P_mean) / P_std)
    out = []
    for b in range(B + 1):
        q = cdf_min + (cdf_max - cdf_min) * (b / B)
        out.append(math.exp(P_mean + P_std * _norm_ppf(q)))
    return out


def sample_sigma_in_range(sigma_lo: float, sigma_hi: float,
                          P_mean: float = -1.2, P_std: float = 1.2) -> float:
    cdf_lo = _norm_cdf((math.log(sigma_lo) - P_mean) / P_std)
    cdf_hi = _norm_cdf((math.log(sigma_hi) - P_mean) / P_std)
    u = random.uniform(cdf_lo, cdf_hi)
    return math.exp(P_mean + P_std * _norm_ppf(u))


# ---------------------------------------------------------------------------
# c_noise embedder, AdaLN, RoPE, CrossLayer, DBlock — verbatim from v4
# ---------------------------------------------------------------------------

class CondEmbed(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim), nn.SiLU(), nn.Linear(dim, dim),
        )

    def forward(self, c_noise: Tensor) -> Tensor:
        if c_noise.dim() == 0:
            c_noise = c_noise.unsqueeze(0)
        half = self.dim // 2
        freqs = torch.exp(
            -math.log(10000.0) * torch.arange(half, device=c_noise.device, dtype=torch.float32) / max(1, half - 1)
        )
        args = c_noise.float()[:, None] * freqs[None, :]
        emb = torch.cat([args.sin(), args.cos()], dim=-1)
        if emb.size(-1) < self.dim:
            emb = F.pad(emb, (0, self.dim - emb.size(-1)))
        return self.mlp(emb)


class AdaRMSNorm(nn.Module):
    def __init__(self, dim: int, cond_dim: int):
        super().__init__()
        self.dim = dim
        self.cond_proj = nn.Linear(cond_dim, 2 * dim)
        nn.init.zeros_(self.cond_proj.weight)
        nn.init.zeros_(self.cond_proj.bias)

    def forward(self, x: Tensor, cond_emb: Tensor) -> Tensor:
        gb = self.cond_proj(cond_emb.to(x.dtype))
        gamma, beta = gb.chunk(2, dim=-1)
        if x.dim() == 3:
            gamma = gamma.unsqueeze(1)
            beta = beta.unsqueeze(1)
        x_normed = F.rms_norm(x, (x.size(-1),))
        return x_normed * (1 + gamma) + beta


class Rotary(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        freq = (1 / 1024) ** torch.linspace(0, 1, steps=dim // 4, dtype=torch.float32)
        self.register_buffer(
            "angular_freq",
            torch.cat([freq, freq.new_zeros(dim // 4)]),
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


class CrossLayer(nn.Module):
    def __init__(self, dim: int, head_dim: int, cond_dim: int):
        super().__init__()
        self.num_heads = dim // head_dim
        self.head_dim = head_dim
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.proj = nn.Linear(dim, dim)
        self.mlp_fc = nn.Linear(dim, 4 * dim)
        self.mlp_proj = nn.Linear(4 * dim, dim)
        self.norm1 = AdaRMSNorm(dim, cond_dim)
        self.norm2 = AdaRMSNorm(dim, cond_dim)
        self.rotary = Rotary(head_dim)

    def project_kv(self, x_emb: Tensor, offset: int) -> tuple[Tensor, Tensor]:
        B, T = x_emb.shape[:2]
        k = self.k(x_emb).view(B, T, self.num_heads, self.head_dim)
        v = self.v(x_emb).view(B, T, self.num_heads, self.head_dim)
        k = F.rms_norm(k, (k.size(-1),))
        k = self.rotary(k, offset=offset)
        return k.transpose(1, 2).contiguous(), v.transpose(1, 2).contiguous()

    def extend_kv(self, x_emb: Tensor,
                  kv_cache: tuple[Tensor, Tensor] | None,
                  offset: int) -> tuple[Tensor, Tensor]:
        k_new, v_new = self.project_kv(x_emb, offset=offset)
        if kv_cache is None:
            return (k_new, v_new)
        k_old, v_old = kv_cache
        return (torch.cat([k_old, k_new], dim=2),
                torch.cat([v_old, v_new], dim=2))

    def _qproj(self, z: Tensor, offset: int) -> Tensor:
        B, T = z.shape[:2]
        q = self.q(z).view(B, T, self.num_heads, self.head_dim)
        q = F.rms_norm(q, (q.size(-1),))
        q = self.rotary(q, offset=offset)
        return q.transpose(1, 2).contiguous()

    def forward_train(self, z: Tensor, k: Tensor, v: Tensor,
                      cond_emb: Tensor) -> Tensor:
        z_in = self.norm1(z, cond_emb)
        q = self._qproj(z_in, offset=0)
        attn = F.scaled_dot_product_attention(q, k, v, scale=0.12, is_causal=True)
        B, T = z.shape[:2]
        attn = attn.transpose(1, 2).contiguous().view(B, T, -1)
        z = z + self.proj(attn)
        h = self.mlp_fc(self.norm2(z, cond_emb))
        h = h.relu().square()
        z = z + self.mlp_proj(h)
        return z

    def forward_infer(self, z: Tensor, kv_cache: tuple[Tensor, Tensor],
                      q_offset: int, cond_emb: Tensor) -> Tensor:
        z_in = self.norm1(z, cond_emb)
        q = self._qproj(z_in, offset=q_offset)
        k, v = kv_cache
        attn = F.scaled_dot_product_attention(q, k, v, scale=0.12)
        B = z.size(0)
        attn = attn.transpose(1, 2).contiguous().view(B, 1, -1)
        z = z + self.proj(attn)
        h = self.mlp_fc(self.norm2(z, cond_emb))
        h = h.relu().square()
        z = z + self.mlp_proj(h)
        return z


class DBlock(nn.Module):
    def __init__(self, dim: int, head_dim: int, n_layers: int, cond_dim: int):
        super().__init__()
        self.layers = nn.ModuleList([
            CrossLayer(dim, head_dim, cond_dim) for _ in range(n_layers)
        ])
        self.norm_out = AdaRMSNorm(dim, cond_dim)

    def forward_train(self, z: Tensor, x_emb: Tensor, cond_emb: Tensor) -> Tensor:
        for layer in self.layers:
            k, v = layer.project_kv(x_emb, offset=0)
            z = layer.forward_train(z, k, v, cond_emb)
        return self.norm_out(z, cond_emb)

    def extend_kv(self, x_emb: Tensor,
                  kv_caches: list[tuple[Tensor, Tensor]] | None,
                  offset: int) -> list[tuple[Tensor, Tensor]]:
        if kv_caches is None:
            kv_caches = [None] * len(self.layers)
        return [layer.extend_kv(x_emb, cache, offset)
                for layer, cache in zip(self.layers, kv_caches)]

    def forward_infer(self, z: Tensor,
                      kv_caches: list[tuple[Tensor, Tensor]],
                      q_offset: int, cond_emb: Tensor) -> Tensor:
        for layer, cache in zip(self.layers, kv_caches):
            z = layer.forward_infer(z, cache, q_offset, cond_emb)
        return self.norm_out(z, cond_emb)


# ---------------------------------------------------------------------------
# Top-level model — same structure as v4, only vocab_size differs
# ---------------------------------------------------------------------------

GPT2_VOCAB_SIZE = 50_257
GPT2_BOS_ID = 50_256   # <|endoftext|> serves as BOS/PAD in GPT-2 BPE
MAX_TOKEN_BYTES = 64


class DBlocksBPEAR(nn.Module):
    def __init__(self, vocab_size: int = GPT2_VOCAB_SIZE, num_layers: int = 6,
                 model_dim: int = 384, head_dim: int = 64,
                 num_blocks: int = 3, cond_dim: int = 128,
                 max_len: int = 512):
        super().__init__()
        assert num_layers % num_blocks == 0
        self.vocab_size = vocab_size
        self.num_layers = num_layers
        self.model_dim = model_dim
        self.head_dim = head_dim
        self.num_blocks = num_blocks
        self.cond_dim = cond_dim
        self.max_len = max_len
        self.layers_per_block = num_layers // num_blocks
        self.embed = nn.Embedding(vocab_size, model_dim)
        # E_out: 50K × 384 ≈ 19M params. Dominates parameter count vs v4's
        # 256 × 384 ≈ 100K. Weight-tied: used both as the noising target
        # lookup (L2-normalised rows) and the CE-head projection.
        self.E_out = nn.Parameter(torch.empty(vocab_size, model_dim))
        nn.init.normal_(self.E_out, std=1.0 / model_dim**0.5)
        self.logit_scale = nn.Parameter(torch.tensor(float(model_dim) ** 0.5))
        self.cond_embed = CondEmbed(cond_dim)
        self.blocks = nn.ModuleList([
            DBlock(model_dim, head_dim, self.layers_per_block, cond_dim)
            for _ in range(num_blocks)
        ])
        # v4's σ_max=5 — kept verbatim. The bound is set by the LogNormal
        # noise distribution's training support, not by the vocab size, so
        # the rationale is identical at the token level.
        boundaries = get_block_sigmas(num_blocks, sigma_max=5.0)
        self.register_buffer("block_sigmas", torch.tensor(boundaries, dtype=torch.float32))
        self.register_buffer("sigma_data", torch.tensor(1.0 / model_dim**0.5, dtype=torch.float32))

    def normalize_eout(self) -> Tensor:
        return F.normalize(self.E_out, dim=-1)

    def block_range(self, b: int, gamma: float = 0.10) -> tuple[float, float]:
        s_lo = float(self.block_sigmas[b])
        s_hi = float(self.block_sigmas[b + 1])
        if gamma > 0.0:
            log_lo, log_hi = math.log(s_lo), math.log(s_hi)
            rng = log_hi - log_lo
            s_lo = max(math.exp(log_lo - gamma * rng), float(self.block_sigmas[0]))
            s_hi = min(math.exp(log_hi + gamma * rng), float(self.block_sigmas[-1]))
        return s_lo, s_hi


def _init_model(model: DBlocksBPEAR) -> None:
    for name, p in model.named_parameters():
        if not name.endswith("weight"):
            continue
        if "proj" in name or "mlp_proj" in name:
            nn.init.zeros_(p)


# ---------------------------------------------------------------------------
# Tokeniser + token-bytes table
# ---------------------------------------------------------------------------

def _split_at_safe_boundaries(s: str, n_chunks: int) -> list[str]:
    if n_chunks <= 1 or len(s) < 1024 * n_chunks:
        return [s]
    target = len(s) // n_chunks
    chunks: list[str] = []
    start = 0
    for _ in range(n_chunks - 1):
        cut = start + target
        while cut < len(s) and not s[cut].isspace():
            cut += 1
        if cut >= len(s):
            break
        chunks.append(s[start:cut])
        start = cut
    chunks.append(s[start:])
    return [c for c in chunks if c]


def _parallel_encode(text: str, encoding, n_threads: int) -> list[int]:
    chunks = _split_at_safe_boundaries(text, n_threads)
    if len(chunks) == 1:
        return encoding.encode_ordinary(chunks[0])
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(chunks)) as ex:
        results = list(ex.map(encoding.encode_ordinary, chunks))
    out: list[int] = []
    for r in results:
        out.extend(r)
    return out


def _build_token_bytes_table(encoding) -> tuple[np.ndarray, np.ndarray]:
    V = encoding.n_vocab
    arr = np.zeros((V, MAX_TOKEN_BYTES), dtype=np.uint8)
    lens = np.zeros(V, dtype=np.int32)
    for tid in range(V):
        try:
            b = encoding.decode_single_token_bytes(tid)
        except Exception:
            continue
        L = min(len(b), MAX_TOKEN_BYTES)
        lens[tid] = L
        arr[tid, :L] = np.frombuffer(b[:L], dtype=np.uint8)
    return arr, lens


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

class TrainConfig:
    def __init__(self,
                 model_dim: int = 384,
                 num_layers: int = 6,
                 head_dim: int = 64,
                 num_blocks: int = 3,
                 cond_dim: int = 128,
                 max_len: int = 512,
                 batch_size: int = 32,
                 baseline_steps: int = 1100,
                 n_steps: int | None = None,
                 gamma: float = 0.10,
                 lambda_ce: float = 1.0,
                 lr: float = 5e-4,
                 weight_decay: float = 0.0,
                 cooldown_frac: float = 0.7,
                 log_every: int = 200,
                 vocab_size: int = GPT2_VOCAB_SIZE):
        self.model_dim = model_dim
        self.num_layers = num_layers
        self.head_dim = head_dim
        self.num_blocks = num_blocks
        self.cond_dim = cond_dim
        self.max_len = max_len
        self.batch_size = batch_size
        self.baseline_steps = baseline_steps
        self.n_steps = n_steps if n_steps is not None else num_blocks * baseline_steps
        self.gamma = gamma
        self.lambda_ce = lambda_ce
        self.lr = lr
        self.weight_decay = weight_decay
        self.cooldown_frac = cooldown_frac
        self.log_every = log_every
        self.vocab_size = vocab_size

    def __repr__(self):
        return (f"TrainConfig(d={self.model_dim} L={self.num_layers} "
                f"B={self.num_blocks} V={self.vocab_size} bs={self.batch_size} "
                f"T={self.max_len} steps={self.n_steps} γ={self.gamma} "
                f"λ_ce={self.lambda_ce})")


def calibrate_sigma_data(model: DBlocksBPEAR) -> float:
    with torch.no_grad():
        E = model.normalize_eout()
        return float(E.std().item())


@contextmanager
def _maybe_autocast(device: torch.device):
    if device.type == "cuda":
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            yield
    else:
        yield


def _train(text: str, cfg: TrainConfig, device: torch.device,
           encoding=None) -> tuple[DBlocksBPEAR, object, np.ndarray, np.ndarray]:
    import tiktoken
    if encoding is None:
        encoding = tiktoken.get_encoding("gpt2")
    # Tokenise: ~4 chars/token for English text, so a ~500MB WikiText
    # train stream → ~125M tokens. Parallel encode keeps tokenisation
    # off the training-time budget — submit.py wraps train() with the
    # wall-clock cap, so seconds spent in encode_ordinary count.
    t_tok = time.monotonic()
    n_threads = max(1, (os.cpu_count() or 1))
    token_ids = _parallel_encode(text, encoding, n_threads)
    token_arr = np.asarray(token_ids, dtype=np.int32)
    print(f"[dblocks-bpe] tokenised {len(text):,} chars → "
          f"{token_arr.size:,} tokens in {time.monotonic()-t_tok:.1f}s "
          f"({len(text)/max(1,token_arr.size):.2f} chars/token)", flush=True)

    token_bytes_arr, token_lens = _build_token_bytes_table(encoding)

    n = token_arr.size
    if n < cfg.max_len + 1:
        raise ValueError(f"need at least {cfg.max_len+1} tokens; got {n}")
    train_tokens = torch.from_numpy(token_arr).to(device=device, dtype=torch.long)

    model = DBlocksBPEAR(
        vocab_size=cfg.vocab_size, num_layers=cfg.num_layers,
        model_dim=cfg.model_dim, head_dim=cfg.head_dim,
        num_blocks=cfg.num_blocks, cond_dim=cfg.cond_dim, max_len=cfg.max_len,
    ).to(device)
    _init_model(model)

    sigma_data = calibrate_sigma_data(model)
    model.sigma_data.fill_(sigma_data)
    print(f"[dblocks-bpe] σ_data calibrated to {sigma_data:.4f}", flush=True)

    boundaries = model.block_sigmas.tolist()
    print(f"[dblocks-bpe] block boundaries: "
          f"[{', '.join(f'{s:.4f}' for s in boundaries)}]", flush=True)

    fused = (device.type == "cuda")
    block_opts = [
        AdamW(blk.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay,
              betas=(0.9, 0.95), eps=1e-10, fused=fused)
        for blk in model.blocks
    ]
    shared_params = (
        list(model.embed.parameters())
        + list(model.cond_embed.parameters())
        + [model.E_out, model.logit_scale]
    )
    shared_opt = AdamW(shared_params, lr=cfg.lr, weight_decay=cfg.weight_decay,
                       betas=(0.9, 0.95), eps=1e-10, fused=fused)
    all_opts = block_opts + [shared_opt]
    for opt in all_opts:
        for g in opt.param_groups:
            g["initial_lr"] = g["lr"]

    n_params = sum(p.numel() for p in model.parameters())
    print(f"[dblocks-bpe] {n_params/1e6:.2f}M params  cfg={cfg}", flush=True)

    def set_lr(step: int) -> None:
        progress = step / max(1, cfg.n_steps)
        if progress < 1 - cfg.cooldown_frac:
            eta = 1.0
        else:
            eta = max(0.0, (1 - progress) / cfg.cooldown_frac)
        for opt in all_opts:
            for g in opt.param_groups:
                g["lr"] = g["initial_lr"] * eta

    model.train()
    t0 = time.monotonic()
    block_loss_ema: list[float | None] = [None] * cfg.num_blocks
    block_count = [0] * cfg.num_blocks
    sd = float(sigma_data)

    for step in range(cfg.n_steps):
        set_lr(step)
        idx = torch.randint(0, n - cfg.max_len - 1, (cfg.batch_size,), device=device)
        offsets = idx[:, None] + torch.arange(cfg.max_len + 1, device=device)[None, :]
        flat = train_tokens[offsets]
        x = flat[:, :-1]   # [B, T]   prefix tokens
        y = flat[:, 1:]    # [B, T]   next tokens (denoising targets)

        b = random.randrange(cfg.num_blocks)
        s_lo, s_hi = model.block_range(b, gamma=cfg.gamma)
        sigma = sample_sigma_in_range(s_lo, s_hi)
        sigma_t = max(sigma, 1e-8)
        c_skip = sd**2 / (sigma_t**2 + sd**2)
        c_out = sigma_t * sd / math.sqrt(sigma_t**2 + sd**2)
        c_in = 1.0 / math.sqrt(sigma_t**2 + sd**2)
        c_noise = 0.25 * math.log(sigma_t)
        weight = (sigma_t**2 + sd**2) / (sigma_t * sd) ** 2

        block_opts[b].zero_grad(set_to_none=True)
        shared_opt.zero_grad(set_to_none=True)

        cond_in = torch.tensor([c_noise], device=device, dtype=torch.float32)

        with _maybe_autocast(device):
            x_emb = model.embed(x)             # [B, T, d]
            E = model.normalize_eout()         # [V, d]
            y_emb = E[y]                       # [B, T, d]
            eps = torch.randn_like(y_emb)
            z = y_emb + sigma_t * eps
            cond = model.cond_embed(cond_in)
            cond = cond.expand(cfg.batch_size, -1)

            z_in = z * c_in
            out = model.blocks[b].forward_train(z_in, x_emb, cond)
            pred_y = out * c_out + z * c_skip  # [B, T, d]

            loss_l2 = weight * (pred_y - y_emb).pow(2).mean()
            pred_y_n = F.normalize(pred_y.float(), dim=-1)
            logits = model.logit_scale * (pred_y_n @ E.float().t())
            loss_ce = F.cross_entropy(
                logits.reshape(-1, model.vocab_size), y.reshape(-1),
            )
            loss = loss_l2 + cfg.lambda_ce * loss_ce

        loss.backward()
        block_opts[b].step()
        shared_opt.step()

        block_count[b] += 1
        l = float(loss.item())
        prev = block_loss_ema[b]
        block_loss_ema[b] = l if prev is None else 0.95 * prev + 0.05 * l

        if cfg.log_every and (step % cfg.log_every == 0 or step == cfg.n_steps - 1):
            elapsed = time.monotonic() - t0
            per_block = "  ".join(
                f"b{i}={(block_loss_ema[i] if block_loss_ema[i] is not None else float('nan')):.3f}"
                f"({block_count[i]})"
                for i in range(cfg.num_blocks)
            )
            print(
                f"[dblocks-bpe] step {step:5d}/{cfg.n_steps}  "
                f"b={b} σ={sigma_t:.3f}  "
                f"loss={l:.4f} (l2={float(loss_l2):.3f} ce={float(loss_ce):.3f})  "
                f"{per_block}  elapsed={elapsed:.0f}s",
                flush=True,
            )

    return model, encoding, token_bytes_arr, token_lens


# ---------------------------------------------------------------------------
# Streaming inference (CharModel) — BPE marginalisation at the boundary
# ---------------------------------------------------------------------------

class DBlocksBPECharModel(CharModel):
    """B-step Euler inference → token logits → per-byte marginalisation.

    Streaming protocol:
      - observe(c) appends c's UTF-8 bytes to `_history`. Whenever the
        un-committed tail (uncommitted bytes) tokenises to ≥ 2 tokens, all
        but the last are committed: their KV is extended into every
        block's cache and we drop their bytes from the pending buffer.
        The last token stays in the buffer because adding more chars can
        still change how its bytes merge.
      - predict() runs B Euler steps from N(0, σ_max²·I), denoised by each
        block in turn from high σ → low σ. The final z is projected onto
        normalised E_out rows; softmax → P(next_token | committed_tokens).
        Then for each token t with bytes starting with the pending buffer
        `p`, P(t) contributes to byte mass at index `bytes(t)[|p|]`. The
        argmax byte across [0, 256) is decoded as a UTF-8 char; if it's
        a non-ASCII continuation byte we return "" (abstain).
    """

    def __init__(self, model: DBlocksBPEAR, encoding,
                 token_bytes_arr: np.ndarray, token_lens: np.ndarray,
                 device: torch.device | None = None):
        self.model = model
        self.encoding = encoding
        self.token_bytes_arr = token_bytes_arr
        self.token_lens = token_lens
        self.device = device or next(model.parameters()).device
        self.model.eval()
        self.B = model.num_blocks
        self.dim = model.model_dim
        self.max_len = model.max_len
        self.sigma_data = float(model.sigma_data.item())
        boundaries = model.block_sigmas.tolist()
        # Same routing as v4: step i goes σ_schedule[i] → σ_schedule[i+1]
        # and is dispatched to block (B-1-i), the block whose [σ_b, σ_{b+1}]
        # range contains the high end of the step.
        self.sigma_schedule = list(reversed(boundaries))
        self._kv: list | None = None
        self._pos: int = 0
        self._history: bytearray = bytearray()
        self._committed_byte_count: int = 0
        self._bos_id: int = GPT2_BOS_ID

    @torch.no_grad()
    def reset(self) -> None:
        self._kv = [None] * self.B
        self._pos = 0
        self._history = bytearray()
        self._committed_byte_count = 0
        # Seed every block's KV cache with a single BOS token (GPT-2's
        # <|endoftext|>, id 50256). Mirrors v4's zero-byte seed plus the
        # bpe_internal_nn_v2 pattern.
        x = torch.tensor([[self._bos_id]], dtype=torch.long, device=self.device)
        x_emb = self.model.embed(x)
        for b in range(self.B):
            self._kv[b] = self.model.blocks[b].extend_kv(
                x_emb, self._kv[b], offset=self._pos,
            )
        self._pos = 1

    def _pending_buffer(self) -> bytes:
        if self._committed_byte_count >= len(self._history):
            return b""
        return bytes(self._history[self._committed_byte_count:])

    @torch.no_grad()
    def _next_token_probs(self) -> np.ndarray:
        """Run B Euler steps over the per-block KV caches and return a
        normalised P(next_token) numpy array (length vocab_size)."""
        assert self._kv is not None
        E = self.model.normalize_eout()
        q_offset = self._pos - 1
        sigma_hi = self.sigma_schedule[0]
        z = torch.randn(1, 1, self.dim, device=self.device) * sigma_hi
        sd = self.sigma_data
        for i in range(self.B):
            sigma = self.sigma_schedule[i]
            sigma_next = self.sigma_schedule[i + 1]
            block_idx = self.B - 1 - i
            sigma_t = max(sigma, 1e-8)
            c_skip = sd**2 / (sigma_t**2 + sd**2)
            c_out = sigma_t * sd / math.sqrt(sigma_t**2 + sd**2)
            c_in = 1.0 / math.sqrt(sigma_t**2 + sd**2)
            c_noise = 0.25 * math.log(sigma_t)
            cond = self.model.cond_embed(
                torch.tensor([c_noise], device=self.device, dtype=torch.float32),
            )
            z_in = z * c_in
            out = self.model.blocks[block_idx].forward_infer(
                z_in, self._kv[block_idx], q_offset=q_offset, cond_emb=cond,
            )
            denoised = out * c_out + z * c_skip
            d = (z - denoised) / sigma_t
            z = z + d * (sigma_next - sigma)
        final = F.normalize(z.squeeze(0).squeeze(0).float(), dim=-1)
        logits = self.model.logit_scale * (final @ E.float().t())  # [V]
        probs = F.softmax(logits, dim=-1).cpu().numpy()
        return probs

    @torch.no_grad()
    def predict(self) -> str:
        if self._kv is None:
            raise RuntimeError("predict() called before reset()")
        p_token = self._next_token_probs()
        pending = self._pending_buffer()
        plen = len(pending)
        if plen == 0:
            active_mask = self.token_lens >= 1
        else:
            pending_arr = np.frombuffer(pending, dtype=np.uint8)
            cmp = self.token_bytes_arr[:, :plen] == pending_arr[None, :]
            prefix_match = cmp.all(axis=1)
            # Token must extend the pending buffer by ≥ 1 byte so it
            # contributes mass to *the next* byte slot.
            active_mask = prefix_match & (self.token_lens > plen)
        active_ids = np.flatnonzero(active_mask)
        if active_ids.size == 0:
            return ""
        active_next_bytes = self.token_bytes_arr[active_ids, plen]
        active_probs = p_token[active_ids]
        mass = np.bincount(
            active_next_bytes.astype(np.int64),
            weights=active_probs.astype(np.float64),
            minlength=256,
        )
        if mass.sum() <= 0.0:
            return ""
        byte_id = int(mass.argmax())
        # Same UTF-8 decode discipline as v4: non-ASCII bytes (≥128) are
        # multi-byte continuations on their own and won't form a valid
        # char; we treat that as abstain.
        try:
            return bytes([byte_id]).decode("utf-8")
        except UnicodeDecodeError:
            return ""

    @torch.no_grad()
    def observe(self, char: str) -> None:
        if self._kv is None:
            raise RuntimeError("observe() called before reset()")
        for byte in char.encode("utf-8"):
            self._history.append(byte)
        self._maybe_commit_tokens()

    def _maybe_commit_tokens(self) -> None:
        if self._committed_byte_count >= len(self._history):
            return
        tail_bytes = bytes(self._history[self._committed_byte_count:])
        try:
            tail_str = tail_bytes.decode("utf-8")
        except UnicodeDecodeError:
            tail_str = tail_bytes.decode("utf-8", errors="replace")
        token_ids = self.encoding.encode_ordinary(tail_str)
        if len(token_ids) <= 1:
            return
        # Commit all but the last — the last may still merge with future
        # chars and so must stay buffered. Same conservatism as
        # bpe_internal_nn_v2.
        new_tokens = token_ids[:-1]
        consumed = sum(
            len(self.encoding.decode_single_token_bytes(t))
            for t in new_tokens
        )
        x = torch.tensor([new_tokens], dtype=torch.long, device=self.device)
        x_emb = self.model.embed(x)
        for b in range(self.B):
            self._kv[b] = self.model.blocks[b].extend_kv(
                x_emb, self._kv[b], offset=self._pos,
            )
        self._pos += len(new_tokens)
        self._committed_byte_count += consumed
        self._maybe_trim_cache()

    def _maybe_trim_cache(self) -> None:
        if self._kv is None:
            return
        cur = self._kv[0][0][0].shape[2]
        if cur < self.max_len:
            return
        keep = self.max_len - 1
        for b in range(self.B):
            self._kv[b] = [(k[:, :, -keep:], v[:, :, -keep:])
                           for (k, v) in self._kv[b]]


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def train(train_text: str, valid_text: str | None = None) -> CharModel:
    seed_env = os.environ.get("SEED")
    if seed_env:
        seed = int(seed_env)
        torch.manual_seed(seed)
        random.seed(seed)
        np.random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        print(f"[dblocks-bpe] SEED={seed}", flush=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = TrainConfig()
    model, encoding, token_bytes_arr, token_lens = _train(train_text, cfg, device)
    return DBlocksBPECharModel(model, encoding, token_bytes_arr, token_lens, device=device)
