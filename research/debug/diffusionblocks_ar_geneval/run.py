"""DiffusionBlocks AR — generative eval debug.

NOT a submission. No CharModel, no leaderboard, no wall-clock cap.

Forks submissions/diffusionblocks_ar_v3/submission.py: same 6L/384d
transformer, B=3 blocks, equi-probability partitioning, EDM schedule,
σ-invariant CE head fix.

Replaces streaming next-byte inference with:
  - Full-sequence Euler generation matching paper Figure 3 (right).
  - Karras ρ=7 schedule with N=50 steps from σ_max=80 → σ_min=0.002.
  - Unconditional generation (BOS-only prefix).

Scores generated text with GPT-2 small token-level perplexity, plus
50 WikiText-103 val anchors for reference.

Usage (on a CUDA host with torch + transformers installed):

    python research/debug/diffusionblocks_ar_geneval/run.py \\
        --data-dir /data \\
        --steps 12000 --batch-size 32 --max-len 1024 \\
        --n-samples 50 --seq-len 256 --n-steps 50 \\
        --out research/debug/diffusionblocks_ar_geneval/

Writes:
  - result.json (per spec)
  - loss_curve.csv (step, b0, b1, b2 EMA)
  - samples.txt (5 raw samples)
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import time
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.optim import AdamW


# ===========================================================================
# Model — copy of v3 (CharModel stripped).
# ===========================================================================

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
    def _num_c(q): return (((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5])
    def _den_d(q): return ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1.0)
    if p < p_low:
        q = math.sqrt(-2.0 * math.log(p))
        return _num_c(q) / _den_d(q)
    if p <= p_high:
        q = p - 0.5
        r = q * q
        num = (((((a[0]*r+a[1])*r+a[2])*r+a[3])*r+a[4])*r+a[5]) * q
        den = (((((b[0]*r+b[1])*r+b[2])*r+b[3])*r+b[4])*r+1.0)
        return num / den
    q = math.sqrt(-2.0 * math.log(1.0 - p))
    return -_num_c(q) / _den_d(q)


def get_block_sigmas(B, sigma_min=0.002, sigma_max=80.0, P_mean=-1.2, P_std=1.2):
    cdf_min = _norm_cdf((math.log(sigma_min) - P_mean) / P_std)
    cdf_max = _norm_cdf((math.log(sigma_max) - P_mean) / P_std)
    return [math.exp(P_mean + P_std * _norm_ppf(cdf_min + (cdf_max - cdf_min) * (b / B)))
            for b in range(B + 1)]


def sample_sigma_in_range(sigma_lo, sigma_hi, P_mean=-1.2, P_std=1.2):
    cdf_lo = _norm_cdf((math.log(sigma_lo) - P_mean) / P_std)
    cdf_hi = _norm_cdf((math.log(sigma_hi) - P_mean) / P_std)
    u = random.uniform(cdf_lo, cdf_hi)
    return math.exp(P_mean + P_std * _norm_ppf(u))


class CondEmbed(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.mlp = nn.Sequential(nn.Linear(dim, dim), nn.SiLU(), nn.Linear(dim, dim))

    def forward(self, c_noise):
        if c_noise.dim() == 0:
            c_noise = c_noise.unsqueeze(0)
        half = self.dim // 2
        freqs = torch.exp(-math.log(10000.0) * torch.arange(half, device=c_noise.device, dtype=torch.float32) / max(1, half - 1))
        args = c_noise.float()[:, None] * freqs[None, :]
        emb = torch.cat([args.sin(), args.cos()], dim=-1)
        if emb.size(-1) < self.dim:
            emb = F.pad(emb, (0, self.dim - emb.size(-1)))
        return self.mlp(emb)


class AdaRMSNorm(nn.Module):
    def __init__(self, dim, cond_dim):
        super().__init__()
        self.dim = dim
        self.cond_proj = nn.Linear(cond_dim, 2 * dim)
        nn.init.zeros_(self.cond_proj.weight)
        nn.init.zeros_(self.cond_proj.bias)

    def forward(self, x, cond_emb):
        gb = self.cond_proj(cond_emb.to(x.dtype))
        gamma, beta = gb.chunk(2, dim=-1)
        if x.dim() == 3:
            gamma = gamma.unsqueeze(1)
            beta = beta.unsqueeze(1)
        return F.rms_norm(x, (x.size(-1),)) * (1 + gamma) + beta


class Rotary(nn.Module):
    def __init__(self, dim):
        super().__init__()
        freq = (1 / 1024) ** torch.linspace(0, 1, steps=dim // 4, dtype=torch.float32)
        self.register_buffer("angular_freq", torch.cat([freq, freq.new_zeros(dim // 4)]))

    def forward(self, x_BTHD, offset=0):
        T = x_BTHD.size(1)
        pos = torch.arange(T, dtype=torch.float32, device=x_BTHD.device) + offset
        theta = torch.outer(pos, self.angular_freq)[None, :, None, :]
        cos, sin = theta.cos(), theta.sin()
        x1, x2 = x_BTHD.to(dtype=torch.float32).chunk(2, dim=-1)
        y1 = x1 * cos + x2 * sin
        y2 = x1 * (-sin) + x2 * cos
        return torch.cat((y1, y2), 3).type_as(x_BTHD)


def make_joint_causal_mask(T_x: int, T_z: int, device) -> Tensor:
    """Boolean attention mask for joint sequence [x_emb || z], shape [N, N].

    Logical positions: x at [0..T_x-1], z at [0..T_z-1]. Attention allowed iff:
      - x query (row < T_x) attends to x keys (col < T_x) with col <= row
        (standard causal LM over the clean prefix).
      - z query at logical t = row - T_x attends to x keys at logical s <= t
        AND z keys at logical s <= t (≤ — z attends to itself).
      - x never attends to z (preserves AR factorization).
    True = attend, False = masked out (PyTorch sdpa convention).
    """
    N = T_x + T_z
    idx = torch.arange(N, device=device)
    is_x = idx < T_x
    logical = torch.where(is_x, idx, idx - T_x)  # [N]
    r_log = logical[:, None]
    c_log = logical[None, :]
    is_x_r = is_x[:, None]
    is_x_c = is_x[None, :]
    # row=x: only allowed col=x AND c_log <= r_log
    # row=z: any col with c_log <= r_log (covers x_{<=t} and z_{<=t})
    mask = ((is_x_r & is_x_c & (c_log <= r_log))
            | ((~is_x_r) & (c_log <= r_log)))
    return mask  # [N, N] bool


class SelfLayer(nn.Module):
    """Joint-sequence causal self-attention layer.

    Operates on the concatenated sequence [x_emb || z]: a single QKV projection
    over all positions, with the custom joint-causal mask built by
    `make_joint_causal_mask`. RoPE positions are *logical* (x at 0..T_x-1, z at
    0..T_z-1) so that z at logical pos t and x at logical pos t share the same
    rotary encoding — semantically they sit at the same sequence position.
    """
    def __init__(self, dim, head_dim, cond_dim):
        super().__init__()
        self.num_heads = dim // head_dim
        self.head_dim = head_dim
        self.qkv = nn.Linear(dim, 3 * dim)
        self.proj = nn.Linear(dim, dim)
        self.mlp_fc = nn.Linear(dim, 4 * dim)
        self.mlp_proj = nn.Linear(4 * dim, dim)
        self.norm1 = AdaRMSNorm(dim, cond_dim)
        self.norm2 = AdaRMSNorm(dim, cond_dim)
        self.rotary = Rotary(head_dim)

    def forward(self, seq, T_x, cond_emb, attn_mask):
        B, N, _ = seq.shape
        T_z = N - T_x
        s_in = self.norm1(seq, cond_emb)
        qkv = self.qkv(s_in).view(B, N, 3, self.num_heads, self.head_dim)
        q = qkv[:, :, 0]
        k = qkv[:, :, 1]
        v = qkv[:, :, 2]
        q = F.rms_norm(q, (q.size(-1),))
        k = F.rms_norm(k, (k.size(-1),))
        q_x, q_z = q[:, :T_x], q[:, T_x:]
        k_x, k_z = k[:, :T_x], k[:, T_x:]
        q_x = self.rotary(q_x, offset=0)
        q_z = self.rotary(q_z, offset=0)
        k_x = self.rotary(k_x, offset=0)
        k_z = self.rotary(k_z, offset=0)
        q = torch.cat([q_x, q_z], dim=1).transpose(1, 2).contiguous()
        k = torch.cat([k_x, k_z], dim=1).transpose(1, 2).contiguous()
        v = v.transpose(1, 2).contiguous()
        attn = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, scale=0.12)
        attn = attn.transpose(1, 2).contiguous().view(B, N, -1)
        seq = seq + self.proj(attn)
        h = self.mlp_fc(self.norm2(seq, cond_emb))
        h = h.relu().square()
        seq = seq + self.mlp_proj(h)
        return seq


class DBlock(nn.Module):
    def __init__(self, dim, head_dim, n_layers, cond_dim):
        super().__init__()
        self.layers = nn.ModuleList([SelfLayer(dim, head_dim, cond_dim) for _ in range(n_layers)])
        self.norm_out = AdaRMSNorm(dim, cond_dim)

    def forward(self, x_emb, z, cond_emb):
        B, T_x, _ = x_emb.shape
        T_z = z.shape[1]
        seq = torch.cat([x_emb, z], dim=1)
        mask = make_joint_causal_mask(T_x, T_z, seq.device)
        for layer in self.layers:
            seq = layer(seq, T_x, cond_emb, mask)
        seq = self.norm_out(seq, cond_emb)
        return seq[:, T_x:, :]


class DBlocksAR(nn.Module):
    def __init__(self, vocab_size=256, num_layers=6, model_dim=384, head_dim=64,
                 num_blocks=3, cond_dim=128, max_len=1024):
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
        self.E_out = nn.Parameter(torch.empty(vocab_size, model_dim))
        nn.init.normal_(self.E_out, std=1.0 / model_dim**0.5)
        self.logit_scale = nn.Parameter(torch.tensor(float(model_dim) ** 0.5))
        self.cond_embed = CondEmbed(cond_dim)
        self.blocks = nn.ModuleList([
            DBlock(model_dim, head_dim, self.layers_per_block, cond_dim)
            for _ in range(num_blocks)
        ])
        boundaries = get_block_sigmas(num_blocks)
        self.register_buffer("block_sigmas", torch.tensor(boundaries, dtype=torch.float32))
        self.register_buffer("sigma_data", torch.tensor(1.0 / model_dim**0.5, dtype=torch.float32))

    def normalize_eout(self):
        return F.normalize(self.E_out, dim=-1)

    def block_range(self, b, gamma=0.10):
        s_lo = float(self.block_sigmas[b])
        s_hi = float(self.block_sigmas[b + 1])
        if gamma > 0.0:
            log_lo, log_hi = math.log(s_lo), math.log(s_hi)
            rng = log_hi - log_lo
            s_lo = max(math.exp(log_lo - gamma * rng), float(self.block_sigmas[0]))
            s_hi = min(math.exp(log_hi + gamma * rng), float(self.block_sigmas[-1]))
        return s_lo, s_hi


def _init_model(model):
    for name, p in model.named_parameters():
        if not name.endswith("weight"):
            continue
        if "proj" in name or "mlp_proj" in name:
            nn.init.zeros_(p)


# ===========================================================================
# Training
# ===========================================================================

@contextmanager
def _maybe_autocast(device):
    if device.type == "cuda":
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            yield
    else:
        yield


def calibrate_sigma_data(model):
    with torch.no_grad():
        return float(model.normalize_eout().std().item())


def train_model(text, *, n_steps, batch_size, max_len, num_layers, num_blocks,
                model_dim, head_dim, cond_dim, gamma, lambda_ce, lr, weight_decay,
                cooldown_frac, log_every, device, loss_csv_path):
    raw = text.encode("utf-8")
    train_bytes = torch.frombuffer(bytearray(raw), dtype=torch.uint8).to(device)
    n = train_bytes.numel()
    if n < max_len + 1:
        raise ValueError(f"need ≥ {max_len+1} bytes; got {n}")

    model = DBlocksAR(vocab_size=256, num_layers=num_layers, model_dim=model_dim,
                      head_dim=head_dim, num_blocks=num_blocks, cond_dim=cond_dim,
                      max_len=max_len).to(device)
    _init_model(model)
    sd = calibrate_sigma_data(model)
    model.sigma_data.fill_(sd)
    print(f"[geneval] σ_data = {sd:.4f}", flush=True)
    print(f"[geneval] block boundaries: {model.block_sigmas.tolist()}", flush=True)

    fused = (device.type == "cuda")
    block_opts = [AdamW(blk.parameters(), lr=lr, weight_decay=weight_decay,
                        betas=(0.9, 0.95), eps=1e-10, fused=fused)
                  for blk in model.blocks]
    shared_params = (list(model.embed.parameters())
                     + list(model.cond_embed.parameters())
                     + [model.E_out, model.logit_scale])
    shared_opt = AdamW(shared_params, lr=lr, weight_decay=weight_decay,
                       betas=(0.9, 0.95), eps=1e-10, fused=fused)
    all_opts = block_opts + [shared_opt]
    for opt in all_opts:
        for g in opt.param_groups:
            g["initial_lr"] = g["lr"]

    n_params = sum(p.numel() for p in model.parameters())
    print(f"[geneval] {n_params/1e6:.2f}M params  steps={n_steps}  bs={batch_size}  T={max_len}", flush=True)

    def set_lr(step):
        progress = step / max(1, n_steps)
        eta = 1.0 if progress < 1 - cooldown_frac else max(0.0, (1 - progress) / cooldown_frac)
        for opt in all_opts:
            for g in opt.param_groups:
                g["lr"] = g["initial_lr"] * eta

    model.train()
    t0 = time.monotonic()
    block_loss_ema = [None] * num_blocks
    block_count = [0] * num_blocks
    loss_rows = []

    for step in range(n_steps):
        set_lr(step)
        idx = torch.randint(0, n - max_len - 1, (batch_size,), device=device)
        offsets = idx[:, None] + torch.arange(max_len + 1, device=device)[None, :]
        flat = train_bytes[offsets].long()
        x = flat[:, :-1]
        y = flat[:, 1:]

        b = random.randrange(num_blocks)
        s_lo, s_hi = model.block_range(b, gamma=gamma)
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
            x_emb = model.embed(x)
            E = model.normalize_eout()
            y_emb = E[y]
            eps = torch.randn_like(y_emb)
            z = y_emb + sigma_t * eps
            cond = model.cond_embed(cond_in).expand(batch_size, -1)
            z_in = z * c_in
            out = model.blocks[b](x_emb, z_in, cond)
            pred_y = out * c_out + z * c_skip
            loss_l2 = weight * (pred_y - y_emb).pow(2).mean()
            pred_y_n = F.normalize(pred_y.float(), dim=-1)
            logits = model.logit_scale * (pred_y_n @ E.float().t())
            loss_ce = F.cross_entropy(logits.reshape(-1, model.vocab_size), y.reshape(-1))
            loss = loss_l2 + lambda_ce * loss_ce

        loss.backward()
        block_opts[b].step()
        shared_opt.step()

        block_count[b] += 1
        l = float(loss.item())
        prev = block_loss_ema[b]
        block_loss_ema[b] = l if prev is None else 0.95 * prev + 0.05 * l

        if log_every and (step % log_every == 0 or step == n_steps - 1):
            elapsed = time.monotonic() - t0
            per_block = "  ".join(
                f"b{i}={(block_loss_ema[i] if block_loss_ema[i] is not None else float('nan')):.3f}"
                f"({block_count[i]})" for i in range(num_blocks)
            )
            print(f"[geneval] step {step:5d}/{n_steps}  b={b} σ={sigma_t:.3f}  "
                  f"loss={l:.4f} (l2={float(loss_l2):.3f} ce={float(loss_ce):.3f})  "
                  f"{per_block}  elapsed={elapsed:.0f}s", flush=True)
            loss_rows.append((step, *(block_loss_ema[i] if block_loss_ema[i] is not None else float("nan")
                                       for i in range(num_blocks))))

    with open(loss_csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["step"] + [f"b{i}" for i in range(num_blocks)])
        for row in loss_rows:
            w.writerow(row)

    final_loss = {f"b{i}": (block_loss_ema[i] if block_loss_ema[i] is not None else float("nan"))
                  for i in range(num_blocks)}
    return model, final_loss


# ===========================================================================
# Full-sequence Euler generation (paper Fig. 3 right)
# ===========================================================================

def karras_sigma_schedule(sigma_min, sigma_max, N, rho=7.0):
    ramp = torch.linspace(0, 1, N)
    min_inv = sigma_min ** (1 / rho)
    max_inv = sigma_max ** (1 / rho)
    sigmas = (max_inv + ramp * (min_inv - max_inv)) ** rho
    sigmas = torch.cat([sigmas, torch.zeros(1)])  # ends at 0
    return sigmas.tolist()


def _route_block(sigma, boundaries):
    """Block whose [σ_b, σ_{b+1}] contains sigma; clamp to ends."""
    B = len(boundaries) - 1
    if sigma >= boundaries[-1]:
        return B - 1
    if sigma <= boundaries[0]:
        return 0
    for b in range(B):
        if boundaries[b] <= sigma <= boundaries[b + 1]:
            return b
    return B - 1


@torch.no_grad()
def generate(model, n_samples=50, seq_len=256, device="cuda", n_steps=50, rho=7.0,
             sigma_min=0.002, sigma_max=80.0):
    """Full-sequence Euler unconditional generation.

    1. z_0 ~ N(0, σ_max² I), shape [n_samples, seq_len, dim].
    2. Walk Karras ρ=7 schedule, N steps. At each step, route by σ to the
       block whose range contains σ_i, run a single Euler step.
    3. At σ_min, argmax z against L2-normalised E_out → token ids → bytes.

    Conditioning prefix x: a single BOS byte (0) per sample. The K/V cache
    seen by every Q position at every step is just that one BOS position —
    unconditional generation matches paper's LM1B/OWT setup.
    """
    model.eval()
    dim = model.model_dim
    sd = float(model.sigma_data.item())
    boundaries = model.block_sigmas.tolist()
    sigmas = karras_sigma_schedule(sigma_min, sigma_max, n_steps, rho=rho)

    # BOS prefix: single zero byte → embedding of shape [n, 1, dim].
    bos = torch.zeros(n_samples, 1, dtype=torch.long, device=device)
    x_emb = model.embed(bos)

    # z_0 over the *target* positions of length seq_len.
    z = torch.randn(n_samples, seq_len, dim, device=device) * sigma_max

    # Joint-sequence forward: each step concatenates [BOS_emb || z_current]
    # along the sequence dim, runs the routed block with the proper joint
    # causal mask, takes the z slice, and applies the EDM Euler update.
    for i in range(n_steps):
        sigma = sigmas[i]
        sigma_next = sigmas[i + 1]
        sigma_t = max(sigma, 1e-8)
        b_idx = _route_block(sigma_t, boundaries)

        c_skip = sd**2 / (sigma_t**2 + sd**2)
        c_out_ = sigma_t * sd / math.sqrt(sigma_t**2 + sd**2)
        c_in = 1.0 / math.sqrt(sigma_t**2 + sd**2)
        c_noise = 0.25 * math.log(sigma_t)
        cond = model.cond_embed(torch.tensor([c_noise], device=device, dtype=torch.float32))
        cond = cond.expand(n_samples, -1)

        z_in = z * c_in
        out = model.blocks[b_idx](x_emb, z_in, cond)
        denoised = out * c_out_ + z * c_skip
        # Euler step on probability-flow ODE: dx/dσ = (x - D(x;σ)) / σ.
        d = (z - denoised) / sigma_t
        z = z + d * (sigma_next - sigma_t)

    # Decode: argmax z @ E_out.T (logit_scale is monotone, irrelevant for argmax).
    E = model.normalize_eout()
    z_n = F.normalize(z.float(), dim=-1)
    logits = z_n @ E.float().t()  # [n, T, V]
    ids = logits.argmax(-1).cpu().numpy().astype(np.uint8)  # [n, T]

    out_strings = []
    for row in ids:
        out_strings.append(bytes(row.tolist()).decode("utf-8", errors="replace"))
    return out_strings


# ===========================================================================
# GPT-2 perplexity scorer
# ===========================================================================

@torch.no_grad()
def gpt2_perplexity(texts, device="cuda", batch_size=4):
    """Token-level perplexity of each text under GPT-2 small.

    Returns {"mean": float, "std": float, "n": int, "per_sample": [...]}.
    """
    from transformers import GPT2LMHeadModel, GPT2TokenizerFast
    tok = GPT2TokenizerFast.from_pretrained("gpt2")
    tok.pad_token = tok.eos_token
    model = GPT2LMHeadModel.from_pretrained("gpt2").to(device).eval()

    per = []
    for t in texts:
        enc = tok(t, return_tensors="pt", truncation=True, max_length=1024).to(device)
        if enc.input_ids.numel() < 2:
            per.append(float("nan"))
            continue
        out = model(**enc, labels=enc.input_ids)
        loss = float(out.loss.item())  # mean NLL per token
        per.append(math.exp(min(loss, 30.0)))  # cap exp to avoid overflow
    arr = np.array(per, dtype=np.float64)
    finite = arr[np.isfinite(arr)]
    return {
        "mean": float(finite.mean()) if finite.size else float("nan"),
        "std": float(finite.std()) if finite.size else float("nan"),
        "n": int(finite.size),
        "per_sample": per,
    }


# ===========================================================================
# Main
# ===========================================================================

def _read_split(data_dir, split):
    p = Path(data_dir) / f"wiki.{split}.raw"
    return p.read_text(encoding="utf-8")


def _sample_val_chunks(val_text, n, seq_len, seed=0):
    """Return n char-windows of length seq_len from val_text (byte-level)."""
    raw = val_text.encode("utf-8")
    rng = random.Random(seed)
    out = []
    for _ in range(n):
        start = rng.randrange(0, max(1, len(raw) - seq_len - 1))
        chunk = raw[start:start + seq_len]
        out.append(chunk.decode("utf-8", errors="replace"))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=os.environ.get("WIKITEXT_DIR", "/data"))
    ap.add_argument("--out", default="research/debug/diffusionblocks_ar_geneval")
    ap.add_argument("--steps", type=int, default=12000)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--max-len", type=int, default=1024)
    ap.add_argument("--num-layers", type=int, default=6)
    ap.add_argument("--num-blocks", type=int, default=3)
    ap.add_argument("--model-dim", type=int, default=384)
    ap.add_argument("--head-dim", type=int, default=64)
    ap.add_argument("--cond-dim", type=int, default=128)
    ap.add_argument("--gamma", type=float, default=0.10)
    ap.add_argument("--lambda-ce", type=float, default=1.0)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--weight-decay", type=float, default=0.0)
    ap.add_argument("--cooldown-frac", type=float, default=0.7)
    ap.add_argument("--log-every", type=int, default=100)
    ap.add_argument("--n-samples", type=int, default=50)
    ap.add_argument("--seq-len", type=int, default=256)
    ap.add_argument("--n-inference-steps", type=int, default=50)
    ap.add_argument("--rho", type=float, default=7.0)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[geneval] loading WikiText-103 from {args.data_dir}", flush=True)
    train_text = _read_split(args.data_dir, "train")
    val_text = _read_split(args.data_dir, "valid")
    print(f"[geneval] train chars={len(train_text):,}  val chars={len(val_text):,}", flush=True)

    loss_csv = out_dir / "loss_curve.csv"
    model, final_loss = train_model(
        train_text, n_steps=args.steps, batch_size=args.batch_size, max_len=args.max_len,
        num_layers=args.num_layers, num_blocks=args.num_blocks, model_dim=args.model_dim,
        head_dim=args.head_dim, cond_dim=args.cond_dim, gamma=args.gamma,
        lambda_ce=args.lambda_ce, lr=args.lr, weight_decay=args.weight_decay,
        cooldown_frac=args.cooldown_frac, log_every=args.log_every, device=device,
        loss_csv_path=loss_csv,
    )

    print(f"[geneval] generating {args.n_samples} samples × {args.seq_len} bytes "
          f"with N={args.n_inference_steps} Karras steps ...", flush=True)
    t_gen0 = time.monotonic()
    samples = generate(model, n_samples=args.n_samples, seq_len=args.seq_len,
                       device=device, n_steps=args.n_inference_steps, rho=args.rho)
    print(f"[geneval] generation took {time.monotonic()-t_gen0:.1f}s", flush=True)

    (out_dir / "samples.txt").write_text("\n\n===\n\n".join(samples))

    val_anchors = _sample_val_chunks(val_text, args.n_samples, args.seq_len, seed=args.seed)

    print("[geneval] scoring with GPT-2 small ...", flush=True)
    ppl_gen = gpt2_perplexity(samples, device=device)
    ppl_val = gpt2_perplexity(val_anchors, device=device)

    result = {
        "model": "diffusionblocks_ar_v3_geneval",
        "train_steps": args.steps,
        "per_block_loss_final": final_loss,
        "gpt2_ppl_generated": {k: ppl_gen[k] for k in ("mean", "std", "n")},
        "gpt2_ppl_wikitext_val": {k: ppl_val[k] for k in ("mean", "std", "n")},
        "samples": samples[:5],
        "config": vars(args),
    }
    (out_dir / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    print(f"[geneval] wrote {out_dir/'result.json'}", flush=True)
    print(f"[geneval] PPL gen = {ppl_gen['mean']:.1f} ± {ppl_gen['std']:.1f}  "
          f"PPL val = {ppl_val['mean']:.1f} ± {ppl_val['std']:.1f}", flush=True)


if __name__ == "__main__":
    main()
