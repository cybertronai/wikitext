"""DiffusionBlocks AR — BPE tokenization + MAUVE eval debug.

NOT a submission. No CharModel, no leaderboard, no wall-clock cap.

Fork of research/debug/diffusionblocks_ar_selfattn/run.py. Changes:
  - Tokenizer: GPT-2 BPE (vocab_size=50,257) replacing raw bytes (256).
  - Architecture: d=384, L=6, B=3 — same ratios as selfattn run.
    Parameter count will be higher than the byte-level run due to the
    embedding table (2 × 50,257 × 384 ≈ 38.6M for embed+E_out alone);
    actual count is logged and recorded in result.json.
  - Eval: MAUVE score (via the `mauve-text` package) replacing GPT-2
    perplexity. 250 prompted continuations from WikiText-103 val.
    n=250 is below the 1K recommended for stable MAUVE estimates —
    treat results as directional only (high-variance warning in result.json).
  - Generation: conditional on a real val prompt (first 32 tokens),
    generates a 50-token continuation via the Euler chain.
    Matches the paper's §5.4 eval setup (MAUVE, prompted, 50 tokens).

Architecture is otherwise identical to selfattn:
  - Joint-sequence causal self-attention over [x_emb || z], length 2T.
  - EDM noise schedule, equi-probability block partitioning.
  - σ-invariant CE head (L2-normalize pred_y before E_out projection).
  - L2 + CE hybrid loss.
  - σ_data calibrated empirically at startup.

Usage:

    pip install mauve-text transformers torch
    python research/debug/diffusionblocks_ar_bpe_mauve/run.py \\
        --data-dir /data \\
        --steps 12000 --batch-size 32 --max-len 256 \\
        --n-prompts 250 --prompt-len 32 --continuation-len 50 \\
        --out research/debug/diffusionblocks_ar_bpe_mauve/

Writes:
  - result.json
  - loss_curve.csv (step, b0, b1, b2 EMA)
  - samples.txt   (5 prompted continuations)
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
from transformers import GPT2TokenizerFast


# ===========================================================================
# Noise schedule helpers (identical to selfattn)
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
    p_low, p_high = 0.02425, 1.0 - 0.02425
    def _nc(q): return (((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5])
    def _nd(q): return ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1.0)
    if p < p_low:
        q = math.sqrt(-2.0 * math.log(p)); return _nc(q) / _nd(q)
    if p <= p_high:
        q = p - 0.5; r = q*q
        return (((((a[0]*r+a[1])*r+a[2])*r+a[3])*r+a[4])*r+a[5])*q / (((((b[0]*r+b[1])*r+b[2])*r+b[3])*r+b[4])*r+1.0)
    q = math.sqrt(-2.0 * math.log(1.0 - p)); return -_nc(q) / _nd(q)


def get_block_sigmas(B, sigma_min=0.002, sigma_max=80.0, P_mean=-1.2, P_std=1.2):
    cdf_min = _norm_cdf((math.log(sigma_min) - P_mean) / P_std)
    cdf_max = _norm_cdf((math.log(sigma_max) - P_mean) / P_std)
    return [math.exp(P_mean + P_std * _norm_ppf(cdf_min + (cdf_max - cdf_min) * b / B))
            for b in range(B + 1)]


def sample_sigma_in_range(sigma_lo, sigma_hi, P_mean=-1.2, P_std=1.2):
    cdf_lo = _norm_cdf((math.log(sigma_lo) - P_mean) / P_std)
    cdf_hi = _norm_cdf((math.log(sigma_hi) - P_mean) / P_std)
    return math.exp(P_mean + P_std * _norm_ppf(random.uniform(cdf_lo, cdf_hi)))


# ===========================================================================
# Model (identical to selfattn except vocab_size is now 50,257)
# ===========================================================================

class CondEmbed(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.mlp = nn.Sequential(nn.Linear(dim, dim), nn.SiLU(), nn.Linear(dim, dim))

    def forward(self, c_noise):
        if c_noise.dim() == 0:
            c_noise = c_noise.unsqueeze(0)
        half = self.dim // 2
        freqs = torch.exp(-math.log(10000.0) * torch.arange(
            half, device=c_noise.device, dtype=torch.float32) / max(1, half - 1))
        args = c_noise.float()[:, None] * freqs[None, :]
        emb = torch.cat([args.sin(), args.cos()], dim=-1)
        if emb.size(-1) < self.dim:
            emb = F.pad(emb, (0, self.dim - emb.size(-1)))
        return self.mlp(emb)


class AdaRMSNorm(nn.Module):
    def __init__(self, dim, cond_dim):
        super().__init__()
        self.cond_proj = nn.Linear(cond_dim, 2 * dim)
        nn.init.zeros_(self.cond_proj.weight)
        nn.init.zeros_(self.cond_proj.bias)

    def forward(self, x, cond_emb):
        gb = self.cond_proj(cond_emb.to(x.dtype))
        gamma, beta = gb.chunk(2, dim=-1)
        if x.dim() == 3:
            gamma, beta = gamma.unsqueeze(1), beta.unsqueeze(1)
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
        x1, x2 = x_BTHD.to(torch.float32).chunk(2, dim=-1)
        return torch.cat((x1*cos + x2*sin, x1*(-sin) + x2*cos), 3).type_as(x_BTHD)


class SelfLayer(nn.Module):
    """Causal self-attention over joint sequence [x_emb || z], length 2T."""
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

    def forward(self, seq, cond_emb):
        B, N, _ = seq.shape
        s_in = self.norm1(seq, cond_emb)
        qkv = self.qkv(s_in).view(B, N, 3, self.num_heads, self.head_dim)
        q, k, v = qkv[:,:,0], qkv[:,:,1], qkv[:,:,2]
        q, k = F.rms_norm(q, (q.size(-1),)), F.rms_norm(k, (k.size(-1),))
        q, k = self.rotary(q), self.rotary(k)
        q = q.transpose(1,2).contiguous()
        k = k.transpose(1,2).contiguous()
        v = v.transpose(1,2).contiguous()
        attn = F.scaled_dot_product_attention(q, k, v, is_causal=True, scale=0.12)
        attn = attn.transpose(1,2).contiguous().view(B, N, -1)
        seq = seq + self.proj(attn)
        h = self.mlp_fc(self.norm2(seq, cond_emb))
        seq = seq + self.mlp_proj(h.relu().square())
        return seq


class DBlock(nn.Module):
    def __init__(self, dim, head_dim, n_layers, cond_dim):
        super().__init__()
        self.layers = nn.ModuleList([SelfLayer(dim, head_dim, cond_dim) for _ in range(n_layers)])
        self.norm_out = AdaRMSNorm(dim, cond_dim)

    def forward(self, x_emb, z, cond_emb):
        T_x = x_emb.shape[1]
        seq = torch.cat([x_emb, z], dim=1)
        for layer in self.layers:
            seq = layer(seq, cond_emb)
        seq = self.norm_out(seq, cond_emb)
        return seq[:, T_x:, :]


class DBlocksAR(nn.Module):
    def __init__(self, vocab_size=50257, num_layers=6, model_dim=384, head_dim=64,
                 num_blocks=3, cond_dim=128, max_len=256):
        super().__init__()
        assert num_layers % num_blocks == 0
        self.vocab_size = vocab_size
        self.num_layers = num_layers
        self.model_dim = model_dim
        self.num_blocks = num_blocks
        self.max_len = max_len
        self.layers_per_block = num_layers // num_blocks
        self.embed = nn.Embedding(vocab_size, model_dim)
        self.E_out = nn.Parameter(torch.empty(vocab_size, model_dim))
        nn.init.normal_(self.E_out, std=1.0 / model_dim**0.5)
        self.logit_scale = nn.Parameter(torch.tensor(float(model_dim)**0.5))
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


def train_model(token_ids, *, n_steps, batch_size, max_len, num_layers, num_blocks,
                model_dim, head_dim, cond_dim, gamma, lambda_ce, lr, weight_decay,
                cooldown_frac, log_every, device, loss_csv_path, vocab_size):
    train_ids = torch.tensor(token_ids, dtype=torch.long, device=device)
    n = train_ids.numel()
    if n < max_len + 1:
        raise ValueError(f"need ≥ {max_len+1} tokens; got {n}")

    model = DBlocksAR(vocab_size=vocab_size, num_layers=num_layers, model_dim=model_dim,
                      head_dim=head_dim, num_blocks=num_blocks, cond_dim=cond_dim,
                      max_len=max_len).to(device)

    sd = calibrate_sigma_data(model)
    model.sigma_data.fill_(sd)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[bpe_mauve] σ_data={sd:.4f}  params={n_params/1e6:.2f}M  "
          f"vocab={vocab_size}  d={model_dim}  L={num_layers}  B={num_blocks}", flush=True)
    print(f"[bpe_mauve] block boundaries: {model.block_sigmas.tolist()}", flush=True)

    fused = device.type == "cuda"
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
        flat = train_ids[offsets]
        x = flat[:, :-1]   # [B, T] context tokens
        y = flat[:, 1:]    # [B, T] target tokens

        b = random.randrange(num_blocks)
        s_lo, s_hi = model.block_range(b, gamma=gamma)
        sigma = sample_sigma_in_range(s_lo, s_hi)
        sigma_t = max(sigma, 1e-8)
        c_skip = sd**2 / (sigma_t**2 + sd**2)
        c_out = sigma_t * sd / math.sqrt(sigma_t**2 + sd**2)
        c_in = 1.0 / math.sqrt(sigma_t**2 + sd**2)
        c_noise = 0.25 * math.log(sigma_t)
        weight = (sigma_t**2 + sd**2) / (sigma_t * sd)**2

        block_opts[b].zero_grad(set_to_none=True)
        shared_opt.zero_grad(set_to_none=True)
        cond_in = torch.tensor([c_noise], device=device, dtype=torch.float32)

        with _maybe_autocast(device):
            x_emb = model.embed(x)
            E = model.normalize_eout()
            y_emb = E[y]
            z = y_emb + sigma_t * torch.randn_like(y_emb)
            cond = model.cond_embed(cond_in).expand(batch_size, -1)
            out = model.blocks[b](x_emb, z * c_in, cond)
            pred_y = out * c_out + z * c_skip
            loss_l2 = weight * (pred_y - y_emb).pow(2).mean()
            pred_y_n = F.normalize(pred_y.float(), dim=-1)
            logits = model.logit_scale * (pred_y_n @ E.float().t())
            loss_ce = F.cross_entropy(logits.reshape(-1, vocab_size), y.reshape(-1))
            loss = loss_l2 + lambda_ce * loss_ce

        loss.backward()
        block_opts[b].step()
        shared_opt.step()

        block_count[b] += 1
        lv = float(loss.item())
        prev = block_loss_ema[b]
        block_loss_ema[b] = lv if prev is None else 0.95 * prev + 0.05 * lv

        if log_every and (step % log_every == 0 or step == n_steps - 1):
            elapsed = time.monotonic() - t0
            per_block = "  ".join(
                f"b{i}={block_loss_ema[i]:.3f}({block_count[i]})"
                if block_loss_ema[i] is not None
                else f"b{i}=n/a({block_count[i]})"
                for i in range(num_blocks)
            )
            print(f"[bpe_mauve] step {step:5d}/{n_steps}  b={b} σ={sigma_t:.3f}  "
                  f"loss={lv:.4f} (l2={float(loss_l2):.3f} ce={float(loss_ce):.3f})  "
                  f"{per_block}  elapsed={elapsed:.0f}s", flush=True)
            loss_rows.append((step, *(block_loss_ema[i] for i in range(num_blocks))))

    with open(loss_csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["step"] + [f"b{i}" for i in range(num_blocks)])
        for row in loss_rows:
            w.writerow(row)

    final_loss = {f"b{i}": block_loss_ema[i] for i in range(num_blocks)}
    return model, final_loss, n_params


# ===========================================================================
# Conditional Euler generation (prompted, 50-token continuation)
# ===========================================================================

def karras_sigma_schedule(sigma_min, sigma_max, N, rho=7.0):
    ramp = torch.linspace(0, 1, N)
    min_inv, max_inv = sigma_min**(1/rho), sigma_max**(1/rho)
    sigmas = (max_inv + ramp * (min_inv - max_inv))**rho
    return torch.cat([sigmas, torch.zeros(1)]).tolist()


def _route_block(sigma, boundaries):
    B = len(boundaries) - 1
    if sigma >= boundaries[-1]: return B - 1
    if sigma <= boundaries[0]:  return 0
    for b in range(B):
        if boundaries[b] <= sigma <= boundaries[b + 1]:
            return b
    return B - 1


@torch.no_grad()
def generate_conditional(model, prompt_ids, continuation_len=50,
                          n_steps=50, rho=7.0, device="cuda",
                          sigma_min=0.002, sigma_max=80.0):
    """Generate a continuation_len-token continuation conditioned on prompt_ids.

    prompt_ids: [1, T_prompt] long tensor — the observed prefix.
    Returns a list of token-id lists, one per sample in the batch.

    The prompt is embedded as x_emb and held fixed across all Euler steps.
    z starts as pure Gaussian noise of shape [1, continuation_len, dim]
    and is denoised toward the target distribution conditioned on x_emb.
    """
    model.eval()
    dim = model.model_dim
    sd = float(model.sigma_data.item())
    boundaries = model.block_sigmas.tolist()
    sigmas = karras_sigma_schedule(sigma_min, sigma_max, n_steps, rho=rho)

    x_emb = model.embed(prompt_ids.to(device))  # [1, T_prompt, d]
    z = torch.randn(1, continuation_len, dim, device=device) * sigma_max

    for i in range(n_steps):
        sigma = sigmas[i]
        sigma_next = sigmas[i + 1]
        sigma_t = max(sigma, 1e-8)
        b_idx = _route_block(sigma_t, boundaries)

        c_skip = sd**2 / (sigma_t**2 + sd**2)
        c_out_ = sigma_t * sd / math.sqrt(sigma_t**2 + sd**2)
        c_in = 1.0 / math.sqrt(sigma_t**2 + sd**2)
        c_noise = 0.25 * math.log(sigma_t)
        cond = model.cond_embed(
            torch.tensor([c_noise], device=device, dtype=torch.float32)
        )  # [1, cond_dim]

        out = model.blocks[b_idx](x_emb, z * c_in, cond)
        denoised = out * c_out_ + z * c_skip
        d = (z - denoised) / sigma_t
        z = z + d * (sigma_next - sigma_t)

    E = model.normalize_eout()
    z_n = F.normalize(z.float(), dim=-1)
    logits = z_n @ E.float().t()          # [1, continuation_len, vocab]
    ids = logits.argmax(-1).squeeze(0)    # [continuation_len]
    return ids.cpu().tolist()


# ===========================================================================
# MAUVE scoring
# ===========================================================================

def compute_mauve(generated_texts, reference_texts, device="cuda", verbose=False):
    """MAUVE score between generated and reference text lists.

    Uses GPT-2 features (default in the mauve-text package).
    Returns the MAUVE scalar (higher = better, max 1.0).
    """
    import mauve
    out = mauve.compute_mauve(
        p_text=generated_texts,
        q_text=reference_texts,
        device_id=0 if device == "cuda" or (hasattr(device, "type") and device.type == "cuda") else -1,
        max_text_length=512,
        verbose=verbose,
        featurize_model_name="gpt2",
    )
    return float(out.mauve)


# ===========================================================================
# Data helpers
# ===========================================================================

def _read_split(data_dir, split):
    p = Path(data_dir) / f"wiki.{split}.raw"
    return p.read_text(encoding="utf-8")


def _tokenize(text, tokenizer):
    return tokenizer.encode(text)


def _sample_prompt_chunks(val_token_ids, n, prompt_len, continuation_len, seed=0):
    """Sample n (prompt, reference_continuation) pairs from val token ids."""
    chunk = prompt_len + continuation_len
    rng = random.Random(seed)
    prompts, refs = [], []
    for _ in range(n):
        start = rng.randrange(0, max(1, len(val_token_ids) - chunk - 1))
        p = val_token_ids[start: start + prompt_len]
        r = val_token_ids[start + prompt_len: start + chunk]
        prompts.append(p)
        refs.append(r)
    return prompts, refs


# ===========================================================================
# Main
# ===========================================================================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=os.environ.get("WIKITEXT_DIR", "/data"))
    ap.add_argument("--out", default="research/debug/diffusionblocks_ar_bpe_mauve")
    ap.add_argument("--steps", type=int, default=12000)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--max-len", type=int, default=256)
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
    ap.add_argument("--n-prompts", type=int, default=250)
    ap.add_argument("--prompt-len", type=int, default=32)
    ap.add_argument("--continuation-len", type=int, default=50)
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

    print(f"[bpe_mauve] loading GPT-2 tokenizer", flush=True)
    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
    vocab_size = tokenizer.vocab_size  # 50257

    print(f"[bpe_mauve] loading WikiText-103 from {args.data_dir}", flush=True)
    train_text = _read_split(args.data_dir, "train")
    val_text = _read_split(args.data_dir, "valid")
    print(f"[bpe_mauve] tokenizing ...", flush=True)
    train_ids = _tokenize(train_text, tokenizer)
    val_ids = _tokenize(val_text, tokenizer)
    print(f"[bpe_mauve] train tokens={len(train_ids):,}  val tokens={len(val_ids):,}", flush=True)

    loss_csv = out_dir / "loss_curve.csv"
    model, final_loss, n_params = train_model(
        train_ids, n_steps=args.steps, batch_size=args.batch_size,
        max_len=args.max_len, num_layers=args.num_layers, num_blocks=args.num_blocks,
        model_dim=args.model_dim, head_dim=args.head_dim, cond_dim=args.cond_dim,
        gamma=args.gamma, lambda_ce=args.lambda_ce, lr=args.lr,
        weight_decay=args.weight_decay, cooldown_frac=args.cooldown_frac,
        log_every=args.log_every, device=device, loss_csv_path=loss_csv,
        vocab_size=vocab_size,
    )

    # --- generation ---
    print(f"[bpe_mauve] sampling {args.n_prompts} prompt/reference pairs from val ...", flush=True)
    prompts, ref_continuations = _sample_prompt_chunks(
        val_ids, args.n_prompts, args.prompt_len, args.continuation_len, seed=args.seed
    )

    print(f"[bpe_mauve] generating {args.n_prompts} continuations "
          f"({args.continuation_len} tokens each, {args.n_inference_steps} Euler steps) ...", flush=True)
    t_gen0 = time.monotonic()
    generated_ids = []
    for i, prompt in enumerate(prompts):
        prompt_tensor = torch.tensor([prompt], dtype=torch.long, device=device)
        cont = generate_conditional(
            model, prompt_tensor, continuation_len=args.continuation_len,
            n_steps=args.n_inference_steps, rho=args.rho, device=device,
        )
        generated_ids.append(cont)
        if (i + 1) % 50 == 0:
            print(f"[bpe_mauve]   generated {i+1}/{args.n_prompts}", flush=True)
    print(f"[bpe_mauve] generation took {time.monotonic()-t_gen0:.1f}s", flush=True)

    # decode to strings for MAUVE
    generated_texts = [tokenizer.decode(ids, skip_special_tokens=True) for ids in generated_ids]
    reference_texts = [tokenizer.decode(ids, skip_special_tokens=True) for ids in ref_continuations]

    # write samples
    sample_lines = []
    for i in range(min(5, len(generated_texts))):
        sample_lines.append(f"=== Sample {i} ===")
        sample_lines.append(f"PROMPT:   {tokenizer.decode(prompts[i])}")
        sample_lines.append(f"GENERATED:{generated_texts[i]}")
        sample_lines.append(f"REFERENCE:{reference_texts[i]}")
    (out_dir / "samples.txt").write_text("\n".join(sample_lines))

    # --- MAUVE ---
    print(f"[bpe_mauve] computing MAUVE (n={args.n_prompts}, "
          f"high-variance warning: n < 1000) ...", flush=True)
    mauve_score = compute_mauve(generated_texts, reference_texts, device=device)
    print(f"[bpe_mauve] MAUVE = {mauve_score:.4f}", flush=True)

    result = {
        "model": "diffusionblocks_ar_bpe_mauve",
        "architecture": "joint_sequence_selfattn_bpe",
        "vocab": "gpt2_bpe",
        "vocab_size": vocab_size,
        "n_params": n_params,
        "sigma_data_calibrated": float(model.sigma_data.item()),
        "train_steps": args.steps,
        "per_block_loss_final": final_loss,
        "block2_loss_still_falling": None,  # fill manually from loss_curve.csv
        "mauve_score": mauve_score,
        "mauve_n_prompts": args.n_prompts,
        "mauve_high_variance_warning": True,  # n=250 < 1000 recommended
        "samples": [
            {"prompt": tokenizer.decode(prompts[i]),
             "generated": generated_texts[i],
             "reference": reference_texts[i]}
            for i in range(min(5, len(generated_texts)))
        ],
        "config": vars(args),
    }
    (out_dir / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    print(f"[bpe_mauve] wrote {out_dir / 'result.json'}", flush=True)
    print(f"[bpe_mauve] MAUVE={mauve_score:.4f}  "
          f"params={n_params/1e6:.2f}M  steps={args.steps}", flush=True)


if __name__ == "__main__":
    main()
