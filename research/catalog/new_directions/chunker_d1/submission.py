"""D1 surprise-rate diagnostic for the Schmidhuber chunker.

Per research/catalog/new_directions/spec_16_chunker.md Phase 0 (D1).

What this does:
  1. Trains a 2-layer, d=128, seq_len=512 byte-level transformer for ~60s
     using the same primitives as submissions/modded_nanogpt.
  2. On the last 1M post-warmup training bytes, runs the trained automatizer
     in inference mode and computes per-byte P_L(true_byte | context).
  3. Reports the surprise rate p_s(tau) = fraction of bytes with
     P_L(true_byte) < tau for tau in {0.01, 0.05, 0.1, 0.3, 0.5}.
  4. Prints a marker-bracketed JSON block to stdout so submit.py's run.log
     captures the result. Returns a dummy CharModel that always predicts
     a space; the submission WILL DQ on accuracy. That is the intended path
     - we only care about the D1 report.

This file deliberately follows the modded_nanogpt primitives (RMSNorm,
RoPE, CausalSelfAttention, MLP, Muon, AdamW, ReLU^2) so the Phase 1
chunker can reuse them.
"""
from __future__ import annotations

__author__ = "@ab-10"

import json
import math
import os
import time

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.optim import AdamW

from wikitext import CharModel


# ---------------------------------------------------------------------------
# Tiny modded-nanogpt-style automatizer (2 layers, d=128, heads=4, T=512)
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
    def __init__(self, dim: int, head_dim: int = 32):
        super().__init__()
        self.num_heads = dim // head_dim
        self.head_dim = head_dim
        hdim = self.num_heads * self.head_dim
        self.q = Linear(dim, hdim)
        self.k = Linear(dim, hdim)
        self.v = Linear(dim, hdim)
        self.proj = Linear(hdim, dim)
        self.rotary = Rotary(head_dim)

    def forward(self, x: Tensor, offset: int = 0) -> Tensor:
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
        y = F.scaled_dot_product_attention(q, k, v, scale=0.12, is_causal=True)
        y = y.transpose(1, 2).contiguous().view(B, T, self.num_heads * self.head_dim)
        return self.proj(y)


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

    def forward(self, x: Tensor) -> Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class TinyGPT(nn.Module):
    def __init__(self, vocab_size: int, num_layers: int, model_dim: int, head_dim: int):
        super().__init__()
        self.vocab_size = vocab_size
        self.embed = nn.Embedding(vocab_size, model_dim).bfloat16()
        self.blocks = nn.ModuleList([Block(model_dim, head_dim=head_dim) for _ in range(num_layers)])
        self.proj = Linear(model_dim, vocab_size)
        self.norm1 = RMSNorm(model_dim)
        self.norm2 = RMSNorm(model_dim)

    def forward(self, inputs: Tensor) -> Tensor:
        x = self.norm1(self.embed(inputs))
        for block in self.blocks:
            x = block(x)
        logits = self.proj(self.norm2(x)).float()
        logits = 15 * logits * (logits.square() + 15**2).rsqrt()
        return logits


# ---------------------------------------------------------------------------
# Muon optimizer (single-GPU)
# ---------------------------------------------------------------------------

def zeropower_via_newtonschulz5(G: Tensor) -> Tensor:
    assert G.ndim >= 2
    X = G.bfloat16()
    if G.size(-2) > G.size(-1):
        X = X.mT
    X = X / (X.norm(dim=(-2, -1), keepdim=True) + 1e-7)
    a, b, c = 2, -1.5, 0.5
    for _ in range(12):
        A = X @ X.mT
        B = b * A + c * A @ A
        X = a * X + B @ X
    if G.size(-2) > G.size(-1):
        X = X.mT
    return X


def muon_update(grad: Tensor, momentum: Tensor, mu: float = 0.95, nesterov: bool = True) -> Tensor:
    momentum.lerp_(grad, 1 - mu)
    update = grad.lerp_(momentum, mu) if nesterov else momentum
    update = zeropower_via_newtonschulz5(update)
    update *= max(1, grad.size(-2) / grad.size(-1)) ** 0.5
    return update


class Muon(torch.optim.Optimizer):
    def __init__(self, params, lr: float = 0.02, weight_decay: float = 0.0, mu: float = 0.95):
        params = list(params)
        assert len(params) >= 1 and isinstance(params[0], torch.nn.Parameter)
        defaults = dict(lr=lr, weight_decay=weight_decay, mu=mu)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self):  # type: ignore[override]
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p]
                if len(state) == 0:
                    state["momentum"] = torch.zeros_like(p)
                update = muon_update(p.grad, state["momentum"], mu=group["mu"])
                p.mul_(1 - group["lr"] * group["weight_decay"])
                p.add_(update, alpha=-group["lr"])


def _init_modded(model: TinyGPT) -> None:
    for name, p in model.named_parameters():
        w = p.data
        if name.endswith("weight"):
            if "proj" in name:
                w.zero_()
            elif "embed" in name:
                w.normal_()
            else:
                w.normal_(std=0.33**0.5 / w.size(-1) ** 0.5)
        elif name.endswith("bias"):
            w.zero_()
        elif name.endswith("gains"):
            w.normal_(mean=1, std=0)
        else:
            raise RuntimeError(f"Uninitialized parameter: {name}")


# ---------------------------------------------------------------------------
# Dummy CharModel — predicts space. Will DQ on accuracy; that is fine.
# ---------------------------------------------------------------------------

class DummyCharModel(CharModel):
    def reset(self) -> None:
        pass

    def predict(self) -> dict[str, float]:
        return {" ": 1.0}

    def observe(self, char: str) -> None:
        pass


# ---------------------------------------------------------------------------
# D1 diagnostic
# ---------------------------------------------------------------------------

D1_TRAIN_SECONDS = 60.0       # ~60 s of training
D1_DIAG_BYTES = 1_000_000     # final 1M training bytes for diagnosis
D1_SEQ_LEN = 512
D1_BATCH_SIZE = 32
D1_DIAG_BATCH = 16            # smaller batch for the diagnostic forward pass
TAUS = (0.01, 0.05, 0.1, 0.3, 0.5)


def _measure_surprise_rate(
    model: TinyGPT,
    train_bytes: Tensor,
    device: torch.device,
    seq_len: int,
    n_target_bytes: int,
    batch_size: int,
) -> dict:
    """Run model on contiguous windows of the LAST n_target_bytes of training,
    compute per-byte P_L(true_byte | context), and return surprise stats.
    """
    n = train_bytes.numel()
    # Use the tail of the corpus, post warm-up region.
    start = max(0, n - n_target_bytes - seq_len)
    end = n
    region = train_bytes[start:end]
    region_n = region.numel()
    # Effective predictable positions: positions 1..region_n-1 conditioned on
    # the preceding context. We slide non-overlapping (seq_len)-byte windows
    # across the region; first position of each window has no context so we
    # mask it out of the statistic.
    stride = seq_len
    # Number of windows fitting (need seq_len+1 bytes per window: input+target)
    n_windows = max(0, (region_n - 1) // stride)
    if n_windows == 0:
        return {"error": "not enough bytes for diagnostic"}

    model.eval()
    probs_true: list[Tensor] = []
    correct: list[Tensor] = []
    # Group windows into batches.
    with torch.no_grad():
        i = 0
        while i < n_windows:
            j = min(i + batch_size, n_windows)
            bs = j - i
            starts = torch.arange(bs, device=device) * stride + i * stride
            offsets = starts[:, None] + torch.arange(stride + 1, device=device)[None, :]
            flat = region[offsets].long()
            x = flat[:, :-1]
            y = flat[:, 1:]
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                logits = model(x)
            # Softmax along byte dim; we don't drop the first position here
            # since the model conditions on its embedding via the byte history.
            # The first byte's prediction has only 1 byte of context which is
            # weak but not invalid; include for simplicity.
            logp = F.log_softmax(logits.float(), dim=-1)
            # gather P_L(true_byte)
            p_true = logp.gather(-1, y.unsqueeze(-1)).squeeze(-1).exp()
            argmax = logits.argmax(dim=-1)
            probs_true.append(p_true.reshape(-1).cpu())
            correct.append((argmax == y).reshape(-1).cpu())
            i = j

    p = torch.cat(probs_true)
    c = torch.cat(correct)
    n_total = p.numel()

    surprise_rates = {}
    for tau in TAUS:
        rate = (p < tau).float().mean().item()
        surprise_rates[f"tau_{tau}"] = rate

    # Percentile/cdf snapshot for richer diagnosis
    quantiles = [0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99]
    qvals = torch.quantile(p, torch.tensor(quantiles)).tolist()

    acc = c.float().mean().item()
    mean_p_true = p.mean().item()
    return {
        "n_bytes_evaluated": int(n_total),
        "surprise_rates": surprise_rates,
        "p_true_quantiles": {f"q{int(q*100)}": v for q, v in zip(quantiles, qvals)},
        "automatizer_accuracy_on_diag": acc,
        "mean_p_true": mean_p_true,
    }


def _verdict(surprise_rates: dict) -> str:
    p10 = surprise_rates["tau_0.1"]
    p30 = surprise_rates["tau_0.3"]
    if p10 > 0.65:
        return "FAIL"
    if p10 <= 0.50 and p30 <= 0.70:
        return "PASS"
    if 0.50 < p10 <= 0.65:
        return "BORDERLINE"
    # p10 <= 0.50 but p30 > 0.70: treat as borderline (high-tau condition fails).
    return "BORDERLINE"


def train(train_text: str, valid_text: str | None = None) -> CharModel:
    print("=" * 72)
    print("[d1] CHUNKER PHASE-0 DIAGNOSTIC: surprise-rate measurement")
    print("=" * 72)
    print("[d1] this submission will DQ on accuracy (dummy CharModel).")
    print("[d1] result of interest: the D1_REPORT_JSON block at the end.")
    print("")

    seed_env = os.environ.get("SEED")
    if seed_env:
        torch.manual_seed(int(seed_env))
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(int(seed_env))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[d1] device: {device}")

    raw = train_text.encode("utf-8")
    train_bytes = torch.frombuffer(bytearray(raw), dtype=torch.uint8).to(device)
    n = train_bytes.numel()
    print(f"[d1] train bytes: {n:,}")

    seq_len = D1_SEQ_LEN
    batch_size = D1_BATCH_SIZE
    head_dim = 32  # d=128, heads=4
    model_dim = 128
    num_layers = 2

    model = TinyGPT(
        vocab_size=256,
        num_layers=num_layers,
        model_dim=model_dim,
        head_dim=head_dim,
    ).to(device)
    _init_modded(model)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[d1] model: {num_layers}-layer transformer, d={model_dim}, "
          f"heads={model_dim//head_dim}, T={seq_len}, params={n_params/1e6:.3f}M")

    block_2d = [p for p in model.blocks.parameters() if p.ndim >= 2]
    scalars = [p for p in model.parameters() if p.ndim < 2]
    opt_adam = AdamW(
        [
            dict(params=[model.embed.weight], lr=0.3),
            dict(params=[model.proj.weight], lr=1.0 / 320),
            dict(params=scalars, lr=0.01),
        ],
        betas=(0.8, 0.95),
        eps=1e-10,
        weight_decay=0.0,
        fused=(device.type == "cuda"),
    )
    opt_muon = Muon(block_2d, lr=0.035, weight_decay=0.025)

    print(f"[d1] training for ~{D1_TRAIN_SECONDS:.0f}s ...")
    model.train()
    t0 = time.monotonic()
    step = 0
    last_loss = float("nan")
    while time.monotonic() - t0 < D1_TRAIN_SECONDS:
        idx = torch.randint(0, n - seq_len - 1, (batch_size,), device=device)
        offsets = idx[:, None] + torch.arange(seq_len + 1, device=device)[None, :]
        flat = train_bytes[offsets].long()
        x = flat[:, :-1]
        y = flat[:, 1:]
        opt_adam.zero_grad(set_to_none=True)
        opt_muon.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            logits = model(x)
            loss = F.cross_entropy(logits.reshape(-1, 256), y.reshape(-1))
        loss.backward()
        opt_adam.step()
        opt_muon.step()
        last_loss = loss.item()
        step += 1
        if step % 200 == 0:
            print(f"[d1] step {step}  loss {last_loss:.4f}  "
                  f"elapsed {time.monotonic()-t0:.1f}s", flush=True)
    train_elapsed = time.monotonic() - t0
    print(f"[d1] trained {step} steps in {train_elapsed:.1f}s, "
          f"final loss {last_loss:.4f}")

    print(f"[d1] measuring surprise rate on last {D1_DIAG_BYTES:,} bytes ...")
    t_diag = time.monotonic()
    diag = _measure_surprise_rate(
        model,
        train_bytes,
        device,
        seq_len=seq_len,
        n_target_bytes=D1_DIAG_BYTES,
        batch_size=D1_DIAG_BATCH,
    )
    diag_elapsed = time.monotonic() - t_diag
    diag["diagnostic_duration_s"] = diag_elapsed
    diag["train_seconds_used"] = train_elapsed
    diag["train_steps"] = step
    diag["final_train_loss"] = last_loss
    diag["model_params"] = n_params
    diag["model_arch"] = {
        "num_layers": num_layers,
        "model_dim": model_dim,
        "heads": model_dim // head_dim,
        "seq_len": seq_len,
        "batch_size": batch_size,
    }
    diag["verdict"] = _verdict(diag["surprise_rates"])

    print(f"[d1] diagnostic completed in {diag_elapsed:.1f}s.")
    print("[d1] surprise-rate table:")
    for tau in TAUS:
        rate = diag["surprise_rates"][f"tau_{tau}"]
        print(f"[d1]   p_s(tau={tau}) = {rate:.4f}")
    print(f"[d1] automatizer accuracy on diagnostic region: "
          f"{diag['automatizer_accuracy_on_diag']:.4f}")
    print(f"[d1] mean P_L(true): {diag['mean_p_true']:.4f}")
    print(f"[d1] verdict: {diag['verdict']}")
    print("")
    print("D1_REPORT_JSON_BEGIN")
    print(json.dumps(diag, indent=2))
    print("D1_REPORT_JSON_END")
    print("")
    print("[d1] returning dummy CharModel; expecting accuracy DQ.")
    return DummyCharModel()
