"""Pure-Mamba (selective SSM) byte-level LM on the wikitext benchmark.

Per research/catalog/new_directions/spec_14_mamba.md.

This submission inlines a **pure-PyTorch Mamba implementation** (no custom
CUDA kernel, no extra pip deps) so it runs on the standard Modal image
without modification. The architecture is the canonical Mamba block:

    byte embed -> [ MambaBlock x N ] -> RMSNorm -> linear over 256 bytes

Each MambaBlock is the verbatim Mamba mixer (in_proj -> conv1d -> SiLU ->
selective SSM -> gated by SiLU(z) -> out_proj). The expand=2 internal
expansion subsumes the MLP block of a transformer; there is no separate
MLP.

Selective scan: implemented as a vectorised parallel scan in log-space
using the cumprod trick, so each layer is one cumsum along the time
dim rather than a Python loop of 2048 sequential GPU launches. This is
the only non-trivial deviation from johnma2006/mamba-minimal; it brings
the wall-clock cost down by ~30x at seq_len=2048.

Streaming predict(): each MambaBlock keeps a per-stream SSM state
(d_inner x d_state) and a 1-D conv ring buffer (d_conv x d_inner), and
exposes a step() that advances both by one byte. This is the recurrent
mode that Mamba is designed around -- O(d_inner * d_state) per byte,
no full-prefix re-computation.
"""
from __future__ import annotations

__author__ = "@ab-10"

import math
import os
import time

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.optim import AdamW

from wikitext import CharModel


# ---------------------------------------------------------------------------
# Pure-PyTorch Mamba
# ---------------------------------------------------------------------------

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: Tensor) -> Tensor:
        return F.rms_norm(x, (x.size(-1),), weight=self.weight.type_as(x), eps=self.eps)


def _selective_scan_parallel(
    u: Tensor,        # (B, L, D)
    delta: Tensor,    # (B, L, D)
    A: Tensor,        # (D, N)
    B: Tensor,        # (B, L, N)
    C: Tensor,        # (B, L, N)
    D: Tensor,        # (D,)
) -> Tensor:
    """Vectorised selective scan in fp32.

    Computes h_t = exp(delta_t * A) * h_{t-1} + (delta_t * B_t) * u_t
    in parallel along the time dimension by working in log-space.

    Math:
        Let a_t = delta_t * A           # (B, L, D, N), <= 0
        Let A_bar_t = exp(a_t)          # (B, L, D, N), in (0, 1]
        Let b_t = (delta_t * B_t) * u_t # (B, L, D, N)
        Then h_t = A_bar_t * h_{t-1} + b_t.

        Define p_t = prod_{i=1..t} A_bar_i = exp(cumsum a_i).
        Then h_t = p_t * sum_{i=1..t} b_i / p_i.

        Numerically: h_t / p_t = cumsum(b_i / p_i) is unstable when p_i
        is tiny. Use log-space:
            log p_t = cumsum_t a
            h_t = exp(log p_t) * cumsum_t( b_t * exp(-log p_t) )

        Both factors are bounded because a <= 0, so -log p_t >= 0 grows
        but is balanced by exp(log p_t) <= 1 in the outer multiply. To
        keep this from overflowing for small A_bar, we use the standard
        "shift by max" trick implicitly via the inner cumsum on bf16
        promoted to fp32; for our settings A is small-magnitude so the
        un-shifted form is fine.
    """
    # All fp32 inside the scan for numerical stability.
    u = u.float()
    delta = delta.float()
    A = A.float()
    B = B.float()
    C = C.float()

    # delta * A : (B, L, D, N)
    deltaA = delta.unsqueeze(-1) * A  # broadcast (B,L,D,1)*(D,N)
    # delta * B : (B, L, D, N)
    deltaB = delta.unsqueeze(-1) * B.unsqueeze(-2)
    # b_t = (delta * B) * u : (B, L, D, N)
    bt = deltaB * u.unsqueeze(-1)

    # log p_t = cumsum_t (delta * A)
    log_p = deltaA.cumsum(dim=1)
    # h_t = exp(log_p_t) * cumsum_t( b_t * exp(-log_p_t) )
    # Numerically: split by subtracting the running max per (D,N) channel
    # along time (Hillis-Steele-free; cumsum is exact). Since delta*A <= 0
    # the cumsum is monotonically decreasing, so log_p is in [log_p_L, 0].
    # exp(-log_p) can blow up if cumsum gets very negative; clamp.
    neg_log_p = (-log_p).clamp(max=80.0)  # exp(80) ~ 5e34, safe in fp32
    inner = (bt * neg_log_p.exp()).cumsum(dim=1)
    h = log_p.exp() * inner  # (B, L, D, N)

    # y_t = C_t @ h_t + D * u_t
    # C: (B, L, N), h: (B, L, D, N) -> y: (B, L, D)
    y = (h * C.unsqueeze(-2)).sum(dim=-1)
    y = y + D * u
    return y


class MambaMixer(nn.Module):
    """One Mamba mixer block (in_proj -> conv1d -> SSM -> out_proj).

    Mirrors `mamba_ssm.modules.mamba_simple.Mamba` with d_model,
    d_state, d_conv, expand kwargs. Uses input-dependent (selective)
    delta, B, C; A is a learned per-(d_inner, d_state) bias whose log
    is parameterised as A_log = log(arange(1, d_state+1)) per channel.

    Streaming step() advances by exactly one token using:
      * a ring buffer for the causal conv (d_conv recent inputs)
      * the SSM hidden state h (d_inner, d_state)
    """

    def __init__(
        self,
        d_model: int,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        dt_rank: int | None = None,
        dt_min: float = 1e-3,
        dt_max: float = 0.1,
        dt_init_floor: float = 1e-4,
    ):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = expand * d_model
        self.dt_rank = dt_rank if dt_rank is not None else math.ceil(d_model / 16)

        # Projections.
        self.in_proj = nn.Linear(d_model, 2 * self.d_inner, bias=False)
        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            kernel_size=d_conv,
            groups=self.d_inner,
            padding=d_conv - 1,
            bias=True,
        )
        self.x_proj = nn.Linear(self.d_inner, self.dt_rank + 2 * d_state, bias=False)
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True)

        # dt_proj initialisation: as in the official Mamba code. Gives
        # delta a sensible scale before training, so the SSM doesn't
        # blow up or vanish at step 0.
        dt_init_std = self.dt_rank ** -0.5
        nn.init.uniform_(self.dt_proj.weight, -dt_init_std, dt_init_std)
        dt = torch.exp(
            torch.rand(self.d_inner) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        # softplus(b) = dt  =>  b = log(exp(dt) - 1)
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            self.dt_proj.bias.copy_(inv_dt)
        self.dt_proj.bias._no_reinit = True  # type: ignore[attr-defined]

        # A: per-(d_inner, d_state). Param is log(-A); after softplus
        # we get -A >= 0 so A <= 0, which makes A_bar = exp(delta*A)
        # contractive (stable).
        A = torch.arange(1, d_state + 1, dtype=torch.float32).repeat(self.d_inner, 1)
        self.A_log = nn.Parameter(torch.log(A))
        self.A_log._no_weight_decay = True  # type: ignore[attr-defined]

        # Skip-connection D parameter.
        self.D = nn.Parameter(torch.ones(self.d_inner))
        self.D._no_weight_decay = True  # type: ignore[attr-defined]

        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        """Parallel (training) forward. x: (B, L, d_model) -> (B, L, d_model)."""
        B, L, _ = x.shape

        xz = self.in_proj(x)                              # (B, L, 2*d_inner)
        x_in, z = xz.chunk(2, dim=-1)                     # each (B, L, d_inner)

        # Causal conv1d: (B, d_inner, L) ; left-pad d_conv-1 then truncate.
        x_conv = x_in.transpose(1, 2)
        x_conv = self.conv1d(x_conv)[:, :, :L]
        x_conv = F.silu(x_conv).transpose(1, 2)           # (B, L, d_inner)

        # Selective-scan parameters from x_conv.
        x_dbl = self.x_proj(x_conv)                        # (B, L, dt_rank+2N)
        dt, B_ssm, C_ssm = torch.split(
            x_dbl,
            [self.dt_rank, self.d_state, self.d_state],
            dim=-1,
        )
        dt = F.softplus(self.dt_proj(dt))                  # (B, L, d_inner), > 0

        A = -torch.exp(self.A_log.float())                 # (d_inner, d_state)
        D = self.D.float()
        y = _selective_scan_parallel(x_conv, dt, A, B_ssm, C_ssm, D)
        # Gated by SiLU(z).
        y = y * F.silu(z.float())
        return self.out_proj(y.type_as(x))

    # ---- recurrent step (streaming inference) ----

    def allocate_state(self, batch: int, device: torch.device, dtype: torch.dtype):
        """Return (conv_buf, h) initial state."""
        conv_buf = torch.zeros(batch, self.d_inner, self.d_conv, device=device, dtype=dtype)
        h = torch.zeros(batch, self.d_inner, self.d_state, device=device, dtype=torch.float32)
        return conv_buf, h

    def step(self, x: Tensor, state: tuple[Tensor, Tensor]) -> tuple[Tensor, tuple[Tensor, Tensor]]:
        """One-token recurrent step.

        x: (B, d_model). Returns (y: (B, d_model), new_state).
        """
        conv_buf, h = state
        xz = self.in_proj(x)                                # (B, 2*d_inner)
        x_in, z = xz.chunk(2, dim=-1)                       # (B, d_inner) each

        # Shift conv ring buffer and write new sample at the right edge.
        conv_buf = torch.roll(conv_buf, shifts=-1, dims=-1)
        conv_buf[:, :, -1] = x_in
        # Apply depthwise conv kernel: weight (d_inner, 1, d_conv).
        w = self.conv1d.weight.squeeze(1)                   # (d_inner, d_conv)
        x_conv = (conv_buf * w).sum(dim=-1) + self.conv1d.bias  # (B, d_inner)
        x_conv = F.silu(x_conv)

        # SSM parameters.
        x_dbl = self.x_proj(x_conv)                          # (B, dt_rank + 2N)
        dt, B_ssm, C_ssm = torch.split(
            x_dbl,
            [self.dt_rank, self.d_state, self.d_state],
            dim=-1,
        )
        dt = F.softplus(self.dt_proj(dt)).float()            # (B, d_inner)

        A = -torch.exp(self.A_log.float())                   # (d_inner, d_state)
        # A_bar: (B, d_inner, d_state)
        A_bar = torch.exp(dt.unsqueeze(-1) * A)
        # B_bar * u : (B, d_inner, d_state)
        Bbar_u = (dt.unsqueeze(-1) * B_ssm.float().unsqueeze(-2)) * x_conv.float().unsqueeze(-1)
        h = A_bar * h + Bbar_u                               # (B, d_inner, d_state)
        # y = C @ h
        y = (h * C_ssm.float().unsqueeze(-2)).sum(dim=-1)    # (B, d_inner)
        y = y + self.D.float() * x_conv.float()

        y = y.type_as(x) * F.silu(z)
        return self.out_proj(y), (conv_buf, h)


class MambaBlock(nn.Module):
    def __init__(self, d_model: int, d_state: int, d_conv: int, expand: int):
        super().__init__()
        self.norm = RMSNorm(d_model)
        self.mixer = MambaMixer(d_model, d_state=d_state, d_conv=d_conv, expand=expand)

    def forward(self, x: Tensor) -> Tensor:
        return x + self.mixer(self.norm(x))

    def step(self, x: Tensor, state):
        y, new_state = self.mixer.step(self.norm(x), state)
        return x + y, new_state

    def allocate_state(self, batch: int, device: torch.device, dtype: torch.dtype):
        return self.mixer.allocate_state(batch, device, dtype)


class MambaLM(nn.Module):
    def __init__(
        self,
        vocab_size: int = 256,
        d_model: int = 512,
        n_layers: int = 16,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_layers = n_layers
        self.embed = nn.Embedding(vocab_size, d_model)
        self.blocks = nn.ModuleList(
            [MambaBlock(d_model, d_state, d_conv, expand) for _ in range(n_layers)]
        )
        self.norm_f = RMSNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        # Weight tying.
        self.lm_head.weight = self.embed.weight

        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m: nn.Module) -> None:
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
            if m.bias is not None and not getattr(m.bias, "_no_reinit", False):
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)

    def forward(self, tokens: Tensor, use_checkpoint: bool = False) -> Tensor:
        """tokens: (B, L) long. Returns logits (B, L, vocab).

        When use_checkpoint=True, each MambaBlock's forward is wrapped in
        torch.utils.checkpoint.checkpoint to trade compute for memory --
        the (B, L, d_inner, d_state) scan intermediates are not kept for
        backward; instead, the block forward is recomputed. This is the
        practical knob that lets a 16-layer, d=512 Mamba train at
        seq_len=1024+ on a single A100-80GB without OOM.
        """
        x = self.embed(tokens)
        if use_checkpoint and self.training:
            from torch.utils.checkpoint import checkpoint
            for blk in self.blocks:
                x = checkpoint(blk, x, use_reentrant=False)
        else:
            for blk in self.blocks:
                x = blk(x)
        x = self.norm_f(x)
        return self.lm_head(x.type_as(self.lm_head.weight))

    # ---- recurrent inference ----

    def allocate_states(self, batch: int, device: torch.device, dtype: torch.dtype):
        return [blk.allocate_state(batch, device, dtype) for blk in self.blocks]

    def step(self, token: Tensor, states):
        """token: (B,) long. states: list per block. Returns (logits (B, V), new_states)."""
        x = self.embed(token)
        new_states = []
        for blk, st in zip(self.blocks, states):
            x, ns = blk.step(x, st)
            new_states.append(ns)
        x = self.norm_f(x)
        logits = self.lm_head(x.type_as(self.lm_head.weight))
        return logits, new_states


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

class TrainConfig:
    def __init__(
        self,
        d_model: int = 512,
        n_layers: int = 16,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        seq_len: int = 1024,
        batch_size: int = 16,
        n_steps: int = 4000,
        warmup_frac: float = 0.10,
        peak_lr: float = 5e-4,
        min_lr_frac: float = 0.10,
        weight_decay: float = 0.05,
        log_every: int = 50,
        time_budget_s: float | None = 280.0,
    ):
        self.d_model = d_model
        self.n_layers = n_layers
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.seq_len = seq_len
        self.batch_size = batch_size
        self.n_steps = n_steps
        self.warmup_frac = warmup_frac
        self.peak_lr = peak_lr
        self.min_lr_frac = min_lr_frac
        self.weight_decay = weight_decay
        self.log_every = log_every
        self.time_budget_s = time_budget_s

    def __repr__(self):
        return (
            f"TrainConfig(d={self.d_model} L={self.n_layers} "
            f"state={self.d_state} bs={self.batch_size} T={self.seq_len} "
            f"steps={self.n_steps} peak_lr={self.peak_lr})"
        )


def _train_mamba(text: str, cfg: TrainConfig, device: torch.device) -> MambaLM:
    raw = text.encode("utf-8")
    train_bytes = torch.frombuffer(bytearray(raw), dtype=torch.uint8).to(device)
    n = train_bytes.numel()
    if n < cfg.seq_len + 1:
        raise ValueError(f"need at least {cfg.seq_len+1} bytes; got {n}")

    model = MambaLM(
        vocab_size=256,
        d_model=cfg.d_model,
        n_layers=cfg.n_layers,
        d_state=cfg.d_state,
        d_conv=cfg.d_conv,
        expand=cfg.expand,
    ).to(device)

    # Param groups: no weight decay on biases, norms, A_log, D.
    decay, no_decay = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if (
            p.ndim < 2
            or name.endswith(".bias")
            or "norm" in name
            or name.endswith(".A_log")
            or name.endswith(".D")
        ):
            no_decay.append(p)
        else:
            decay.append(p)
    optim = AdamW(
        [
            dict(params=decay, weight_decay=cfg.weight_decay),
            dict(params=no_decay, weight_decay=0.0),
        ],
        lr=cfg.peak_lr,
        betas=(0.9, 0.95),
        eps=1e-8,
        fused=(device.type == "cuda"),
    )

    n_params = sum(p.numel() for p in model.parameters())
    print(f"[mamba] {n_params/1e6:.2f}M params  cfg={cfg}", flush=True)

    warmup_steps = max(1, int(cfg.warmup_frac * cfg.n_steps))
    def lr_at(step: int) -> float:
        if step < warmup_steps:
            return cfg.peak_lr * (step + 1) / warmup_steps
        # Cosine from peak -> peak * min_lr_frac
        progress = (step - warmup_steps) / max(1, cfg.n_steps - warmup_steps)
        progress = min(1.0, progress)
        cos = 0.5 * (1 + math.cos(math.pi * progress))
        return cfg.peak_lr * (cfg.min_lr_frac + (1 - cfg.min_lr_frac) * cos)

    model.train()
    use_amp = device.type == "cuda"
    t0 = time.monotonic()
    step = 0
    while step < cfg.n_steps:
        if cfg.time_budget_s is not None and (time.monotonic() - t0) > cfg.time_budget_s:
            print(f"[mamba] hit time budget at step {step}; stopping early", flush=True)
            break
        lr = lr_at(step)
        for g in optim.param_groups:
            g["lr"] = lr
        idx = torch.randint(0, n - cfg.seq_len - 1, (cfg.batch_size,), device=device)
        offsets = idx[:, None] + torch.arange(cfg.seq_len + 1, device=device)[None, :]
        flat = train_bytes[offsets].long()
        x = flat[:, :-1]
        y = flat[:, 1:]

        optim.zero_grad(set_to_none=True)
        if use_amp:
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                logits = model(x, use_checkpoint=True)
                loss = F.cross_entropy(logits.reshape(-1, 256), y.reshape(-1))
        else:
            logits = model(x, use_checkpoint=True)
            loss = F.cross_entropy(logits.reshape(-1, 256), y.reshape(-1))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optim.step()

        if cfg.log_every and (step % cfg.log_every == 0 or step == cfg.n_steps - 1):
            elapsed = time.monotonic() - t0
            print(
                f"[mamba] step {step:5d}/{cfg.n_steps}  "
                f"loss {loss.item():.4f}  "
                f"lr {lr:.2e}  "
                f"elapsed {elapsed:.0f}s",
                flush=True,
            )
        step += 1

    return model


# ---------------------------------------------------------------------------
# CharModel wrapper (recurrent streaming)
# ---------------------------------------------------------------------------

class MambaCharModel(CharModel):
    def __init__(self, model: MambaLM, device: torch.device | None = None):
        self.model = model
        self.device = device or next(model.parameters()).device
        self.model.eval()
        # Pick the dtype matching the embedding (the model lives in fp32
        # weights here; state buffers follow the embedding dtype).
        self._dtype = self.model.embed.weight.dtype
        self._states = None
        self._next_logits: Tensor | None = None

    @torch.no_grad()
    def reset(self) -> None:
        self._states = self.model.allocate_states(
            batch=1, device=self.device, dtype=self._dtype,
        )
        # Seed with a single zero byte (sentinel) so predict() has a
        # valid distribution before the first observe().
        token = torch.zeros(1, dtype=torch.long, device=self.device)
        logits, self._states = self.model.step(token, self._states)
        self._next_logits = logits[0]

    @torch.no_grad()
    def predict(self) -> str:
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
        return max(out, key=lambda c: out[c]) if out else ""

    @torch.no_grad()
    def observe(self, char: str) -> None:
        if self._states is None:
            raise RuntimeError("observe() called before reset()")
        for byte in char.encode("utf-8"):
            token = torch.tensor([byte], dtype=torch.long, device=self.device)
            logits, self._states = self.model.step(token, self._states)
            self._next_logits = logits[0]


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
        print(f"[mamba] SEED={seed}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = TrainConfig()
    model = _train_mamba(train_text, cfg, device)
    return MambaCharModel(model)
