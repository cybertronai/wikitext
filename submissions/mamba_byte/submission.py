"""MambaByte tiny — selective state-space model on byte-level WikiText-103.

Paradigm: WTX-I023 (Mamba/SSM, linear-time sequence model). Claude-tag
CLA-005-adjacent. Hypothesis: at byte granularity a selective SSM with
linear-time complexity can chew through a longer context window than
quadratic attention at the same wall-clock, picking up cheap accuracy
from local format / repetition structure.

Reference:
  * Gu & Dao 2024, "Mamba: Linear-Time Sequence Modeling with Selective
    State Spaces" (arxiv 2312.00752).
  * Wang et al. 2024, "MambaByte: Token-free Selective State Space
    Model" (arxiv 2401.13660).

Implementation choice
---------------------
We use a **pure-PyTorch selective SSM** — no ``mamba-ssm`` /
``causal-conv1d`` CUDA kernels. Rationale:

* The Modal ``ghcr.io/ab-10/wikitext-bench:latest`` image bundles torch
  2.5.1+cu124 but NOT ``mamba-ssm``. Installing at ``train()`` time
  would burn ~30-60 s of the 300 s wall-clock cap and is brittle
  (sdist builds against a specific torch ABI).
* The pure-PyTorch fallback is built from ``torch.cumsum`` / ``exp``
  primitives that already fuse well in bf16 on A100. We give up the
  fully-fused selective_scan_cuda speedup but keep the asymptotic
  O(n*d_state) memory and time.

Architecture
------------
4 stacked Mamba blocks, each consisting of:

  x -> in_proj -> (z, x') with expand=2
  x' -> conv1d(kernel=4, causal) -> silu -> selective_ssm
  z  -> silu                       -> gate
  out = out_proj(ssm_out * gate)

The selective SSM is the standard discretized form

  h_t = exp(dt * A) * h_{t-1} + dt * B * x_t
  y_t = C * h_t + D * x_t

with ``dt``, ``B``, ``C`` data-dependent (selective) and ``A`` a
learned negative real diagonal initialized as ``-(1..N)``.

Streaming
---------
The Mamba block has a tiny O(1) recurrent state:

  * conv1d window: last (d_conv - 1) inputs (per channel)
  * ssm hidden:    h of shape (d_inner, d_state)

The CharModel wrapper caches (conv_state, ssm_state) per layer and
takes one recurrent step per observed byte. This is the killer feature
of SSMs vs attention: streaming cost is independent of context length.
"""
from __future__ import annotations

__author__ = "@claude-mamba"

import math
import os
import time

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.optim import AdamW

from wikitext import CharModel


# ---------------------------------------------------------------------------
# Selective scan (pure PyTorch)
# ---------------------------------------------------------------------------

SCAN_CHUNK = 32  # chunk size for the chunked selective scan


def selective_scan(
    u: Tensor,       # (B, L, D)        — input x'
    delta: Tensor,   # (B, L, D)        — dt, already softplus'd, positive
    A: Tensor,       # (D, N)           — log-A (we'll exp(A_log) then negate)
    B_in: Tensor,    # (B, L, N)        — input projection of B
    C_in: Tensor,    # (B, L, N)        — input projection of C
    D_skip: Tensor,  # (D,)             — direct skip per channel
) -> Tensor:
    """Pure-PyTorch selective scan, chunked for numerical stability.

    Computes ``y_t = C_t @ h_t + D * x_t`` where
    ``h_t = exp(dt_t * A) * h_{t-1} + (dt_t * B_t) * x_t``.

    Within each chunk of size ``SCAN_CHUNK`` we use the standard
    log-cumsum parallel-scan trick:

        h_local_t = exp(cs_t) * sum_{k<=t} exp(-cs_{k-1}) * b_k

    where ``cs_t = sum_{j<=t} dt_j * A`` (negative). Across chunks we
    carry the running hidden state ``h_carry`` and add its contribution
    ``exp(cs_t) * h_carry`` to every position in the chunk. This keeps
    ``exp(-cs)`` bounded by ``exp(chunk_size * max|dt*A|)`` which is well
    inside fp32 range for our config (dt_max=0.1, A_max=8 -> per-step
    decay ~0.9, chunk-32 -> 32*0.9=28.8 in log-space, exp(28.8)=3e12,
    fine).

    Shapes: B=batch, L=seq, D=d_inner, N=d_state.
    Output: (B, L, D).
    """
    B, L, D = u.shape
    N = A.shape[-1]
    # A is parameterised as -exp(A_log) per Mamba convention.
    A_neg = -torch.exp(A.float())                                # (D, N)
    delta_f = delta.float()
    u_f = u.float()
    B_f = B_in.float()
    C_f = C_in.float()
    deltaA = delta_f.unsqueeze(-1) * A_neg                       # (B, L, D, N)
    deltaB_u = (
        delta_f.unsqueeze(-1) * B_f.unsqueeze(2) * u_f.unsqueeze(-1)
    )                                                            # (B, L, D, N)

    # Carry (running hidden state) across chunks.
    h_carry = u.new_zeros(B, D, N, dtype=torch.float32)
    out_chunks: list[Tensor] = []
    for start in range(0, L, SCAN_CHUNK):
        end = min(L, start + SCAN_CHUNK)
        log_decay = deltaA[:, start:end]                          # (B, T, D, N)
        b_t = deltaB_u[:, start:end]                              # (B, T, D, N)
        c_t = C_f[:, start:end]                                   # (B, T, N)

        cs = torch.cumsum(log_decay, dim=1)                       # (B, T, D, N)
        # h_local_t = exp(cs_t) * cumsum_k (b_k * exp(-cs_{k-1}))
        # with cs_{-1} = 0, i.e. inner_k = b_k * exp(log_decay_k - cs_k).
        inner = b_t * torch.exp(log_decay - cs)                   # (B, T, D, N)
        inner_cs = torch.cumsum(inner, dim=1)                     # (B, T, D, N)
        exp_cs = torch.exp(cs)                                    # (B, T, D, N)
        # Add carry contribution: h_t also includes exp(cs_t) * h_carry.
        h = exp_cs * (inner_cs + h_carry.unsqueeze(1))            # (B, T, D, N)

        y = (h * c_t.unsqueeze(2)).sum(dim=-1)                    # (B, T, D)
        out_chunks.append(y)

        # Update carry to the final state h_{end-1}, detached from this
        # chunk's exp_cs but with the gradient still flowing through the
        # recurrence implicitly via the next chunk's cumsum (which will
        # see h_carry as a leaf wrt that chunk). This is the standard
        # "scan with carry" pattern.
        h_carry = h[:, -1]                                        # (B, D, N)

    y_full = torch.cat(out_chunks, dim=1)                         # (B, L, D)
    y_full = y_full + D_skip.float() * u_f
    return y_full.to(u.dtype)


# ---------------------------------------------------------------------------
# Mamba block
# ---------------------------------------------------------------------------

class MambaBlock(nn.Module):
    """One selective-SSM block.

    Streaming-mode forward (T=1) uses a recurrent step that updates a
    small per-block state (conv buffer + ssm hidden). Training-mode
    forward (T>1, no state) uses the parallel selective_scan above.
    """

    def __init__(
        self,
        d_model: int,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        dt_rank: str | int = "auto",
    ):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.d_inner = expand * d_model
        if dt_rank == "auto":
            self.dt_rank = max(1, math.ceil(d_model / 16))
        else:
            self.dt_rank = int(dt_rank)

        # in_proj: x -> [x', z]
        self.in_proj = nn.Linear(d_model, 2 * self.d_inner, bias=False)
        # depthwise causal conv1d on x'
        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            kernel_size=d_conv,
            groups=self.d_inner,
            padding=d_conv - 1,
            bias=True,
        )
        # x_proj: x' -> [dt_low_rank, B, C]
        self.x_proj = nn.Linear(self.d_inner, self.dt_rank + 2 * d_state, bias=False)
        # dt projection: low-rank -> per-channel dt
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True)

        # A parameter (per-(inner, state) channel), parameterised as
        # log so A = -exp(A_log). Init A = -(1..N) repeated per channel.
        A = torch.arange(1, d_state + 1, dtype=torch.float32).unsqueeze(0)
        A = A.repeat(self.d_inner, 1)                            # (d_inner, N)
        self.A_log = nn.Parameter(torch.log(A))
        # D: skip-connection scalar per inner channel.
        self.D = nn.Parameter(torch.ones(self.d_inner))

        # out_proj: gated SSM output -> d_model.
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)

        # dt bias init: per Mamba paper, bias initialised so that
        # softplus(bias) is uniformly in [dt_min, dt_max]. We use
        # dt_min=1e-3, dt_max=1e-1.
        dt_init_std = self.dt_rank ** -0.5
        nn.init.uniform_(self.dt_proj.weight, -dt_init_std, dt_init_std)
        dt = torch.exp(
            torch.rand(self.d_inner) * (math.log(1e-1) - math.log(1e-3))
            + math.log(1e-3)
        ).clamp(min=1e-4)
        inv_dt = dt + torch.log(-torch.expm1(-dt))   # softplus^{-1}
        with torch.no_grad():
            self.dt_proj.bias.copy_(inv_dt)
        # Avoid re-init of dt_proj.bias by std init below.
        self.dt_proj.bias._no_reinit = True  # type: ignore[attr-defined]

    # -- training-mode forward (parallel scan over L) ----------------------

    def forward(self, x: Tensor) -> Tensor:
        """``x``: (B, L, d_model). Returns (B, L, d_model)."""
        B, L, _ = x.shape
        xz = self.in_proj(x)                                     # (B, L, 2*d_inner)
        x_, z = xz.chunk(2, dim=-1)                              # each (B, L, d_inner)

        # Depthwise causal conv1d: input (B, d_inner, L); we keep only the
        # first L outputs (since padding=d_conv-1, conv output length is
        # L + d_conv - 1).
        x_conv = self.conv1d(x_.transpose(1, 2))[:, :, :L]       # (B, d_inner, L)
        x_act = F.silu(x_conv).transpose(1, 2)                   # (B, L, d_inner)

        # Project to dt-low-rank, B, C.
        x_dbl = self.x_proj(x_act)                               # (B, L, dt_rank+2N)
        dt_low, B_in, C_in = x_dbl.split(
            [self.dt_rank, self.d_state, self.d_state], dim=-1
        )
        dt = F.softplus(self.dt_proj(dt_low))                    # (B, L, d_inner)

        y = selective_scan(x_act, dt, self.A_log, B_in, C_in, self.D)  # (B,L,d_inner)
        y = y * F.silu(z)
        return self.out_proj(y)

    # -- streaming-mode forward (single token, with state) -----------------

    @torch.no_grad()
    def step(
        self,
        x: Tensor,                                # (B, d_model)
        conv_state: Tensor,                       # (B, d_inner, d_conv)
        ssm_state: Tensor,                        # (B, d_inner, d_state)
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Single-step recurrent update. Returns (out, conv_state', ssm_state')."""
        xz = self.in_proj(x)                                     # (B, 2*d_inner)
        x_, z = xz.chunk(2, dim=-1)                              # each (B, d_inner)

        # Roll the conv window: drop oldest, append newest.
        conv_state = torch.roll(conv_state, shifts=-1, dims=-1)
        conv_state[:, :, -1] = x_
        # Conv1d weight: (d_inner, 1, d_conv). For depthwise we just elementwise-
        # multiply across the window and sum.
        w = self.conv1d.weight.squeeze(1)                        # (d_inner, d_conv)
        x_conv = (conv_state * w).sum(dim=-1) + self.conv1d.bias  # (B, d_inner)
        x_act = F.silu(x_conv)                                   # (B, d_inner)

        # Project per-token.
        x_dbl = self.x_proj(x_act)                               # (B, dt_rank+2N)
        dt_low, B_in, C_in = x_dbl.split(
            [self.dt_rank, self.d_state, self.d_state], dim=-1
        )
        dt = F.softplus(self.dt_proj(dt_low))                    # (B, d_inner)

        # Discretize and step the SSM.
        A_neg = -torch.exp(self.A_log.float())                   # (d_inner, N)
        # dt: (B, d_inner) -> (B, d_inner, 1); A_neg: (d_inner, N) -> (1, d_inner, N)
        deltaA = torch.exp(dt.float().unsqueeze(-1) * A_neg.unsqueeze(0))  # (B, d_inner, N)
        deltaB_u = (
            dt.float().unsqueeze(-1)                              # (B, d_inner, 1)
            * B_in.float().unsqueeze(1)                           # (B, 1, N)
            * x_act.float().unsqueeze(-1)                         # (B, d_inner, 1)
        )                                                         # (B, d_inner, N)
        ssm_state = deltaA * ssm_state.float() + deltaB_u
        y = (ssm_state * C_in.float().unsqueeze(1)).sum(dim=-1)  # (B, d_inner)
        y = y + self.D.float() * x_act.float()
        y = y.to(x.dtype) * F.silu(z)
        return self.out_proj(y), conv_state, ssm_state.to(x.dtype)


# ---------------------------------------------------------------------------
# Full model
# ---------------------------------------------------------------------------

class MambaLM(nn.Module):
    def __init__(
        self,
        vocab_size: int = 256,
        d_model: int = 192,
        n_layer: int = 4,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_layer = n_layer
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = expand * d_model

        self.embed = nn.Embedding(vocab_size, d_model)
        self.blocks = nn.ModuleList(
            [MambaBlock(d_model, d_state=d_state, d_conv=d_conv, expand=expand)
             for _ in range(n_layer)]
        )
        self.norms = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(n_layer)])
        self.norm_f = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        # Weight-tie head to embedding (saves params, common in byte LMs).
        self.lm_head.weight = self.embed.weight
        # Default nn.Embedding init is N(0, 1) which through the tied head
        # produces logits with std ~sqrt(d_model). For d_model=192 the
        # initial loss is ~185 vs the expected ln(256) ~ 5.55. Rescale
        # the embedding to GPT-style 0.02 std so the first step is sane.
        nn.init.normal_(self.embed.weight, mean=0.0, std=0.02)

    def forward(self, inputs: Tensor) -> Tensor:
        x = self.embed(inputs)
        for blk, norm in zip(self.blocks, self.norms):
            x = x + blk(norm(x))
        x = self.norm_f(x)
        return self.lm_head(x)

    @torch.no_grad()
    def step(
        self,
        token: Tensor,                                # (B,) long
        states: list[tuple[Tensor, Tensor]],          # per-layer (conv, ssm)
    ) -> tuple[Tensor, list[tuple[Tensor, Tensor]]]:
        x = self.embed(token)                         # (B, d_model)
        new_states: list[tuple[Tensor, Tensor]] = []
        for blk, norm, (cs, ss) in zip(self.blocks, self.norms, states):
            h, cs2, ss2 = blk.step(norm(x), cs, ss)
            x = x + h
            new_states.append((cs2, ss2))
        x = self.norm_f(x)
        logits = self.lm_head(x)                      # (B, vocab)
        return logits, new_states

    def init_states(self, batch_size: int, device: torch.device, dtype=torch.float32
                    ) -> list[tuple[Tensor, Tensor]]:
        states = []
        for _ in range(self.n_layer):
            cs = torch.zeros(batch_size, self.d_inner, self.d_conv,
                             device=device, dtype=dtype)
            ss = torch.zeros(batch_size, self.d_inner, self.d_state,
                             device=device, dtype=dtype)
            states.append((cs, ss))
        return states


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

class TrainConfig:
    def __init__(
        self,
        d_model: int = 192,
        n_layer: int = 4,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        # ctx_len 2048 / bs 64 OOM'd on A100-40GB: the chunked selective
        # scan materialises ~5-7 fp32 tensors of shape (B, L, d_inner, N)
        # per layer for backward, ~64-90 GB total at original config.
        # Halving both dims (16x memory cut) brings activations to
        # ~8-11 GB which fits with headroom.
        ctx_len: int = 1024,
        batch_size: int = 16,
        # Shrinking ctx*bs by 8x cuts per-step FLOPs by ~8x too, so we
        # have wall-clock headroom to take more steps and recover some
        # of the lost token throughput. 4000 steps at ~16k tokens/step
        # = 64M tokens trained, vs the original 1500x131k = 197M; still
        # less data than originally targeted but a defensible trade.
        n_steps: int = 4000,
        lr: float = 3e-4,
        weight_decay: float = 0.1,
        warmup_frac: float = 0.05,
        log_every: int = 100,
    ):
        self.d_model = d_model
        self.n_layer = n_layer
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.ctx_len = ctx_len
        self.batch_size = batch_size
        self.n_steps = n_steps
        self.lr = lr
        self.weight_decay = weight_decay
        self.warmup_frac = warmup_frac
        self.log_every = log_every

    def __repr__(self):
        return (f"TrainConfig(d={self.d_model} L={self.n_layer} "
                f"d_state={self.d_state} d_conv={self.d_conv} "
                f"expand={self.expand} ctx={self.ctx_len} "
                f"bs={self.batch_size} steps={self.n_steps})")


def _train_mamba(text: str, cfg: TrainConfig, device: torch.device) -> MambaLM:
    raw = text.encode("utf-8")
    train_bytes = torch.frombuffer(bytearray(raw), dtype=torch.uint8).to(device)
    n = train_bytes.numel()
    if n < cfg.ctx_len + 1:
        raise ValueError(f"need at least {cfg.ctx_len+1} bytes; got {n}")

    model = MambaLM(
        vocab_size=256,
        d_model=cfg.d_model,
        n_layer=cfg.n_layer,
        d_state=cfg.d_state,
        d_conv=cfg.d_conv,
        expand=cfg.expand,
    ).to(device)

    # AdamW with weight-decay split: don't decay 1-D params (norms, biases,
    # A_log, D).
    decay, no_decay = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if p.ndim < 2 or "A_log" in name or name.endswith(".D") or "norm" in name.lower():
            no_decay.append(p)
        else:
            decay.append(p)
    optimizer = AdamW(
        [
            dict(params=decay, weight_decay=cfg.weight_decay),
            dict(params=no_decay, weight_decay=0.0),
        ],
        lr=cfg.lr,
        betas=(0.9, 0.95),
        eps=1e-8,
        fused=(device.type == "cuda"),
    )

    n_params = sum(p.numel() for p in model.parameters())
    print(f"[mamba] {n_params/1e6:.2f}M params  cfg={cfg}", flush=True)

    warmup_steps = max(1, int(cfg.warmup_frac * cfg.n_steps))

    def set_lr(step: int) -> None:
        if step < warmup_steps:
            eta = step / warmup_steps
        else:
            progress = (step - warmup_steps) / max(1, cfg.n_steps - warmup_steps)
            eta = 0.5 * (1 + math.cos(math.pi * progress))
        for g in optimizer.param_groups:
            g["lr"] = cfg.lr * eta

    model.train()
    use_amp = device.type == "cuda"
    t0 = time.monotonic()
    for step in range(cfg.n_steps):
        set_lr(step)
        idx = torch.randint(0, n - cfg.ctx_len - 1, (cfg.batch_size,), device=device)
        offsets = idx[:, None] + torch.arange(cfg.ctx_len + 1, device=device)[None, :]
        flat = train_bytes[offsets].long()
        x = flat[:, :-1]
        y = flat[:, 1:]

        optimizer.zero_grad(set_to_none=True)
        if use_amp:
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                logits = model(x)
                loss = F.cross_entropy(logits.reshape(-1, 256), y.reshape(-1))
        else:
            logits = model(x)
            loss = F.cross_entropy(logits.reshape(-1, 256), y.reshape(-1))
        loss.backward()
        # Gradient clipping — selective-scan can produce sharp grads
        # through the cumsum/exp path. 1.0 is a safe default for SSMs.
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        if cfg.log_every and (step % cfg.log_every == 0 or step == cfg.n_steps - 1):
            elapsed = time.monotonic() - t0
            print(
                f"[mamba] step {step:5d}/{cfg.n_steps}  "
                f"loss {loss.item():.4f}  "
                f"elapsed {elapsed:.0f}s",
                flush=True,
            )

    return model


# ---------------------------------------------------------------------------
# Streaming CharModel wrapper
# ---------------------------------------------------------------------------

class MambaByteCharModel(CharModel):
    """Streaming Mamba CharModel.

    Per-byte cost is O(n_layer * d_inner * d_state) — independent of how
    many bytes have been observed. The state is just (conv_state,
    ssm_state) per layer.
    """

    def __init__(self, model: MambaLM, device: torch.device | None = None):
        self.model = model
        self.device = device or next(model.parameters()).device
        self.model.eval()
        self._states: list[tuple[Tensor, Tensor]] | None = None
        self._next_logits: Tensor | None = None

    @torch.no_grad()
    def reset(self) -> None:
        # Initialise zero state and seed with a single zero byte so the
        # first predict() has a valid distribution before any real char.
        # Same convention as the modded_nanogpt baseline.
        self._states = self.model.init_states(1, self.device, dtype=torch.float32)
        seed = torch.zeros(1, dtype=torch.long, device=self.device)
        logits, self._states = self.model.step(seed, self._states)
        self._next_logits = logits[0]

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
        if self._states is None:
            raise RuntimeError("observe() called before reset()")
        for byte in char.encode("utf-8"):
            token = torch.tensor([byte], dtype=torch.long, device=self.device)
            logits, self._states = self.model.step(token, self._states)
            self._next_logits = logits[0]


# ---------------------------------------------------------------------------
# Entry point — `submit.py` looks for this signature.
# ---------------------------------------------------------------------------

# Tiny-train threshold: below this many train bytes we shrink the config
# so the end-to-end smoke test runs in seconds on CPU.
SMOKE_TRAIN_BYTES = 10_000


def train(train_text: str, valid_text: str | None = None) -> CharModel:
    seed_env = os.environ.get("SEED")
    if seed_env:
        seed = int(seed_env)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        print(f"[mamba] SEED={seed}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    raw = train_text.encode("utf-8")
    is_smoke = len(raw) < SMOKE_TRAIN_BYTES or os.environ.get("SMOKE_TEST_ONLY") == "1"

    if is_smoke:
        # Shrink for end-to-end smoke: keep architecture shape (SSM block,
        # conv, gating) but slash compute. Clamp ctx_len to the corpus.
        ctx = max(8, min(64, max(8, len(raw) // 4)))
        cfg = TrainConfig(
            d_model=32,
            n_layer=2,
            d_state=8,
            d_conv=4,
            expand=2,
            ctx_len=ctx,
            batch_size=2,
            n_steps=2,
            log_every=0,
        )
        print(f"[mamba] SMOKE mode (train={len(raw)} bytes)  ctx={ctx}", flush=True)
    else:
        cfg = TrainConfig()

    model = _train_mamba(train_text, cfg, device)
    return MambaByteCharModel(model)
