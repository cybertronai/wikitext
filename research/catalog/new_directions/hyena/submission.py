"""Hyena over GPT-2 BPE tokens, with byte-level marginalisation at eval.

Per research/catalog/new_directions/spec_13_hyena.md and the README's
"Internal representations" clarification.

Hyena's published 20%-compute-reduction-on-WikiText-103 claim is at
*word/BPE* level. At byte level under the 300 s budget v1/v2 only fit
1200 steps; the implicit-filter MLP could not converge. BPE-internal
representation cuts the effective sequence length by ~4× (avg ~4 bytes
per GPT-2 BPE token), lifting the per-step throughput enough that the
filter MLP can actually learn inside the budget.

Architecture:
  * Tokeniser: GPT-2 BPE (deterministic merge tables; not a "pretrained
    weight" — see README "Internal representations" / rule 1).
  * Hyena stack over BPE tokens (vocab=50257, d=384, L=6, seq=512).
  * Output: P(next_bpe_token | committed_bpe_context).

Streaming eval (CharModel.predict → P(next_char | observed_chars)):
  Maintain (committed_tokens, partial_bytes), where partial_bytes are
  the bytes since the last token boundary that's stable under further
  byte arrivals. Run Hyena on committed_tokens to get a distribution
  over the next BPE token. For each candidate next byte c, marginalise
  over tokens whose UTF-8 bytes start with partial_bytes + (c, ...).

Training recipe stays modded-nanogpt-style: Muon for 2-D non-filter
weights, AdamW for embeddings / 1-D scalars / filter MLP. Window decay
and filter-MLP routing fixes from v2 are kept.
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
# Hyena operator (unchanged from v2 — filter routing + window decay fixes
# applied; only sequence-mixer-side, vocab-agnostic)
# ---------------------------------------------------------------------------

class _DtypeAwareLinear(nn.Linear):
    """Linear that casts weight/bias to the input dtype on each call.

    The token embedding is bfloat16 but stored params are fp32 — same
    pattern as modded-nanogpt's mixed-precision training path.
    """
    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__(in_features, out_features, bias=bias)

    def forward(self, x: Tensor) -> Tensor:
        return F.linear(
            x,
            self.weight.type_as(x),
            self.bias.type_as(x) if self.bias is not None else None,
        )


def causal_fft_conv(x: Tensor, h: Tensor) -> Tensor:
    """Causal 1-D convolution of x with h via FFT.

    x: (B, T, D)    h: (D, T)    returns (B, T, D).

    Standard recipe: zero-pad both to length 2T, rfft, multiply,
    irfft, slice first T samples. Done in fp32 because torch.fft does
    not support bf16 on CUDA.
    """
    B, T, D = x.shape
    x_ = x.transpose(1, 2).float()
    h_ = h.float()
    N = 2 * T
    Xf = torch.fft.rfft(x_, n=N)
    Hf = torch.fft.rfft(h_, n=N)
    Yf = Xf * Hf
    y = torch.fft.irfft(Yf, n=N)[..., :T]
    return y.transpose(1, 2).to(x.dtype)


class HyenaOperator(nn.Module):
    """Order-2 Hyena operator.

        v = x_0
        for i in 1..order:
            v = v * x_i
            v = conv(h_i, v)
        out = out_proj(v)

    Filter h_i comes from a tiny MLP mapping position -> filter value.
    Window decay magnitude is 1.0 (gentle) — v1's 4.0 over-attenuated
    long taps; the filter could not see distant positions. The filter
    MLP's parameters are routed to AdamW, never Muon (v1's Muon-on-
    Linear(1, 64) was the original failure mode).
    """
    def __init__(
        self,
        d_model: int,
        order: int = 2,
        seq_len: int = 512,
        filter_hidden: int = 64,
    ):
        super().__init__()
        self.d_model = d_model
        self.order = order
        self.seq_len = seq_len
        self.in_proj = _DtypeAwareLinear(d_model, (order + 1) * d_model)
        self.out_proj = _DtypeAwareLinear(d_model, d_model)
        self.filter_mlp = nn.Sequential(
            nn.Linear(1, filter_hidden, bias=True),
            nn.SiLU(),
            nn.Linear(filter_hidden, filter_hidden, bias=True),
            nn.SiLU(),
            nn.Linear(filter_hidden, order * d_model, bias=True),
        )
        pos = torch.arange(seq_len, dtype=torch.float32).unsqueeze(-1) / max(1, seq_len)
        self.register_buffer("pos", pos)
        decay = torch.exp(-torch.arange(seq_len, dtype=torch.float32) / max(1, seq_len) * 1.0)
        self.register_buffer("window", decay)

    def forward(self, x: Tensor) -> Tensor:
        B, T, D = x.shape
        assert D == self.d_model, f"channel mismatch: {D} vs {self.d_model}"
        assert T <= self.seq_len, f"seq {T} exceeds buffer {self.seq_len}"

        projs = self.in_proj(x).chunk(self.order + 1, dim=-1)

        pos = self.pos[:T].float()
        filt = self.filter_mlp(pos)
        filt = filt * self.window[:T].unsqueeze(-1)
        filt = filt.view(T, self.order, self.d_model).permute(1, 2, 0).contiguous()

        v = projs[0]
        for i in range(self.order):
            v = v * projs[i + 1]
            v = causal_fft_conv(v, filt[i])

        return self.out_proj(v)


# ---------------------------------------------------------------------------
# Backbone (Hyena replaces attention)
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
    def __init__(self, dim: int, seq_len: int, order: int = 2):
        super().__init__()
        self.mixer = HyenaOperator(dim, order=order, seq_len=seq_len)
        self.mlp = MLP(dim)
        self.norm1 = RMSNorm(dim)
        self.norm2 = RMSNorm(dim)

    def forward(self, x: Tensor) -> Tensor:
        x = x + self.mixer(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class HyenaLM(nn.Module):
    """Hyena LM with tied embedding/head. Large vocab (50257) makes
    embedding/head 19 M params each — tying halves that footprint.
    """
    def __init__(
        self,
        vocab_size: int,
        num_layers: int,
        model_dim: int,
        max_len: int = 512,
        hyena_order: int = 2,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.max_len = max_len
        self.embed = nn.Embedding(vocab_size, model_dim).bfloat16()
        self.blocks = nn.ModuleList(
            [Block(model_dim, seq_len=max_len, order=hyena_order)
             for _ in range(num_layers)]
        )
        self.norm1 = RMSNorm(model_dim)
        self.norm2 = RMSNorm(model_dim)
        # Tied head: reuse self.embed.weight in forward.

    def forward(self, inputs: Tensor) -> Tensor:
        x = self.norm1(self.embed(inputs))
        for block in self.blocks:
            x = block(x)
        x = self.norm2(x)
        # Tied head via F.linear with embedding weight (bf16).
        logits = F.linear(x, self.embed.weight.type_as(x)).float()
        logits = 15 * logits * (logits.square() + 15**2).rsqrt()
        return logits


# ---------------------------------------------------------------------------
# Muon optimizer (unchanged from v2)
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


# ---------------------------------------------------------------------------
# Init
# ---------------------------------------------------------------------------

def _init_modded(model: HyenaLM) -> None:
    for name, p in model.named_parameters():
        w = p.data
        if name.endswith("weight"):
            is_resid_out = (
                name.endswith("mlp.proj.weight")
                or name.endswith("mixer.out_proj.weight")
            )
            if is_resid_out:
                w.zero_()
            elif "embed" in name:
                w.normal_(std=0.02)
            else:
                w.normal_(std=0.33**0.5 / w.size(-1) ** 0.5)
        elif name.endswith("bias"):
            w.zero_()
        elif name.endswith("gains"):
            w.normal_(mean=1, std=0)
        else:
            raise RuntimeError(f"Uninitialized parameter: {name}")


# ---------------------------------------------------------------------------
# BPE encoding (tiktoken, GPT-2 vocab — deterministic merge tables)
# ---------------------------------------------------------------------------

def _get_bpe():
    """Load GPT-2 BPE encoder via tiktoken. Cached at module level."""
    import tiktoken
    enc = tiktoken.get_encoding("gpt2")
    return enc


def _build_token_bytes(enc) -> list[bytes]:
    """Return list mapping token_id -> raw UTF-8 bytes of that token.

    tiktoken's `decode_single_token_bytes(t)` returns the byte sequence
    that token id `t` decodes to. We materialise the full mapping once.
    """
    n = enc.n_vocab
    out: list[bytes] = []
    for t in range(n):
        try:
            out.append(enc.decode_single_token_bytes(t))
        except KeyError:
            # gpt2 vocab is dense over [0, n_vocab), but be safe.
            out.append(b"")
    return out


def _build_prefix_index(token_bytes: list[bytes]) -> dict[bytes, list[int]]:
    """Map every byte prefix → list of token IDs starting with it.

    50K tokens * avg ~4 bytes * (len+1) prefix-positions ≈ 250 K entries.
    Memory ~10-20 MB; one-time build ~50 ms.
    """
    index: dict[bytes, list[int]] = {}
    for tid, tb in enumerate(token_bytes):
        for L in range(len(tb) + 1):
            prefix = bytes(tb[:L])
            index.setdefault(prefix, []).append(tid)
    return index


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

class TrainConfig:
    """Hyena BPE-internal config.

    vocab is GPT-2 BPE's 50257. seq=512 tokens ≈ 2000 chars context.
    d_model dropped to 384 (vs v2's 512) to keep params ~30M with the
    large embedding matrix; tied head halves the vocab-projection cost.
    """
    def __init__(
        self,
        model_dim=384,
        num_layers=6,
        hyena_order=2,
        max_len=512,
        batch_size=24,
        n_steps=2000,
        bpe_train_chars=120_000_000,  # ~30 M BPE tokens
        cooldown_frac=0.7,
        embed_lr=0.05,
        scalar_lr=0.01,
        muon_lr=0.030,
        muon_wd=0.025,
        filter_lr=5e-4,
        log_every=50,
        time_budget_s=270.0,
    ):
        self.model_dim = model_dim
        self.num_layers = num_layers
        self.hyena_order = hyena_order
        self.max_len = max_len
        self.batch_size = batch_size
        self.n_steps = n_steps
        self.bpe_train_chars = bpe_train_chars
        self.cooldown_frac = cooldown_frac
        self.embed_lr = embed_lr
        self.scalar_lr = scalar_lr
        self.muon_lr = muon_lr
        self.muon_wd = muon_wd
        self.filter_lr = filter_lr
        self.log_every = log_every
        self.time_budget_s = time_budget_s

    def __repr__(self):
        return (f"TrainConfig(d={self.model_dim} L={self.num_layers} "
                f"order={self.hyena_order} bs={self.batch_size} "
                f"T={self.max_len} steps={self.n_steps} "
                f"bpe_chars={self.bpe_train_chars})")


def _train_hyena_bpe(
    text: str,
    cfg: TrainConfig,
    device: torch.device,
) -> tuple[HyenaLM, object, list[bytes], dict[bytes, list[int]]]:
    """Return (trained model, bpe encoder, token_bytes, prefix_index).

    The encoder + token_bytes + prefix_index are returned so the
    CharModel wrapper doesn't have to rebuild them post-train.
    """
    t_start = time.monotonic()

    # 1) BPE encoder + supporting structures.
    print("[hyena] loading GPT-2 BPE encoder ...", flush=True)
    enc = _get_bpe()
    print(f"[hyena] vocab size: {enc.n_vocab}", flush=True)
    print("[hyena] building token-bytes table ...", flush=True)
    token_bytes = _build_token_bytes(enc)
    print("[hyena] building byte-prefix index ...", flush=True)
    prefix_index = _build_prefix_index(token_bytes)
    print(f"[hyena] prefix index: {len(prefix_index):,} entries  "
          f"(setup elapsed {time.monotonic() - t_start:.1f}s)", flush=True)

    # 2) Pre-encode train text (cap to keep BPE pass under ~30 s).
    n_chars = min(cfg.bpe_train_chars, len(text))
    print(f"[hyena] BPE-encoding first {n_chars:,} chars ...", flush=True)
    t_enc = time.monotonic()
    train_tokens = enc.encode_ordinary(text[:n_chars])
    enc_secs = time.monotonic() - t_enc
    print(f"[hyena] encoded {len(train_tokens):,} tokens in {enc_secs:.1f}s "
          f"({len(train_tokens)/max(1e-6,enc_secs)/1e3:.0f}K tok/s)", flush=True)
    train_tok_t = torch.tensor(train_tokens, dtype=torch.long, device=device)
    n_train = train_tok_t.numel()
    if n_train < cfg.max_len + 1:
        raise ValueError(f"need at least {cfg.max_len+1} tokens; got {n_train}")

    # 3) Build model.
    vocab = enc.n_vocab
    model = HyenaLM(
        vocab_size=vocab,
        num_layers=cfg.num_layers,
        model_dim=cfg.model_dim,
        max_len=cfg.max_len,
        hyena_order=cfg.hyena_order,
    ).to(device)
    _init_modded(model)

    # 4) Optimizer routing — filter_mlp out of Muon (v2 fix).
    embed_params = [model.embed.weight]
    filter_2d = [
        p for n, p in model.blocks.named_parameters()
        if p.ndim >= 2 and "filter_mlp" in n
    ]
    block_2d = [
        p for n, p in model.blocks.named_parameters()
        if p.ndim >= 2 and "filter_mlp" not in n
    ]
    scalars = [p for p in model.parameters() if p.ndim < 2]

    optimizer1 = AdamW(
        [
            dict(params=embed_params, lr=cfg.embed_lr),
            dict(params=scalars, lr=cfg.scalar_lr),
            dict(params=filter_2d, lr=cfg.filter_lr),
        ],
        betas=(0.8, 0.95),
        eps=1e-10,
        weight_decay=0.0,
        fused=(device.type == "cuda"),
    )
    optimizer2 = Muon(block_2d, lr=cfg.muon_lr, weight_decay=cfg.muon_wd)
    optimizers = [optimizer1, optimizer2]
    for opt in optimizers:
        for g in opt.param_groups:
            g["initial_lr"] = g["lr"]

    n_params = sum(p.numel() for p in model.parameters())
    print(f"[hyena] {n_params/1e6:.2f}M params  cfg={cfg}", flush=True)

    def set_lr(step: int) -> None:
        progress = step / max(1, cfg.n_steps)
        if progress < 1 - cfg.cooldown_frac:
            eta = 1.0
        else:
            eta = max(0.0, (1 - progress) / cfg.cooldown_frac)
        for opt in optimizers:
            for g in opt.param_groups:
                g["lr"] = g["initial_lr"] * eta

    # 5) Train loop with time-budget guard (early-stop if we get close to cap).
    model.train()
    use_amp = device.type == "cuda"
    t0 = time.monotonic()
    setup_elapsed = t0 - t_start
    print(f"[hyena] setup phase used {setup_elapsed:.1f}s; "
          f"training budget ~{cfg.time_budget_s - setup_elapsed:.0f}s", flush=True)

    for step in range(cfg.n_steps):
        if cfg.time_budget_s is not None and (time.monotonic() - t_start) > cfg.time_budget_s:
            print(f"[hyena] hit time budget at step {step}; stopping early", flush=True)
            break

        set_lr(step)
        idx = torch.randint(0, n_train - cfg.max_len - 1, (cfg.batch_size,), device=device)
        offsets = idx[:, None] + torch.arange(cfg.max_len + 1, device=device)[None, :]
        flat = train_tok_t[offsets]
        x = flat[:, :-1]
        y = flat[:, 1:]

        for opt in optimizers:
            opt.zero_grad(set_to_none=True)

        if use_amp:
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                logits = model(x)
                loss = F.cross_entropy(
                    logits.reshape(-1, vocab),
                    y.reshape(-1),
                )
        else:
            logits = model(x)
            loss = F.cross_entropy(
                logits.reshape(-1, vocab),
                y.reshape(-1),
            )
        loss.backward()
        for opt in optimizers:
            opt.step()

        if cfg.log_every and (step % cfg.log_every == 0 or step == cfg.n_steps - 1):
            elapsed = time.monotonic() - t0
            print(
                f"[hyena] step {step:5d}/{cfg.n_steps}  "
                f"loss {loss.item():.4f}  elapsed {elapsed:.0f}s",
                flush=True,
            )

    return model, enc, token_bytes, prefix_index


# ---------------------------------------------------------------------------
# Streaming CharModel — BPE-marginal next-char distribution
# ---------------------------------------------------------------------------

class HyenaBPECharModel(CharModel):
    """Char-level streaming wrapper over a BPE-internal Hyena model.

    Maintains the canonical BPE re-tokenization of the byte buffer at
    every observe(). The LAST token of that tokenisation is treated as
    "active partial" — its bytes may still extend before crystallising
    into a longer token. Everything before it is "committed_context",
    fed to Hyena to obtain a distribution over the next BPE token.

    predict() then marginalises over candidate tokens (those whose
    bytes start with active_partial) to obtain P(next byte | observed).

    Caching: Hyena is re-run only when committed_context changes. Most
    observe() calls just extend the active partial and reuse the cached
    next-token distribution.
    """

    def __init__(
        self,
        model: HyenaLM,
        enc,
        token_bytes: list[bytes],
        prefix_index: dict[bytes, list[int]],
        device: torch.device | None = None,
    ):
        self.model = model
        self.enc = enc
        self.token_bytes = token_bytes
        self.prefix_index = prefix_index
        self.device = device or next(model.parameters()).device
        self.model.eval()
        self._token_byte_lens = [len(tb) for tb in token_bytes]
        # State (initialised in reset()).
        self._byte_buf: bytearray = bytearray()
        self._cached_committed: tuple[int, ...] | None = None
        self._cached_next_logp: Tensor | None = None  # log-probs over vocab

    @torch.no_grad()
    def reset(self) -> None:
        self._byte_buf = bytearray()
        self._cached_committed = None
        self._cached_next_logp = None

    # Bound the per-call tiktoken work and the model's effective context.
    # Model max_len is 512 BPE tokens ≈ 2000 bytes; we re-encode the
    # most recent ~3000 bytes (with a small overlap to be safe), and
    # the model sees the LAST max_len tokens of that. Bytes older than
    # this slip out of the streaming context — same effective
    # truncation as the byte-level submissions.
    _RETOKENISE_TAIL_BYTES: int = 3000

    @torch.no_grad()
    def _retokenise(self) -> tuple[list[int], bytes]:
        """Re-encode the recent tail of the byte buffer; return
        (committed_tokens, partial_bytes).

        partial_bytes = bytes of the LAST token of the canonical
        tokenisation of the tail. That token is treated as "still in
        flight" — the next byte might extend it into a longer token.
        Everything before it (within the tail window) is committed.
        """
        if not self._byte_buf:
            return [], b""
        tail = self._byte_buf[-self._RETOKENISE_TAIL_BYTES :]
        try:
            text = bytes(tail).decode("utf-8", errors="replace")
        except Exception:
            text = ""
        tokens = self.enc.encode_ordinary(text)
        if not tokens:
            return [], b""
        last = tokens[-1]
        last_bytes = self.token_bytes[last]
        return tokens[:-1], bytes(last_bytes)

    @torch.no_grad()
    def _hyena_next_logits(self, committed: list[int]) -> Tensor:
        """Forward Hyena over committed_tokens, return logits at the
        last position (i.e., P over next BPE token).

        If committed is empty we use a BOS-style single zero token to
        get the model's prior on the very first token.
        """
        if not committed:
            ids = [0]
        else:
            # Cap to max_len; Hyena's filter buffer is sized for max_len.
            ids = committed[-self.model.max_len :]
        x = torch.tensor(ids, dtype=torch.long, device=self.device).unsqueeze(0)
        with torch.amp.autocast(
            self.device.type, dtype=torch.bfloat16,
            enabled=(self.device.type == "cuda"),
        ):
            logits = self.model(x)
        return logits[0, -1].float()

    @torch.no_grad()
    def _ensure_distribution(self, committed: list[int]) -> Tensor:
        """Return P(next BPE token | committed) on CPU (vocab-sized
        Tensor), using the cache when committed_context is unchanged.

        Cache key uses only the slice the model actually sees
        (last max_len tokens) — re-tokenisations beyond the model's
        effective context don't invalidate the cache.
        """
        effective = committed[-self.model.max_len :]
        key = tuple(effective)
        if key == self._cached_committed and self._cached_next_logp is not None:
            return self._cached_next_logp
        logits = self._hyena_next_logits(effective)
        # Normalise in fp64 on CPU to keep mass conservation clean
        # during downstream byte marginalisation.
        p = F.softmax(logits.float(), dim=-1).double().cpu()
        self._cached_committed = key
        self._cached_next_logp = p
        return p

    @torch.no_grad()
    def predict(self) -> dict[str, float]:
        committed, partial = self._retokenise()
        p = self._ensure_distribution(committed)  # CPU fp64, shape (V,)
        partial_len = len(partial)

        # Accumulate per-byte mass in a single consistent unit:
        # probability under the model that the very next byte is c.
        # Two mutually-exclusive branches:
        #
        #   (A) Extension: the next BPE token starts with `partial` and
        #       has length > partial_len. Then byte c = bytes(t)[partial_len].
        #       Mass: p[t].
        #
        #   (B) Boundary: the next BPE token is exactly `partial` (so
        #       partial commits as a complete token), AND the token AFTER
        #       that begins with c. Mass: p[partial_token] * P(next-token
        #       starts with c | context after committing partial).
        #       We approximate the after-context distribution with p
        #       (cheap; one Hyena fwd per predict instead of two).
        #
        # Both branches output mass in the same units; the final
        # renormalise distributes the *remainder* across bytes the
        # candidate set never proposed.
        byte_mass = [0.0] * 256

        # (A) Extension branch.
        cands = self.prefix_index.get(partial, [])
        boundary_token_id: int | None = None
        if cands:
            cand_t = torch.tensor(cands, dtype=torch.long)
            cand_p = p[cand_t].tolist()
            for tid, pt in zip(cands, cand_p):
                tb = self.token_bytes[tid]
                if len(tb) > partial_len:
                    byte_mass[tb[partial_len]] += pt
                elif len(tb) == partial_len:
                    # Exactly equal to partial — boundary candidate.
                    boundary_token_id = tid

        # (B) Boundary branch (only if partial itself is a token).
        if boundary_token_id is not None and partial_len > 0:
            boundary_p = float(p[boundary_token_id])
            if boundary_p > 0:
                # P(first byte of new token = c | context) ≈
                # sum over t' starting with c of p[t'].
                for c in range(256):
                    tids_c = self.prefix_index.get(bytes([c]), [])
                    if not tids_c:
                        continue
                    tids_t = torch.tensor(tids_c, dtype=torch.long)
                    p_first = float(p[tids_t].sum())
                    byte_mass[c] += boundary_p * p_first

        # Renormalise. If no candidates produced mass (very early in the
        # stream / unseen partial), fall back to uniform over ASCII.
        # Match modded_nanogpt's convention: only emit single-byte UTF-8
        # keys (bytes 0x00-0x7F). Multi-byte chars (rare in WikiText)
        # are always wrong under this scheme but consistent across
        # submissions, so the comparison is apples-to-apples.
        total = sum(byte_mass)
        if total <= 0:
            return {chr(b): 1.0 / 128 for b in range(128)}
        inv = 1.0 / total
        out: dict[str, float] = {}
        for b in range(256):
            m = byte_mass[b]
            if m <= 0:
                continue
            try:
                ch = bytes([b]).decode("utf-8")
            except UnicodeDecodeError:
                continue
            out[ch] = m * inv
        return out

    @torch.no_grad()
    def observe(self, char: str) -> None:
        # Append the UTF-8 bytes of the observed char to the buffer.
        # The next predict() will re-tokenise from scratch (cheap with
        # tiktoken — Rust under the hood) and use the cache if
        # committed_context is unchanged.
        for b in char.encode("utf-8"):
            self._byte_buf.append(b)


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
        print(f"[hyena] SEED={seed}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = TrainConfig()
    model, enc, token_bytes, prefix_index = _train_hyena_bpe(train_text, cfg, device)
    return HyenaBPECharModel(model, enc, token_bytes, prefix_index, device=device)
