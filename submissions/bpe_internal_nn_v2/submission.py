"""bpe_internal_nn_v2 — Internal BPE NN with multiprocess encode + step cap.

Fixes for v1 DQ (300s timeout, hit step ~1200/1500):
  * MULTIPROCESS encode: split train_text into N_PROC chunks at safe
    UTF-8 boundaries, encode_ordinary each in a multiprocessing.Pool
    (fork start method). v1 took 74s single-threaded; expected ~10-15s
    with 8 workers.
  * STEP CAP at 1000: v1's loss had largely converged by step 1000
    (4.40 → 4.25 from step 1000 → 1200 in v1).
  * smaller `max_len`=384 (vs 512): less compute per step, marginal acc
    loss expected.

Rest of the pipeline (transformer arch, marginalization, KV cache,
re-tokenize-tail at observe) is unchanged from v1, which subagent_2
validated end-to-end (loss 4.25 at step 1200 = 1.34 bpc, well above
the floor needed for char-acc 0.70).

Expected: 15-25 kJ / 0.71-0.74 acc. First clean run of the BPE paradigm.
"""
from __future__ import annotations

__author__ = "@subagent-xorfix-2026-05-19"

import concurrent.futures
import os
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.optim import AdamW

from wikitext import CharModel


# ===========================================================================
# Constants
# ===========================================================================

GPT2_VOCAB = 50_257
MAX_TOKEN_BYTES = 128
RETOKENIZE_TAIL = 256
SMOKE_TRAIN_BYTES = 50_000

# Number of parallel threads for tiktoken encode. tiktoken is a Rust
# library that releases the GIL during encode_ordinary, so threads
# parallelize effectively. Modal A100 host has ~8-12 vCPUs.
N_ENCODE_THREADS = 8


# ===========================================================================
# Architecture (identical to v1)
# ===========================================================================


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
        angular_freq = (1 / 1024) ** torch.linspace(
            0, 1, steps=dim // 4, dtype=torch.float32
        )
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

    def forward(self, x, kv_cache=None, offset=0):
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

    def forward(self, x):
        x = self.fc(x)
        x = x.relu().square()
        return self.proj(x)


class Block(nn.Module):
    def __init__(self, dim: int, head_dim: int):
        super().__init__()
        self.attn = CausalSelfAttention(dim, head_dim=head_dim)
        self.mlp = MLP(dim)
        self.norm1 = RMSNorm(dim)
        self.norm2 = RMSNorm(dim)

    def forward(self, x, kv_cache=None, offset=0):
        h, new_kv = self.attn(self.norm1(x), kv_cache, offset=offset)
        x = x + h
        x = x + self.mlp(self.norm2(x))
        return x, new_kv


class GPT(nn.Module):
    def __init__(self, vocab_size: int, num_layers: int, model_dim: int,
                 head_dim: int = 64, max_len: int = 1024):
        super().__init__()
        self.vocab_size = vocab_size
        self.max_len = max_len
        self.embed = nn.Embedding(vocab_size, model_dim).bfloat16()
        self.blocks = nn.ModuleList(
            [Block(model_dim, head_dim=head_dim) for _ in range(num_layers)]
        )
        self.proj = Linear(model_dim, vocab_size)
        self.norm1 = RMSNorm(model_dim)
        self.norm2 = RMSNorm(model_dim)

    def forward(self, inputs, kv_caches=None, offset=0):
        x = self.norm1(self.embed(inputs))
        new_caches = []
        for i, block in enumerate(self.blocks):
            kv = kv_caches[i] if kv_caches is not None else None
            x, new_kv = block(x, kv, offset=offset)
            new_caches.append(new_kv)
        logits = self.proj(self.norm2(x)).float()
        logits = 15 * logits * (logits.square() + 15**2).rsqrt()
        return logits, new_caches


# ===========================================================================
# Muon
# ===========================================================================


def zeropower_via_newtonschulz5(G):
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


def muon_update(grad, momentum, mu=0.95, nesterov=True):
    momentum.lerp_(grad, 1 - mu)
    update = grad.lerp_(momentum, mu) if nesterov else momentum
    update = zeropower_via_newtonschulz5(update)
    update *= max(1, grad.size(-2) / grad.size(-1)) ** 0.5
    return update


class Muon(torch.optim.Optimizer):
    def __init__(self, params, lr=0.02, weight_decay=0.0, mu=0.95):
        params = list(params)
        defaults = dict(lr=lr, weight_decay=weight_decay, mu=mu)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self):
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


def _init_modded(model):
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


# ===========================================================================
# Training loop
# ===========================================================================


class TrainConfig:
    def __init__(
        self,
        model_dim=256,
        num_layers=4,
        head_dim=64,
        max_len=384,
        batch_size=32,
        n_steps=1000,
        cooldown_frac=0.7,
        embed_lr=0.3,
        head_lr=1.0 / 320,
        scalar_lr=0.01,
        muon_lr=0.035,
        muon_wd=0.025,
        log_every=100,
    ):
        self.model_dim = model_dim
        self.num_layers = num_layers
        self.head_dim = head_dim
        self.max_len = max_len
        self.batch_size = batch_size
        self.n_steps = n_steps
        self.cooldown_frac = cooldown_frac
        self.embed_lr = embed_lr
        self.head_lr = head_lr
        self.scalar_lr = scalar_lr
        self.muon_lr = muon_lr
        self.muon_wd = muon_wd
        self.log_every = log_every

    def __repr__(self):
        return (f"TrainConfig(d={self.model_dim} L={self.num_layers} "
                f"H={self.model_dim//self.head_dim} bs={self.batch_size} "
                f"T={self.max_len} steps={self.n_steps})")


def _train_bpe(
    token_ids_gpu: Tensor, vocab_size: int, cfg: TrainConfig,
    device: torch.device,
) -> GPT:
    n = token_ids_gpu.numel()
    if n < cfg.max_len + 1:
        raise ValueError(f"need at least {cfg.max_len+1} tokens; got {n}")
    model = GPT(
        vocab_size=vocab_size,
        num_layers=cfg.num_layers,
        model_dim=cfg.model_dim,
        head_dim=cfg.head_dim,
        max_len=cfg.max_len,
    ).to(device)
    _init_modded(model)
    block_2d = [p for p in model.blocks.parameters() if p.ndim >= 2]
    scalars = [p for p in model.parameters() if p.ndim < 2]
    optimizer1 = AdamW(
        [
            dict(params=[model.embed.weight], lr=cfg.embed_lr),
            dict(params=[model.proj.weight], lr=cfg.head_lr),
            dict(params=scalars, lr=cfg.scalar_lr),
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
    print(f"[bpe_nn] NN {n_params/1e6:.2f}M params  cfg={cfg}", flush=True)

    def set_lr(step: int) -> None:
        progress = step / cfg.n_steps
        if progress < 1 - cfg.cooldown_frac:
            eta = 1.0
        else:
            eta = max(0.0, (1 - progress) / cfg.cooldown_frac)
        for opt in optimizers:
            for g in opt.param_groups:
                g["lr"] = g["initial_lr"] * eta

    model.train()
    use_amp = device.type == "cuda"
    t0 = time.monotonic()
    for step in range(cfg.n_steps):
        set_lr(step)
        idx = torch.randint(0, n - cfg.max_len - 1, (cfg.batch_size,), device=device)
        offsets = idx[:, None] + torch.arange(cfg.max_len + 1, device=device)[None, :]
        flat = token_ids_gpu[offsets].long()
        x = flat[:, :-1]
        y = flat[:, 1:]
        for opt in optimizers:
            opt.zero_grad(set_to_none=True)
        if use_amp:
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                logits, _ = model(x)
                loss = F.cross_entropy(logits.reshape(-1, vocab_size), y.reshape(-1))
        else:
            logits, _ = model(x)
            loss = F.cross_entropy(logits.reshape(-1, vocab_size), y.reshape(-1))
        loss.backward()
        for opt in optimizers:
            opt.step()
        if cfg.log_every and (step % cfg.log_every == 0 or step == cfg.n_steps - 1):
            elapsed = time.monotonic() - t0
            print(
                f"[bpe_nn] step {step:5d}/{cfg.n_steps}  "
                f"loss {loss.item():.4f}  elapsed {elapsed:.0f}s",
                flush=True,
            )
    return model


# ===========================================================================
# Multiprocess tiktoken encode
# ===========================================================================


def _split_at_safe_boundaries(s: str, n_chunks: int) -> list[str]:
    """Split string into ~equal chunks at whitespace boundaries to keep
    BPE merge behavior consistent. (BPE is anchored at whitespace in
    GPT-2's pre-tokenizer, so splitting at space/newline avoids drift.)
    """
    if n_chunks <= 1 or len(s) < 1024 * n_chunks:
        return [s]
    target = len(s) // n_chunks
    chunks: list[str] = []
    start = 0
    for i in range(n_chunks - 1):
        cut = start + target
        # Find next whitespace at-or-after cut.
        while cut < len(s) and not s[cut].isspace():
            cut += 1
        if cut >= len(s):
            break
        chunks.append(s[start:cut])
        start = cut
    chunks.append(s[start:])
    return [c for c in chunks if c]


def _parallel_encode(train_str: str, encoding, n_threads: int) -> list:
    """Encode train_str with tiktoken GPT-2 across n_threads workers.

    tiktoken's encode_ordinary is implemented in Rust and releases the
    Python GIL, so a ThreadPoolExecutor gives true parallelism without
    the picklability constraints of multiprocessing.

    Splits at whitespace boundaries so BPE merges line up identically
    with single-process encode (GPT-2 pre-tokenizer is whitespace-aware).
    """
    chunks = _split_at_safe_boundaries(train_str, n_threads)
    if len(chunks) == 1:
        return encoding.encode_ordinary(chunks[0])
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(chunks)) as ex:
        results = list(ex.map(encoding.encode_ordinary, chunks))
    out: list = []
    for r in results:
        out.extend(r)
    return out


# ===========================================================================
# Token-bytes table
# ===========================================================================


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


# ===========================================================================
# CharModel — verbatim from v1
# ===========================================================================


class BPECharModel(CharModel):
    def __init__(
        self,
        model: GPT,
        encoding,
        token_bytes_arr: np.ndarray,
        token_lens: np.ndarray,
        device: torch.device,
    ):
        self.model = model
        self.encoding = encoding
        self.token_bytes_arr = token_bytes_arr
        self.token_lens = token_lens
        self.device = device
        self.model.eval()
        self._kv: list[tuple[Tensor, Tensor]] | None = None
        self._next_logits: Tensor | None = None
        self._pos: int = 0
        self._committed_byte_count: int = 0
        self._history: bytearray = bytearray()
        self._bos_id: int = 50_256

    @torch.no_grad()
    def reset(self) -> None:
        self._kv = None
        self._pos = 0
        self._committed_byte_count = 0
        self._history = bytearray()
        x = torch.tensor([[self._bos_id]], dtype=torch.long, device=self.device)
        logits, self._kv = self.model(x, None, offset=self._pos)
        self._next_logits = logits[0, -1]
        self._pos = 1

    def _pending_buffer(self) -> bytes:
        if self._committed_byte_count >= len(self._history):
            return b""
        return bytes(self._history[self._committed_byte_count:])

    @torch.no_grad()
    def predict(self) -> dict[str, float]:
        if self._next_logits is None:
            raise RuntimeError("predict() called before reset()")
        p_token = F.softmax(self._next_logits.float(), dim=-1).cpu().numpy()
        pending = self._pending_buffer()
        plen = len(pending)
        if plen == 0:
            active_mask = self.token_lens >= 1
        else:
            pending_arr = np.frombuffer(pending, dtype=np.uint8)
            cmp = self.token_bytes_arr[:, :plen] == pending_arr[None, :]
            prefix_match = cmp.all(axis=1)
            active_mask = prefix_match & (self.token_lens > plen)
        active_ids = np.flatnonzero(active_mask)
        if active_ids.size == 0:
            p = 1.0 / 95.0
            return {chr(c): p for c in range(32, 127)}
        active_next_bytes = self.token_bytes_arr[active_ids, plen]
        active_probs = p_token[active_ids]
        mass = np.bincount(
            active_next_bytes.astype(np.int64),
            weights=active_probs.astype(np.float64),
            minlength=256,
        )
        total = mass.sum()
        if total <= 0.0:
            p = 1.0 / 95.0
            return {chr(c): p for c in range(32, 127)}
        mass = mass / total
        out: dict[str, float] = {}
        for byte_id in range(256):
            if mass[byte_id] <= 0.0:
                continue
            try:
                ch = bytes([byte_id]).decode("utf-8")
            except UnicodeDecodeError:
                continue
            out[ch] = float(mass[byte_id])
        return out

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
        new_tokens = token_ids[:-1]
        n_new = len(new_tokens)
        consumed_bytes_len = sum(
            len(self.encoding.decode_single_token_bytes(t))
            for t in new_tokens
        )
        x = torch.tensor([new_tokens], dtype=torch.long, device=self.device)
        logits, self._kv = self.model(x, self._kv, offset=self._pos)
        self._next_logits = logits[0, -1]
        self._pos += n_new
        self._committed_byte_count += consumed_bytes_len
        self._maybe_trim_cache()

    def _maybe_trim_cache(self) -> None:
        if self._kv is None:
            return
        cur = self._kv[0][0].shape[2]
        if cur < self.model.max_len:
            return
        keep = self.model.max_len - 1
        self._kv = [(k[:, :, -keep:], v[:, :, -keep:]) for k, v in self._kv]


class _EmptyCharModel(CharModel):
    def reset(self) -> None: pass
    def predict(self) -> dict[str, float]:
        p = 1.0 / 95.0
        return {chr(c): p for c in range(32, 127)}
    def observe(self, char: str) -> None: pass


# ===========================================================================
# Entry point
# ===========================================================================


def train(train_text: str, valid_text: str | None = None) -> CharModel:
    if os.environ.get("SMOKE_TEST_ONLY") == "1":
        return _EmptyCharModel()

    seed_env = os.environ.get("SEED")
    if seed_env:
        seed = int(seed_env)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        print(f"[bpe_nn] SEED={seed}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    raw = train_text.encode("utf-8")
    is_smoke = len(raw) < SMOKE_TRAIN_BYTES

    import tiktoken
    encoding = tiktoken.get_encoding("gpt2")
    V = encoding.n_vocab
    assert V == GPT2_VOCAB, f"expected vocab {GPT2_VOCAB}, got {V}"

    print(f"[bpe_nn] device={device} is_smoke={is_smoke} train_bytes={len(raw):,}",
          flush=True)

    t0 = time.monotonic()
    token_bytes_arr, token_lens = _build_token_bytes_table(encoding)
    print(f"[bpe_nn] built token_bytes table ({V} tokens)  "
          f"{time.monotonic()-t0:.1f}s", flush=True)

    # Encode train_text into BPE tokens (multiprocess).
    t0 = time.monotonic()
    train_str = raw.decode("utf-8", errors="replace")
    if is_smoke:
        # Skip threadpool overhead on tiny corpus.
        train_token_ids = encoding.encode_ordinary(train_str)
    else:
        train_token_ids = _parallel_encode(train_str, encoding, N_ENCODE_THREADS)
    n_tokens = len(train_token_ids)
    print(f"[bpe_nn] encoded train (threads={N_ENCODE_THREADS}): "
          f"{n_tokens:,} tokens "
          f"({len(raw)/max(1,n_tokens):.2f} bytes/token)  "
          f"{time.monotonic()-t0:.1f}s", flush=True)

    token_ids_gpu = torch.tensor(train_token_ids, dtype=torch.int32, device=device)
    del train_token_ids, train_str

    if is_smoke:
        cfg = TrainConfig(
            model_dim=64, num_layers=2, head_dim=32,
            max_len=min(64, max(8, n_tokens // 4)),
            batch_size=2, n_steps=4, log_every=0,
        )
    else:
        cfg = TrainConfig()

    model = _train_bpe(token_ids_gpu, V, cfg, device)

    return BPECharModel(
        model=model,
        encoding=encoding,
        token_bytes_arr=token_bytes_arr,
        token_lens=token_lens,
        device=device,
    )
