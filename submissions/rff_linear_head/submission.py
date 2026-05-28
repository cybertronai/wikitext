"""RFF linear-head char-LM (no attention).

Per experiments/kernel_methods/experiment_02_rff_linear_head_charlm.md.

Architecture (sub-trivial char-LM, attention-free):

    byte ids  (B, W)
      -> embed(256, 128)                  learned
      -> causal 1D conv (width 8)          learned
      -> RFF projection (d=128 -> k=4096)  FROZEN  (Rahimi/Recht 2007)
      -> linear head (k=4096 -> 256)       learned

RFF:  z = sqrt(2/k) * cos(W @ x + b),  W ~ N(0, sigma^-2 I),  b ~ U(0, 2pi)
sigma chosen by median-heuristic on 4K init activations (fallback sqrt(d)).

Training: AdamW lr=3e-3, wd=0, time-based loop targeting ~250s wall.
CharModel wrapper keeps a rolling 8-byte history and recomputes the
forward on the last W bytes at each predict() call.
"""
from __future__ import annotations

__author__ = "@ab-10"

import math
import os
import time
from collections import deque

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.optim import AdamW

from wikitext import CharModel


# ---------------------------------------------------------------------------
# Hyperparameters
# ---------------------------------------------------------------------------

VOCAB = 256
D_EMBED = 128
W_CONV = 8          # causal context width fed to the conv
K_RFF = 4096
BATCH = 64
SEQ_LEN = 512       # training-time context length
LR = 3e-3
WD = 0.0
TRAIN_BUDGET_S = 250.0


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class RFF(nn.Module):
    """Frozen Random Fourier Features for a Gaussian (RBF) kernel."""

    def __init__(self, d: int, k: int, sigma: float):
        super().__init__()
        self.d = d
        self.k = k
        self.sigma = float(sigma)
        # W ~ N(0, sigma^-2 I): equivalent to drawing N(0, I) and dividing by sigma.
        self.register_buffer("W", torch.randn(d, k) / self.sigma)
        self.register_buffer("b", torch.rand(k) * (2.0 * math.pi))

    def forward(self, x: Tensor) -> Tensor:
        # x: (..., d) -> (..., k)
        return math.sqrt(2.0 / self.k) * torch.cos(x @ self.W + self.b)


class RFFCharLM(nn.Module):
    """Embed -> causal 1D conv (width W) -> frozen RFF -> linear head."""

    def __init__(
        self,
        vocab: int = VOCAB,
        d: int = D_EMBED,
        w: int = W_CONV,
        k: int = K_RFF,
        sigma: float | None = None,
    ):
        super().__init__()
        self.vocab = vocab
        self.d = d
        self.w = w
        self.k = k

        self.embed = nn.Embedding(vocab, d)
        # Causal: we left-pad with (w-1) so that output at t depends on
        # input bytes [t-w+1 ... t]. Conv weight shape: (d_out, d_in, w).
        self.conv = nn.Conv1d(d, d, kernel_size=w, bias=True)
        # RFF needs a sigma; we set it after computing the median heuristic.
        # Use the cheap default until calibrate_sigma() overwrites it.
        if sigma is None:
            sigma = math.sqrt(d)
        self.rff = RFF(d, k, sigma)
        self.head = nn.Linear(k, vocab, bias=True)

        # Init
        nn.init.normal_(self.embed.weight, mean=0.0, std=1.0)
        nn.init.kaiming_uniform_(self.conv.weight, a=math.sqrt(5))
        nn.init.zeros_(self.conv.bias)
        nn.init.normal_(self.head.weight, mean=0.0, std=0.01)
        nn.init.zeros_(self.head.bias)

    # ------------------------------------------------------------------
    # Forward helpers
    # ------------------------------------------------------------------

    def _conv_features(self, x: Tensor) -> Tensor:
        """x: (B, T) byte ids -> (B, T, d) causal-conv features.

        We left-pad with (w-1) zeros so each output position t sees
        exactly bytes [t-w+1 .. t] of the input (causal).
        """
        e = self.embed(x)                              # (B, T, d)
        h = e.transpose(1, 2)                          # (B, d, T)
        h = F.pad(h, (self.w - 1, 0))                  # left-pad time dim
        h = self.conv(h)                               # (B, d, T)
        return h.transpose(1, 2)                       # (B, T, d)

    def forward(self, x: Tensor) -> Tensor:
        c = self._conv_features(x)                     # (B, T, d)
        z = self.rff(c)                                # (B, T, k)
        logits = self.head(z)                          # (B, T, vocab)
        return logits

    def forward_last(self, x: Tensor) -> Tensor:
        """Predict only the next byte given a single history slice.

        x: (1, T) where T <= self.w (we use the last W bytes).
        Returns logits of shape (vocab,).
        """
        e = self.embed(x)                              # (1, T, d)
        h = e.transpose(1, 2)                          # (1, d, T)
        T = h.size(-1)
        if T < self.w:
            h = F.pad(h, (self.w - T, 0))              # left-pad to width
        # Now T == w; conv yields (1, d, 1).
        h = self.conv(h)
        c = h.transpose(1, 2)                          # (1, 1, d)
        z = self.rff(c)                                # (1, 1, k)
        logits = self.head(z)                          # (1, 1, vocab)
        return logits[0, 0]

    # ------------------------------------------------------------------
    # Sigma calibration (median heuristic)
    # ------------------------------------------------------------------

    @torch.no_grad()
    def calibrate_sigma(self, byte_tensor: Tensor, n_samples: int = 4096) -> float:
        """Pick sigma via median-heuristic on n_samples pre-RFF activations.

        Samples random length-w windows from byte_tensor, runs them through
        embed+conv, computes pairwise L2 distances on a subset, and sets
        sigma to the median. Falls back to sqrt(d) if anything goes wrong.
        """
        try:
            n = byte_tensor.numel()
            if n < self.w + 1:
                raise RuntimeError("dataset shorter than window")
            device = self.embed.weight.device
            # Draw n_samples random positions (each a window of size w).
            idx = torch.randint(0, n - self.w, (n_samples,), device=device)
            offs = idx[:, None] + torch.arange(self.w, device=device)[None, :]
            x = byte_tensor[offs].long()               # (n_samples, w)
            # Run through embed + conv as a batch where the "T" dim equals w.
            e = self.embed(x)                          # (n_samples, w, d)
            h = e.transpose(1, 2)                      # (n_samples, d, w)
            h = self.conv(h)                           # (n_samples, d, 1)
            c = h.squeeze(-1)                          # (n_samples, d)
            # Median pairwise distance on a 1024-sample subset (cheap, fine).
            m = min(1024, c.size(0))
            cs = c[:m].float()
            d2 = torch.cdist(cs, cs)                   # (m, m)
            mask = torch.triu(torch.ones_like(d2, dtype=torch.bool), diagonal=1)
            dists = d2[mask]
            sigma = float(dists.median().item())
            if not math.isfinite(sigma) or sigma <= 0:
                raise RuntimeError(f"non-finite/zero sigma: {sigma}")
            # Replace RFF buffers in-place (rescale W).
            self.rff.W.copy_(self.rff.W * (self.rff.sigma / sigma))
            self.rff.sigma = sigma
            return sigma
        except Exception as e:
            print(f"[rff] calibrate_sigma fell back to sqrt(d): {e}", flush=True)
            return self.rff.sigma


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def _train(text: str, device: torch.device) -> RFFCharLM:
    raw = text.encode("utf-8")
    train_bytes = torch.frombuffer(bytearray(raw), dtype=torch.uint8).to(device)
    n = train_bytes.numel()
    if n < SEQ_LEN + 1:
        raise ValueError(f"need at least {SEQ_LEN+1} bytes; got {n}")

    model = RFFCharLM().to(device)

    # Median-heuristic sigma on init activations.
    sigma_default = math.sqrt(D_EMBED)
    sigma_used = model.calibrate_sigma(train_bytes, n_samples=4096)
    print(f"[rff] sigma_default(sqrt(d))={sigma_default:.4f}  sigma_used={sigma_used:.4f}",
          flush=True)

    n_params = sum(p.numel() for p in model.parameters())
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    # RFF buffers (W, b) are not parameters (they are buffers) — confirm.
    print(f"[rff] params total={n_params/1e6:.3f}M  trainable={n_trainable/1e6:.3f}M  "
          f"(embed {model.embed.weight.numel()/1e6:.3f}M, conv {sum(p.numel() for p in model.conv.parameters())/1e6:.3f}M, "
          f"head {sum(p.numel() for p in model.head.parameters())/1e6:.3f}M)",
          flush=True)

    optim = AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=LR,
        weight_decay=WD,
        betas=(0.9, 0.95),
        eps=1e-8,
        fused=(device.type == "cuda"),
    )

    model.train()
    use_amp = device.type == "cuda"
    t0 = time.monotonic()
    step = 0
    last_loss = float("nan")
    rff_var_logged = False

    # Train until we run out of budget (well under MAX_TRAIN_SECONDS=300).
    while True:
        elapsed = time.monotonic() - t0
        if elapsed >= TRAIN_BUDGET_S:
            break

        idx = torch.randint(0, n - SEQ_LEN - 1, (BATCH,), device=device)
        offsets = idx[:, None] + torch.arange(SEQ_LEN + 1, device=device)[None, :]
        flat = train_bytes[offsets].long()
        x = flat[:, :-1]
        y = flat[:, 1:]

        optim.zero_grad(set_to_none=True)
        if use_amp:
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                logits = model(x)
                loss = F.cross_entropy(logits.reshape(-1, VOCAB), y.reshape(-1))
        else:
            logits = model(x)
            loss = F.cross_entropy(logits.reshape(-1, VOCAB), y.reshape(-1))
        loss.backward()
        optim.step()
        last_loss = float(loss.item())

        if not rff_var_logged and step == 50:
            with torch.no_grad():
                # Log RFF feature variance per-dim on a fresh batch (sanity).
                c = model._conv_features(x).float()
                z = model.rff(c).float()
                fv = z.var(dim=(0, 1)).mean().item()
                cnorm = c.norm(dim=-1).mean().item()
                print(f"[rff] step {step}  RFF feature var (mean over k)={fv:.4f}  "
                      f"(expected ~0.5 for white input)  ||c||={cnorm:.3f}",
                      flush=True)
            rff_var_logged = True

        if step % 100 == 0:
            print(f"[rff] step {step:6d}  loss {last_loss:.4f}  elapsed {elapsed:6.1f}s",
                  flush=True)
        step += 1

    elapsed = time.monotonic() - t0
    print(f"[rff] DONE  steps={step}  final_loss {last_loss:.4f}  elapsed {elapsed:.1f}s",
          flush=True)

    with torch.no_grad():
        head_norm = model.head.weight.float().norm().item()
        emb_norm = model.embed.weight.float().norm().item()
        print(f"[rff] ||W_out||_F={head_norm:.4f}  ||embed||_F={emb_norm:.4f}",
              flush=True)

    return model


# ---------------------------------------------------------------------------
# CharModel wrapper
# ---------------------------------------------------------------------------

class RFFCharModel(CharModel):
    """Streaming char model: rolls a W-byte deque, re-forwards on predict."""

    def __init__(self, model: RFFCharLM, device: torch.device | None = None):
        self.model = model
        self.device = device or next(model.parameters()).device
        self.model.eval()
        self.w = model.w
        self._buf: deque[int] = deque(maxlen=self.w)

    def reset(self) -> None:
        self._buf.clear()

    @torch.no_grad()
    def predict(self) -> dict[str, float]:
        # Use last w bytes (left-pad inside forward_last if shorter).
        if len(self._buf) == 0:
            # Use a single zero byte as a benign starting context.
            x = torch.zeros(1, 1, dtype=torch.long, device=self.device)
        else:
            x = torch.tensor([list(self._buf)], dtype=torch.long, device=self.device)
        logits = self.model.forward_last(x).float()
        probs = F.softmax(logits, dim=-1)
        out: dict[str, float] = {}
        for byte_id, p in enumerate(probs.tolist()):
            try:
                ch = bytes([byte_id]).decode("utf-8")
            except UnicodeDecodeError:
                continue
            out[ch] = p
        return out

    def observe(self, char: str) -> None:
        for byte in char.encode("utf-8"):
            self._buf.append(int(byte))


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
        print(f"[rff] SEED={seed}", flush=True)
    else:
        torch.manual_seed(0)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[rff] device={device}  d={D_EMBED}  w={W_CONV}  k={K_RFF}  "
          f"batch={BATCH}  seq={SEQ_LEN}  lr={LR}  budget_s={TRAIN_BUDGET_S}",
          flush=True)
    model = _train(train_text, device)
    return RFFCharModel(model, device)
