"""P2-A — Random Projection Floor (FF investigation diagnostic).

Identical architecture and ridge readout to pass-2's causal FF, but ALL
five layers are frozen-random Gaussian. No FF training loop, no negative
sampling. The ridge readout does all the prediction work.

If this run's val char-acc is within 0.02 of pass-2's 0.279, FF is
contributing zero representational value to that setup — a load-bearing
piece of diagnostic evidence for the FF investigation.

Spec: .survey/ff_runs/phase2/P2-A_random_projection/design.md
"""
from __future__ import annotations

__author__ = "@survey-ff-p2a"

import os
import time

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from wikitext import CharModel


VOCAB = 256
K = 24
INPUT_DIM = (K + 1) * VOCAB
N_LAYERS = 5
WIDTH = 384

RIDGE_N_FIT = 80_000
RIDGE_LAMBDA = 1.0
FEATURE_DIM = (N_LAYERS - 1) * WIDTH

EVAL_BATCH = 256
RIDGE_FORWARD_BATCH = 512


class FFLayer(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.lin = nn.Linear(in_dim, out_dim, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        return F.relu(self.lin(x))


def l2_normalize(x: Tensor, eps: float = 1e-8) -> Tensor:
    return x / (x.norm(dim=-1, keepdim=True) + eps)


class FrozenStack(nn.Module):
    """Same shape as pass-2's FFStack but every layer is frozen random."""

    def __init__(self):
        super().__init__()
        layers = [FFLayer(INPUT_DIM, WIDTH)]
        for _ in range(N_LAYERS - 1):
            layers.append(FFLayer(WIDTH, WIDTH))
        self.layers = nn.ModuleList(layers)
        for layer in self.layers:
            for p in layer.parameters():
                p.requires_grad_(False)

    @torch.no_grad()
    def features(self, x: Tensor) -> Tensor:
        h = x
        feats = []
        for li, layer in enumerate(self.layers):
            a = layer(h)
            if li >= 1:
                feats.append(l2_normalize(a))
            h = l2_normalize(a)
        return torch.cat(feats, dim=-1)


def build_input(context_bytes: Tensor, candidate_byte: Tensor | None) -> Tensor:
    B = context_bytes.size(0)
    ctx_oh = F.one_hot(context_bytes, VOCAB).float()
    if candidate_byte is None:
        cand_oh = torch.zeros(B, 1, VOCAB, device=context_bytes.device)
    else:
        cand_oh = F.one_hot(candidate_byte, VOCAB).float().unsqueeze(1)
    full = torch.cat([ctx_oh, cand_oh], dim=1)
    return full.reshape(B, INPUT_DIM)


@torch.no_grad()
def _extract_features(
    model: FrozenStack,
    train_bytes: Tensor,
    sample_idx: Tensor,
    batch_size: int,
) -> tuple[Tensor, Tensor]:
    n_fit = sample_idx.numel()
    feats = torch.empty(n_fit, FEATURE_DIM, device=train_bytes.device, dtype=torch.float32)
    targets = torch.empty(n_fit, device=train_bytes.device, dtype=torch.long)
    arange_k = torch.arange(K, device=train_bytes.device)
    model.eval()
    for start in range(0, n_fit, batch_size):
        end = min(start + batch_size, n_fit)
        idx = sample_idx[start:end]
        offsets = idx[:, None] + arange_k[None, :]
        ctx = train_bytes[offsets].long()
        tgt = train_bytes[idx + K].long()
        x = build_input(ctx, None)
        feats[start:end] = model.features(x)
        targets[start:end] = tgt
    return feats, targets


@torch.no_grad()
def _solve_ridge(phi: Tensor, targets: Tensor, lam: float) -> Tensor:
    D = phi.shape[1]
    device = phi.device
    dtype = torch.float32
    phi32 = phi.to(dtype)
    A = phi32.T @ phi32 + lam * torch.eye(D, device=device, dtype=dtype)
    B = torch.zeros(D, VOCAB, device=device, dtype=dtype)
    B.index_add_(1, targets, phi32.T)
    return torch.linalg.solve(A, B)


def _train(train_text: str, device: torch.device, seed: int) -> tuple[FrozenStack, Tensor]:
    raw = train_text.encode("utf-8")
    train_bytes = torch.frombuffer(bytearray(raw), dtype=torch.uint8).to(device)
    n = train_bytes.numel()
    if n < K + 1:
        raise ValueError(f"need at least {K+1} bytes; got {n}")

    gen = torch.Generator(device=device)
    gen.manual_seed(seed)

    model = FrozenStack().to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[p2a] {n_params/1e6:.2f}M params  layers={N_LAYERS} width={WIDTH} K={K}  "
          f"(all frozen-random; no FF training)")

    max_start = n - K - 1
    t_ridge = time.monotonic()
    sample_idx = torch.randint(0, max_start, (RIDGE_N_FIT,), device=device, generator=gen)
    phi, tgt = _extract_features(model, train_bytes, sample_idx, RIDGE_FORWARD_BATCH)
    W = _solve_ridge(phi, tgt, RIDGE_LAMBDA)
    print(f"[ridge] N_fit={RIDGE_N_FIT} D={FEATURE_DIM} lam={RIDGE_LAMBDA} "
          f"fit_s={time.monotonic()-t_ridge:.1f}", flush=True)

    with torch.no_grad():
        diag_n = min(20_000, RIDGE_N_FIT)
        pred = (phi[:diag_n] @ W).argmax(dim=-1)
        train_acc = (pred == tgt[:diag_n]).float().mean().item()
        print(f"[ridge] train_subset_acc={train_acc:.4f}  (n={diag_n})", flush=True)
    return model, W


class FrozenRidgeCharModel(CharModel):
    def __init__(self, model: FrozenStack, W: Tensor, device: torch.device):
        self.model = model
        self.W = W
        self.device = device
        self.model.eval()
        self._ctx: list[int] = []

    @torch.no_grad()
    def reset(self) -> None:
        self._ctx = []

    @torch.no_grad()
    def _build_one(self) -> Tensor:
        pad = K - len(self._ctx)
        ctx_bytes = ([0] * pad + self._ctx) if pad > 0 else self._ctx[-K:]
        ctx_t = torch.tensor(ctx_bytes, dtype=torch.long, device=self.device).unsqueeze(0)
        return build_input(ctx_t, None)

    @torch.no_grad()
    def predict(self) -> dict[str, float]:
        x = self._build_one()
        phi = self.model.features(x)
        logits = phi @ self.W
        probs = F.softmax(logits.squeeze(0), dim=-1)
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
        for byte in char.encode("utf-8"):
            self._ctx.append(byte)
            if len(self._ctx) > K:
                self._ctx = self._ctx[-K:]


def train(train_text: str, valid_text: str | None = None) -> CharModel:
    seed_env = os.environ.get("SEED")
    seed = int(seed_env) if seed_env else 0
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    print(f"[p2a] SEED={seed}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, W = _train(train_text, device, seed)
    return FrozenRidgeCharModel(model, W, device)
