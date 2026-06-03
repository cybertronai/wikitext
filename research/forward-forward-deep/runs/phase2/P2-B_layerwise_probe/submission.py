"""P2-B — Per-Layer Ridge Probes (FF investigation diagnostic).

Identical FF training to pass-2; the only change is the readout. Fit
FIVE independent ridge readouts (one per layer 1..5) and print each
readout's accuracy on a 20K-char chunk of the val stream. The submission
uses whichever single-layer readout has the best train-subset accuracy.

The five probe accuracies are the diagnostic deliverable — they reveal
whether FF builds hierarchy across the stack.

Spec: .survey/ff_runs/phase2/P2-B_layerwise_probe/design.md
"""
from __future__ import annotations

__author__ = "@survey-ff-p2b"

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
THETA = 2.0
BATCH = 256
N_STEPS = 14000
LR = 3e-4
BETAS = (0.9, 0.99)

HARD_NEG_EVERY = 500
HARD_NEG_FRACTION = 0.5
HARD_NEG_TOPK = 5
HARD_NEG_REFIT_N = 20_000

RIDGE_N_FIT = 80_000
RIDGE_LAMBDA = 1.0

EVAL_BATCH = 256
RIDGE_FORWARD_BATCH = 512

# 20K-char diagnostic probe over the val stream.
PROBE_N = 20_000


class FFLayer(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.lin = nn.Linear(in_dim, out_dim, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        return F.relu(self.lin(x))


def l2_normalize(x: Tensor, eps: float = 1e-8) -> Tensor:
    return x / (x.norm(dim=-1, keepdim=True) + eps)


class FFStack(nn.Module):
    def __init__(self):
        super().__init__()
        layers = [FFLayer(INPUT_DIM, WIDTH)]
        for _ in range(N_LAYERS - 1):
            layers.append(FFLayer(WIDTH, WIDTH))
        self.layers = nn.ModuleList(layers)
        for p in self.layers[0].parameters():
            p.requires_grad_(False)

    def forward_all(self, x: Tensor) -> list[Tensor]:
        acts: list[Tensor] = []
        h = x
        for layer in self.layers:
            a = layer(h.detach())
            acts.append(a)
            h = l2_normalize(a)
        return acts

    @torch.no_grad()
    def per_layer_features(self, x: Tensor) -> list[Tensor]:
        """Return [LN(a_1), LN(a_2), ..., LN(a_N)] — one per layer."""
        feats = []
        h = x
        for layer in self.layers:
            a = layer(h)
            ln = l2_normalize(a)
            feats.append(ln)
            h = ln
        return feats


class UnigramSampler:
    def __init__(self, byte_counts: Tensor, device: torch.device, generator: torch.Generator):
        probs = byte_counts.float() / byte_counts.float().sum()
        self.probs = probs.to(device)
        self.device = device
        self.generator = generator

    def sample(self, true_bytes: Tensor) -> Tensor:
        n = true_bytes.numel()
        neg = torch.multinomial(self.probs, n, replacement=True, generator=self.generator)
        for _ in range(8):
            mask = neg == true_bytes
            if not mask.any():
                break
            neg[mask] = torch.multinomial(
                self.probs, int(mask.sum()), replacement=True, generator=self.generator,
            )
        mask = neg == true_bytes
        if mask.any():
            neg[mask] = (true_bytes[mask] + 1) % VOCAB
        return neg


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
def _extract_per_layer_features(
    model: FFStack, train_bytes: Tensor, sample_idx: Tensor, batch_size: int,
) -> tuple[list[Tensor], Tensor]:
    n_fit = sample_idx.numel()
    feats = [
        torch.empty(n_fit, WIDTH, device=train_bytes.device, dtype=torch.float32)
        for _ in range(N_LAYERS)
    ]
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
        per_layer = model.per_layer_features(x)
        for li in range(N_LAYERS):
            feats[li][start:end] = per_layer[li]
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


def _build_negatives(
    true_bytes: Tensor,
    unigram_sampler: UnigramSampler,
    hard_logits: Tensor | None,
    hard_fraction: float,
    topk: int,
    generator: torch.Generator,
) -> Tensor:
    B = true_bytes.numel()
    if hard_logits is None or hard_fraction <= 0.0:
        return unigram_sampler.sample(true_bytes)
    n_hard = int(round(B * hard_fraction))
    neg = unigram_sampler.sample(true_bytes)
    if n_hard == 0:
        return neg
    masked = hard_logits.clone()
    masked.scatter_(1, true_bytes.unsqueeze(1), float("-inf"))
    topk_vals, topk_idx = masked.topk(topk, dim=-1)
    pick = torch.randint(0, topk, (B,), device=true_bytes.device, generator=generator)
    hard_neg = topk_idx.gather(1, pick.unsqueeze(1)).squeeze(1)
    perm = torch.randperm(B, device=true_bytes.device, generator=generator)
    swap_pos = perm[:n_hard]
    neg[swap_pos] = hard_neg[swap_pos]
    mask = neg == true_bytes
    if mask.any():
        neg[mask] = (true_bytes[mask] + 1) % VOCAB
    return neg


def _train_ff(train_text: str, device: torch.device, seed: int) -> tuple[FFStack, list[Tensor], int]:
    """Train FF, fit per-layer ridges, return (model, [W_1..W_N], best_layer_idx)."""
    raw = train_text.encode("utf-8")
    train_bytes = torch.frombuffer(bytearray(raw), dtype=torch.uint8).to(device)
    n = train_bytes.numel()
    if n < K + 1:
        raise ValueError(f"need at least {K+1} bytes; got {n}")

    byte_counts = torch.bincount(train_bytes.long(), minlength=VOCAB).cpu() + 1
    gen = torch.Generator(device=device)
    gen.manual_seed(seed)

    model = FFStack().to(device)
    optimizers = [
        torch.optim.Adam(model.layers[i].parameters(), lr=LR, betas=BETAS, weight_decay=0.0)
        for i in range(1, N_LAYERS)
    ]
    unigram = UnigramSampler(byte_counts, device, gen)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"[p2b] {n_params/1e6:.2f}M params  layers={N_LAYERS} width={WIDTH} K={K}  "
          f"steps={N_STEPS}  (per-layer ridge probes)")

    model.train()
    t0 = time.monotonic()
    log_every = 1000
    max_start = n - K - 1
    hard_W: Tensor | None = None

    # For hard-neg sampling we still need a single concat-of-layers ridge,
    # built exactly like pass-2 — keeps training behaviour identical.
    @torch.no_grad()
    def _concat_features(x: Tensor) -> Tensor:
        h = x
        feats = []
        for li, layer in enumerate(model.layers):
            a = layer(h)
            if li >= 1:
                feats.append(l2_normalize(a))
            h = l2_normalize(a)
        return torch.cat(feats, dim=-1)

    def _refit_hardneg() -> Tensor:
        sample_idx = torch.randint(0, max_start, (HARD_NEG_REFIT_N,), device=device, generator=gen)
        feats_list, tgt = _extract_per_layer_features(model, train_bytes, sample_idx, RIDGE_FORWARD_BATCH)
        # Concat layers 2..N for the hard-neg readout, matching pass-2.
        phi = torch.cat(feats_list[1:], dim=-1)
        return _solve_ridge(phi, tgt, RIDGE_LAMBDA)

    for step in range(N_STEPS):
        if step > 0 and step % HARD_NEG_EVERY == 0:
            t_re = time.monotonic()
            hard_W = _refit_hardneg()
            model.train()
            print(f"[p2b] step {step:5d}  hard-neg ridge refit ({HARD_NEG_REFIT_N} samples, "
                  f"{time.monotonic()-t_re:.2f}s)", flush=True)

        idx = torch.randint(0, max_start, (BATCH,), device=device, generator=gen)
        offsets = idx[:, None] + torch.arange(K + 1, device=device)[None, :]
        windows = train_bytes[offsets].long()
        ctx = windows[:, :K]
        true_byte = windows[:, K]

        if hard_W is not None:
            with torch.no_grad():
                phi_ctx = _concat_features(build_input(ctx, None))
                hard_logits = phi_ctx @ hard_W
            model.train()
        else:
            hard_logits = None

        neg_byte = _build_negatives(
            true_byte, unigram, hard_logits, HARD_NEG_FRACTION, HARD_NEG_TOPK, gen,
        )

        x_pos = build_input(ctx, true_byte)
        x_neg = build_input(ctx, neg_byte)
        x = torch.cat([x_pos, x_neg], dim=0)
        acts = model.forward_all(x)

        total = 0.0
        for li in range(1, N_LAYERS):
            a = acts[li]
            g_pos = (a[:BATCH] ** 2).sum(dim=-1)
            g_neg = (a[BATCH:] ** 2).sum(dim=-1)
            loss = F.softplus(THETA - g_pos).mean() + F.softplus(g_neg - THETA).mean()
            opt = optimizers[li - 1]
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            total += loss.item()

        if step % log_every == 0 or step == N_STEPS - 1:
            elapsed = time.monotonic() - t0
            print(f"[p2b] step {step:5d}/{N_STEPS}  loss(sum) {total:.4f}  "
                  f"elapsed {elapsed:.0f}s", flush=True)

    # ---- Per-layer ridge fits ----
    t_ridge = time.monotonic()
    sample_idx = torch.randint(0, max_start, (RIDGE_N_FIT,), device=device, generator=gen)
    feats_list, tgt = _extract_per_layer_features(model, train_bytes, sample_idx, RIDGE_FORWARD_BATCH)
    W_list = [_solve_ridge(phi, tgt, RIDGE_LAMBDA) for phi in feats_list]
    print(f"[ridge] per-layer fits: N_fit={RIDGE_N_FIT} per_dim={WIDTH} "
          f"fit_s={time.monotonic()-t_ridge:.1f}", flush=True)

    # Train-subset accuracy per layer (probe — picks the "submission" readout).
    with torch.no_grad():
        diag_n = min(20_000, RIDGE_N_FIT)
        accs = []
        for li in range(N_LAYERS):
            pred = (feats_list[li][:diag_n] @ W_list[li]).argmax(dim=-1)
            acc = (pred == tgt[:diag_n]).float().mean().item()
            accs.append(acc)
            print(f"[probe-train] layer {li+1}: train_acc={acc:.4f}", flush=True)
    best_layer = max(range(N_LAYERS), key=lambda i: accs[i])
    print(f"[probe-train] best layer by train acc: layer {best_layer+1} (acc={accs[best_layer]:.4f})",
          flush=True)
    return model, W_list, best_layer


class PerLayerCharModel(CharModel):
    """Uses the best single-layer ridge as the active predictor."""

    def __init__(
        self,
        model: FFStack,
        W_list: list[Tensor],
        best_layer: int,
        device: torch.device,
        valid_probe: tuple[Tensor, Tensor] | None = None,
    ):
        self.model = model
        self.W_list = W_list
        self.best_layer = best_layer
        self.device = device
        self.model.eval()
        self._ctx: list[int] = []
        self._probe_done = False
        self._valid_probe = valid_probe

    @torch.no_grad()
    def reset(self) -> None:
        self._ctx = []
        # On the first reset (start of the gated val pass), run the
        # five-readout diagnostic over the first PROBE_N val chars.
        if not self._probe_done and self._valid_probe is not None:
            feats_list, tgt = self._valid_probe
            for li in range(N_LAYERS):
                logits = feats_list[li] @ self.W_list[li]
                pred = logits.argmax(dim=-1)
                acc = (pred == tgt).float().mean().item()
                print(f"[probe-val] layer {li+1}: val_acc(n={tgt.numel()})={acc:.4f}",
                      flush=True)
            self._probe_done = True

    @torch.no_grad()
    def _build_one(self) -> Tensor:
        pad = K - len(self._ctx)
        ctx_bytes = ([0] * pad + self._ctx) if pad > 0 else self._ctx[-K:]
        ctx_t = torch.tensor(ctx_bytes, dtype=torch.long, device=self.device).unsqueeze(0)
        return build_input(ctx_t, None)

    @torch.no_grad()
    def predict(self) -> str:
        x = self._build_one()
        per_layer = self.model.per_layer_features(x)
        phi = per_layer[self.best_layer]
        logits = phi @ self.W_list[self.best_layer]
        probs = F.softmax(logits.squeeze(0), dim=-1)
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
        for byte in char.encode("utf-8"):
            self._ctx.append(byte)
            if len(self._ctx) > K:
                self._ctx = self._ctx[-K:]


@torch.no_grad()
def _build_valid_probe(
    model: FFStack, valid_text: str | None, device: torch.device,
) -> tuple[Tensor, Tensor] | None:
    if not valid_text:
        return None
    raw = valid_text.encode("utf-8")[: PROBE_N + K]
    if len(raw) < K + 1:
        return None
    vbytes = torch.frombuffer(bytearray(raw), dtype=torch.uint8).to(device)
    n_pos = vbytes.numel() - K
    n_pos = min(n_pos, PROBE_N)
    if n_pos <= 0:
        return None
    idx = torch.arange(n_pos, device=device)
    feats_list, tgt = _extract_per_layer_features(model, vbytes, idx, RIDGE_FORWARD_BATCH)
    return feats_list, tgt


def train(train_text: str, valid_text: str | None = None) -> CharModel:
    seed_env = os.environ.get("SEED")
    seed = int(seed_env) if seed_env else 0
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    print(f"[p2b] SEED={seed}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, W_list, best_layer = _train_ff(train_text, device, seed)
    # Build a val-set probe so per-layer accuracies are reported on the
    # actual val stream — far more informative than train-subset acc.
    valid_probe = _build_valid_probe(model, valid_text, device)
    return PerLayerCharModel(model, W_list, best_layer, device, valid_probe)
