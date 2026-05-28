"""Mono-Forward No-Grad (N3): closed-form, zero-SGD layered byte predictor.

Strict ablation of `mono_forward_v2` (0.7346 / 46.2 kJ): remove **all**
backpropagation. The block weights and probe heads are replaced by:

  - block_l: a fixed random nonlinear featurizer phi_l of a byte-context
    window of length K_l (different K per layer for feature diversity).
  - probe_l: a closed-form multiclass ridge classifier R_l fit by
    Cholesky on (Phi_l^T Phi_l + lambda I) W = Phi_l^T Y_l, where Y_l is
    the *residual target* — the one-hot next-byte minus cumulative logits
    from layers 0..l-1.

Inference logits at a position = sum_l R_l(phi_l(window)), then softmax.

This is gradient-boosted ridge regression over fixed random nonlinear
features of byte windows. It survives the three structural findings:
- stochasticity filter: ridge fits expectation of Phi^T Y (no one-shot WTA);
- Paradigm-A ceiling (single fixed-feature ridge ~0.37): defeated by
  additive boosting against successive residuals;
- n-gram floor: byte-context window is the sufficient statistic for any
  n-gram, so the lower bound is at least competitive with small-window
  smoothing.

The experiment is a falsification probe. If even boosted random-feature
ridge cannot clear the 0.70 floor on byte windows, the lesson is that
*layer-local closed-form supervision is insufficient without parameter
adaptation*, which would reinforce the "Mono-Forward needs SGD on the
blocks" conclusion latent in the v2 PASS.
"""
from __future__ import annotations

__author__ = "@armin-claude-1m"

import math
import os
import time

import torch
import torch.nn.functional as F
from torch import Tensor

from wikitext import CharModel


# ---------------------------------------------------------------------------
# Featurizer: hashed n-gram one-hot + raw byte-position concat
# ---------------------------------------------------------------------------

class HashedContextFeaturizer:
    """Map a byte-window (B, K) of bytes in [0, 256) to a dense feature
    vector of dimension d_feat.

    Features:
      1. Per-position raw byte one-hots: K * 256 dims, but encoded as a
         single random-projection matrix P_pos of shape (K, 256, d_pos)
         summed across positions, optionally weighted by 1/sqrt(K).
         Compactly: index P_pos[k, byte_k, :] then sum over k.
         This is a random projection of the trigram-like positional
         encoding; with d_pos large enough this preserves enough info to
         distinguish frequent byte-patterns.
      2. Random projection of n-gram hashes: for each suffix length
         m in [1..K], take a 64-bit FNV-style hash of the suffix bytes
         modulo d_hash; one-hot at that bucket. Sum over m. Random sign
         to get +/-1 buckets so collisions cancel in expectation
         (count-sketch flavour).
      3. ReLU activation on the concatenation, fp32.

    The featurizer is *fixed at construction time* and never updated.
    """

    def __init__(
        self,
        K: int,
        d_pos: int,
        d_hash: int,
        device: torch.device,
        seed: int = 0,
    ):
        self.K = K
        self.d_pos = d_pos
        self.d_hash = d_hash
        self.device = device
        self.d_feat = d_pos + d_hash

        g = torch.Generator(device="cpu").manual_seed(seed)
        # Random projection table: (K, 256, d_pos). Indexed by (k, byte_k).
        # Normalize so the sum across K positions has unit variance.
        std_pos = 1.0 / math.sqrt(K)
        self.P_pos = (
            torch.randn(K, 256, d_pos, generator=g) * std_pos
        ).to(device=device, dtype=torch.float32)

        # Random rolling-hash multipliers (one per suffix-length m).
        # Use distinct primes to randomize across suffix lengths.
        primes = [
            1000003, 1000033, 1000037, 1000039, 1000081, 1000099,
            1000117, 1000121, 1000133, 1000151, 1000159, 1000171,
            1000183, 1000187, 1000193, 1000199, 1000211, 1000213,
            1000231, 1000249, 1000253, 1000273, 1000289, 1000291,
            1000303, 1000313, 1000333, 1000357, 1000367, 1000381,
            1000393, 1000397,
        ]
        # Use first K primes; if K > 32, cycle.
        self.hash_mults = torch.tensor(
            [primes[m % len(primes)] for m in range(K)],
            dtype=torch.int64, device=device,
        )
        # Random hash bias per suffix length to decorrelate.
        bias_g = torch.Generator(device="cpu").manual_seed(seed + 17)
        self.hash_bias = torch.randint(
            0, 1 << 30, (K,), generator=bias_g, dtype=torch.int64,
        ).to(device)
        # Random sign per (suffix-length, bucket). Count-sketch flavour.
        sign_g = torch.Generator(device="cpu").manual_seed(seed + 31)
        self.hash_signs = (
            (torch.randint(0, 2, (K, d_hash), generator=sign_g, dtype=torch.int8) * 2 - 1)
            .to(device=device, dtype=torch.float32)
        )

    def features(self, windows: Tensor) -> Tensor:
        """Featurize a batch of windows.

        Args:
            windows: (N, K) int64 in [0, 256). Position 0 is the oldest
                byte, position K-1 is the most recent observed byte.
                The byte at position K is the one we want to predict.

        Returns:
            (N, d_feat) fp32 features.
        """
        N, K = windows.shape
        assert K == self.K, f"window K={K} != featurizer K={self.K}"

        # ---- Part 1: positional random projection ------------------------
        # Want f_pos[n, :] = sum_k P_pos[k, windows[n, k], :]
        # Implement via gather: P_pos[k] is (256, d_pos); index by bytes.
        # Reshape P_pos to (K*256, d_pos); index by (k*256 + byte) per (n,k).
        idx = (torch.arange(K, device=self.device).unsqueeze(0) * 256
               + windows)  # (N, K)
        P_flat = self.P_pos.view(K * 256, self.d_pos)
        # gather rows: (N, K, d_pos), then sum over K.
        f_pos = P_flat[idx.reshape(-1)].view(N, K, self.d_pos).sum(dim=1)

        # ---- Part 2: rolling-hash n-gram features ------------------------
        # For each suffix length m in [1, K], compute h_m =
        #   ((sum_{i=K-m}^{K-1} bytes[i] * mult^{K-1-i}) + bias_m) mod d_hash
        # then add hash_signs[m, h_m] to feature bucket h_m.
        #
        # Vectorize across (N, K): compute h_m for *every* suffix length
        # using a cumulative rolling hash.
        #
        # Define rolling hash from the right end:
        #   h[0] = bytes[K-1]
        #   h[m] = h[m-1] * mult + bytes[K-1-m]    (for m in 1..K-1)
        # This gives K hash values per row, each one a hash of the last
        # (m+1)-byte suffix.
        windows_i64 = windows.to(torch.int64)
        # Reverse so position 0 = newest byte.
        rev = windows_i64.flip(dims=[1])  # (N, K) — rev[:, m] = byte at suffix length m+1
        # Build h[:, m] = sum_{j=0}^{m} rev[:, j] * mult^j  (mod d_hash)
        # Use cumulative power-of-mult trick. But d_hash is small (< 2^17);
        # using int64 modular arithmetic with chain.
        d_hash = self.d_hash
        mult = int(1_000_003)
        # Cumulative powers of mult, mod d_hash.
        powers = torch.empty(K, dtype=torch.int64, device=self.device)
        p = 1
        for j in range(K):
            powers[j] = p
            p = (p * mult) % d_hash
        # Now h[:, m] = sum_{j=0}^{m} (rev[:, j] * powers[j]) mod d_hash
        rev_mod = (rev * powers.unsqueeze(0)) % d_hash  # (N, K)
        # Cumulative sum modulo d_hash.
        cum = torch.cumsum(rev_mod, dim=1) % d_hash  # (N, K)
        # Add bias per suffix length.
        h = (cum + self.hash_bias.unsqueeze(0)) % d_hash  # (N, K)

        # Sum count-sketch contributions: for each (n, m), add
        # hash_signs[m, h[n, m]] to bucket h[n, m].
        # Vectorize with scatter_add.
        f_hash = torch.zeros(N, d_hash, device=self.device, dtype=torch.float32)
        # signs[n, m] = hash_signs[m, h[n, m]]
        # gather: hash_signs is (K, d_hash); index hash_signs[m, h]
        m_idx = torch.arange(K, device=self.device).unsqueeze(0).expand(N, K)
        signs = self.hash_signs[m_idx, h]  # (N, K)
        f_hash.scatter_add_(1, h, signs)

        # ---- Combine + nonlinearity --------------------------------------
        feats = torch.cat([f_pos, f_hash], dim=1)
        # ReLU — cheap, breaks the "linear-on-fixed-features" Paradigm-A
        # collapse by giving the readout a piecewise-linear basis.
        feats = F.relu(feats)
        # Append a bias term (constant 1) so the ridge has an intercept.
        bias = torch.ones(N, 1, device=self.device, dtype=torch.float32)
        feats = torch.cat([feats, bias], dim=1)
        return feats


# ---------------------------------------------------------------------------
# Closed-form layer fit: incremental Gram + Cholesky-solved multiclass ridge
# ---------------------------------------------------------------------------

class RidgeLayer:
    """One layer of the boosting stack.

    Holds:
      - the featurizer (fixed at construction time);
      - the fit ridge weights W of shape (d_feat + 1, 256);
      - hyperparameters.

    Accumulates Phi^T Phi and Phi^T Y over minibatches, then Cholesky-solves
    once at the end of the fit phase.
    """

    def __init__(
        self,
        featurizer: HashedContextFeaturizer,
        ridge_lambda: float,
        device: torch.device,
    ):
        self.featurizer = featurizer
        self.K = featurizer.K
        self.ridge_lambda = ridge_lambda
        self.device = device
        self.d_feat = featurizer.d_feat + 1  # +1 for bias term
        self.W: Tensor | None = None  # (d_feat+1, 256), to be fit

        # Accumulators in fp32. TF32 matmul on A100 is ~150 TFLOP/s, so
        # per-batch Gram updates stay cheap. Cholesky upgrades to fp64
        # internally for numeric stability of the small d_feat×d_feat solve.
        self._PtP = torch.zeros(
            self.d_feat, self.d_feat, device=device, dtype=torch.float32,
        )
        self._PtY = torch.zeros(
            self.d_feat, 256, device=device, dtype=torch.float32,
        )
        self._N_seen = 0

    def accumulate(self, windows: Tensor, targets: Tensor) -> None:
        """Add a minibatch to the Gram and right-hand side.

        Args:
            windows: (N, K) int byte windows.
            targets: (N, 256) fp32 residual targets (one-hot - cumulative
                ridge predictions from previous layers).
        """
        feats = self.featurizer.features(windows)  # (N, d_feat)
        self._PtP.addmm_(feats.T, feats)
        self._PtY.addmm_(feats.T, targets)
        self._N_seen += feats.shape[0]

    def fit(self) -> None:
        """Cholesky-solve (PtP + lambda * N * I) W = PtY.

        Upgrades to fp64 for the solve (d_feat×d_feat is small, ~10ms
        on A100 even at fp64); the fp32 Gram is preserved for memory.
        """
        # Scale ridge by N to keep regularization comparable across data sizes.
        reg = self.ridge_lambda * max(1, self._N_seen)
        PtP64 = self._PtP.to(torch.float64)
        PtY64 = self._PtY.to(torch.float64)
        A = PtP64 + reg * torch.eye(
            self.d_feat, device=self.device, dtype=torch.float64,
        )
        # Cholesky solve.
        try:
            L = torch.linalg.cholesky(A)
            W64 = torch.cholesky_solve(PtY64, L)
        except Exception:
            # Fallback to LU.
            W64 = torch.linalg.solve(A, PtY64)
        self.W = W64.to(torch.float32)
        # Free accumulators.
        self._PtP = None  # type: ignore[assignment]
        self._PtY = None  # type: ignore[assignment]

    def predict_logits(self, windows: Tensor) -> Tensor:
        """Return (N, 256) raw logits for a batch of windows."""
        if self.W is None:
            raise RuntimeError("predict_logits called before fit")
        feats = self.featurizer.features(windows)  # (N, d_feat)
        return feats @ self.W  # (N, 256)


# ---------------------------------------------------------------------------
# Boosting stack training
# ---------------------------------------------------------------------------

def _build_windows(
    bytes_tensor: Tensor,
    K_max: int,
    n_positions: int,
    rng: torch.Generator,
) -> tuple[Tensor, Tensor]:
    """Sample n_positions random training positions and return
    (windows_Kmax, targets_int).

    windows_Kmax: (n_positions, K_max) bytes — context window for each
        position (positions K_max..). Positions earlier than K_max are
        padded with byte 0.

    targets_int: (n_positions,) byte indices in [0, 256).

    Positions where the context is too short get zero-padded contexts
    (this is fine — the model treats byte 0 as a real symbol; trivially
    rare in real data anyway).
    """
    n = bytes_tensor.numel()
    # Sample positions in [0, n-1]; the target is bytes_tensor[pos].
    # Context window is bytes_tensor[pos-K_max : pos], left-padded with 0.
    pos = torch.randint(0, n, (n_positions,), generator=rng, device=bytes_tensor.device)
    # Build offsets matrix: (n_positions, K_max) of indices into bytes_tensor.
    offs = torch.arange(-K_max, 0, device=bytes_tensor.device).unsqueeze(0) + pos.unsqueeze(1)
    # Clip negative indices to 0 (those positions sample byte_tensor[0]); we
    # mask them after the gather to be 0.
    neg_mask = offs < 0
    offs_clipped = offs.clamp(min=0, max=n - 1)
    windows = bytes_tensor[offs_clipped].to(torch.int64)
    windows = windows.masked_fill(neg_mask, 0)
    targets = bytes_tensor[pos].to(torch.int64)
    return windows, targets


def _train_stack(
    train_bytes: Tensor,
    *,
    layer_Ks: list[int],
    d_pos: int,
    d_hash: int,
    ridge_lambda: float,
    n_train_positions: int,
    batch_size: int,
    boost_lr: float,
    max_seconds: float,
    seed: int,
    device: torch.device,
) -> list[RidgeLayer]:
    """Train L = len(layer_Ks) boosting layers on byte windows.

    For each layer l (in order):
      1. Sample n_train_positions windows of length max(K) (we slice K_l
         out at featurize time).
      2. Compute residual target Y_l = onehot(y) - sigma * sum_{l'<l} R_{l'}(phi_{l'})
         where sigma is a smoothing scalar mapping accumulated logits to
         probabilities. Here we use the linear-logit formulation: just
         subtract previous-layer raw logits; final inference applies
         softmax at the end.
      3. Accumulate Phi^T Phi, Phi^T Y over minibatches; fit by Cholesky.
      4. Add this layer to the stack; proceed to next.

    Returns the list of fit RidgeLayer objects.
    """
    t_start = time.monotonic()
    layers: list[RidgeLayer] = []
    K_max = max(layer_Ks)

    # Sample windows ONCE per layer fit (re-sample each layer so feature
    # accumulation sees fresh data).
    n_batches = max(1, n_train_positions // batch_size)

    for l, K in enumerate(layer_Ks):
        elapsed = time.monotonic() - t_start
        budget_left = max_seconds - elapsed
        if budget_left < 5.0:
            print(f"[mfng] layer {l} skipped — budget exhausted "
                  f"(elapsed={elapsed:.1f}s)", flush=True)
            break

        rng = torch.Generator(device=device).manual_seed(seed + l * 1009)
        # Per-layer featurizer with its own context length K_l. Each layer
        # uses the last K_l bytes of the (longer) window we sample.
        feat_l = HashedContextFeaturizer(
            K=K, d_pos=d_pos, d_hash=d_hash,
            device=device, seed=seed + l * 7919,
        )
        layer = RidgeLayer(feat_l, ridge_lambda=ridge_lambda, device=device)

        # Accumulate Phi^T Phi, Phi^T Y across batches.
        for b in range(n_batches):
            if time.monotonic() - t_start > max_seconds - 2.0:
                break
            windows_full, targets = _build_windows(
                train_bytes, K_max, batch_size, rng,
            )
            windows_l = windows_full[:, -K:]  # use last K bytes for this layer

            # Build target Y: one-hot - sum of previous layers' raw logits.
            Y = F.one_hot(targets, num_classes=256).to(torch.float32)
            for prev in layers:
                prev_K = prev.K
                prev_windows = windows_full[:, -prev_K:]
                Y = Y - boost_lr * prev.predict_logits(prev_windows)

            layer.accumulate(windows_l, Y)

        layer.fit()
        layers.append(layer)
        elapsed = time.monotonic() - t_start
        print(
            f"[mfng] layer {l+1}/{len(layer_Ks)} fit  K={K}  d_feat={feat_l.d_feat+1}  "
            f"N_seen={layer._N_seen}  total_t={elapsed:.1f}s",
            flush=True,
        )

    print(
        f"[mfng] stack trained: {len(layers)} layers  "
        f"total_t={time.monotonic()-t_start:.1f}s",
        flush=True,
    )
    return layers


# ---------------------------------------------------------------------------
# Streaming CharModel wrapper
# ---------------------------------------------------------------------------

class MFNGCharModel(CharModel):
    """Streaming inference: maintain a rolling byte buffer of length
    max(K_l). For each predict(), featurize the window once per layer
    and sum logits.
    """

    def __init__(
        self,
        layers: list[RidgeLayer],
        boost_lr: float,
        device: torch.device,
    ):
        self.layers = layers
        self.boost_lr = boost_lr
        self.device = device
        self.K_max = max((l.K for l in layers), default=1)
        # Rolling buffer of the last K_max observed bytes (int8 storage,
        # int64 view for indexing).
        self._buf: list[int] = []

    def reset(self) -> None:
        self._buf = []

    def predict(self) -> dict[str, float]:
        # Build a window of length K_max, left-pad with 0 if buffer < K_max.
        if len(self._buf) >= self.K_max:
            ctx = self._buf[-self.K_max:]
        else:
            ctx = [0] * (self.K_max - len(self._buf)) + self._buf
        window_full = torch.tensor(ctx, dtype=torch.int64, device=self.device).unsqueeze(0)

        # Accumulate logits across layers.
        logits = torch.zeros(1, 256, device=self.device, dtype=torch.float32)
        for l_idx, layer in enumerate(self.layers):
            K = layer.K
            window_l = window_full[:, -K:]
            l_logits = layer.predict_logits(window_l)
            # Boosting weight: first layer uses 1.0 (it fits the raw target),
            # subsequent layers use boost_lr (they fit residuals scaled by
            # boost_lr in training; inference must match).
            if l_idx == 0:
                logits = logits + l_logits
            else:
                logits = logits + self.boost_lr * l_logits

        probs = F.softmax(logits.squeeze(0), dim=-1)
        # Build output dict.
        out: dict[str, float] = {}
        probs_list = probs.tolist()
        for byte_id, p in enumerate(probs_list):
            try:
                ch = bytes([byte_id]).decode("utf-8")
            except UnicodeDecodeError:
                continue
            out[ch] = p
        return out

    def observe(self, char: str) -> None:
        for byte in char.encode("utf-8"):
            self._buf.append(byte)
            # Cap the buffer to K_max to keep memory bounded.
            if len(self._buf) > self.K_max:
                self._buf = self._buf[-self.K_max:]


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def train(train_text: str, valid_text: str | None = None) -> CharModel:
    seed_env = os.environ.get("SEED")
    seed = int(seed_env) if seed_env else 0
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Enable TF32 for fp32 matmuls — Gram updates dominate training cost
    # and tolerate the ~1e-3 precision loss; ridge regularization
    # absorbs any small numerical drift.
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    print(f"[mfng] device={device}  SEED={seed}", flush=True)

    # ---- Data ------------------------------------------------------------
    raw = train_text.encode("utf-8")
    train_bytes = torch.frombuffer(bytearray(raw), dtype=torch.uint8).to(device)
    print(f"[mfng] train_bytes: {train_bytes.numel():,}", flush=True)

    # ---- Hyperparameters -------------------------------------------------
    # Context windows per layer: grow from short (catches positional
    # bigrams) to long (catches longer-range structure). 6 layers to
    # mirror the mono_forward_v2 6-block stack.
    layer_Ks = [2, 4, 8, 16, 24, 32]
    d_pos = 1024     # random projection dim of per-position one-hots
    d_hash = 8192    # count-sketch hash dim for n-gram features
    ridge_lambda = 1e-4
    boost_lr = 0.5   # shrinkage factor for subsequent layers (boosting)

    # Training scale. d_feat ≈ d_pos + d_hash = ~9k. PtP is ~9k×9k =
    # 81M fp64 = ~650 MB. We have 80 GB HBM — fine for one layer at a
    # time. Cholesky on 9k×9k fp64 is ~30 GFLOP → seconds on A100.
    batch_size = 8192       # per accumulation batch
    n_train_positions = 8_000_000  # 8M positions total per layer ~ 1k batches
    # Total budget: 300 s minus margin. Featurize + Gram update is the
    # bottleneck. Estimate 1k batches/layer * 6 layers * ~30 ms/batch
    # = ~180 s. Leave 30 s margin for Cholesky + setup.
    max_seconds = 260.0

    # ---- Train the boosting stack ---------------------------------------
    layers = _train_stack(
        train_bytes,
        layer_Ks=layer_Ks,
        d_pos=d_pos,
        d_hash=d_hash,
        ridge_lambda=ridge_lambda,
        n_train_positions=n_train_positions,
        batch_size=batch_size,
        boost_lr=boost_lr,
        max_seconds=max_seconds,
        seed=seed,
        device=device,
    )

    return MFNGCharModel(layers, boost_lr=boost_lr, device=device)
