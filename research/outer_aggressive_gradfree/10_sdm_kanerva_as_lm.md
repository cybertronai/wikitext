# 10 · Sparse Distributed Memory (Kanerva) as the Entire LM

## Mechanism

Pentti Kanerva's *Sparse Distributed Memory* (1988, MIT Press) is a
classical associative memory that uses random *hard-location addresses* —
random hyperplanes in a high-dim binary space — as content-addressable
buckets. Each datum (key, value) is written to *all locations whose hard
address is within Hamming radius r of the key*. A read at key k_q averages
the values stored at all locations within Hamming radius r of k_q. The
addressing is fully decoupled from the content, so all writes are local
Hebbian additions to a content matrix.

Critically:

- Addresses A ∈ {-1,+1}^{M × d_addr} are *random and fixed at init*. M
  is the number of hard locations (10^4-10^6).
- Content matrix C ∈ R^{M × d_content} starts at zero.
- Write(k, v): mask = (sign(k · A^T) > threshold). C[mask] += v.
- Read(k): mask = same condition. Return mean of C[mask].

This is the **Universal Hopfield Memory** instantiation with binary
similarity + threshold separation + mean projection — see #5 for the
general framework.

For char-LM: encode byte windows into binary {-1, +1} addresses via
random hyperplane projection. Write (window_address, next_byte_one_hot)
pairs. At inference, read at the current window's address, softmax the
result.

**No gradient. No optimization. No closed-form solve.** Just hard-mask
sums and means.

## Seed papers

- Kanerva, *Sparse Distributed Memory*, MIT Press 1988. The original.
- Wu, Hutchins, Szegedy, Rabe, *Memorizing Transformers*, ICLR 2022.
  The modern kNN-LM is an SDM with a softer addressing function.
- Bricken & Pehlevan, *Attention Approximates Sparse Distributed Memory*,
  NeurIPS 2021 (arXiv 2111.05498). Establishes that transformer attention
  IS SDM with cosine similarity + softmax separation. This means the
  existing `hopfield_layer` and SDM are mathematically very close — but
  the SDM has zero learned parameters, making it a clean baseline.

## Why it could work here

- **Zero gradient steps, zero closed-form solves.** Training is a single
  pass of pure additions into a sparse-mask matrix.
- **SDM addressing capacity is O(M)** patterns reliably. With M = 10^6
  hard locations and full memory, capacity is enormous.
- **Bricken-Pehlevan equivalence** says that if attention works, SDM
  works (with cosine vs. Hamming similarity). The performance gap
  measures how much the soft softmax matters.
- **A100 efficient:** the address-mask computation is a single (B, d_addr)
  × (d_addr, M) matmul + threshold. With M = 10^6, d_addr = 256, that's
  a 256 MB matmul, ~2 GB HBM traffic, ~1 ms.

## Threshold of plausibility

SDM has the same fixed-feature problem as #5 and #8 — it doesn't learn
representations. The addressing capacity ≠ the *useful* capacity for
modal-byte prediction. Patterns stored at addresses corresponding to rare
contexts dilute the modal-byte signal at common contexts.

But: SDM's hard masking is actually a useful inductive bias for byte LM
because it implements *exact-match lookup with locality*. A context that
appeared before with byte b will read out b plus a small noise mix from
nearby contexts. This is closer to k-NN than to RFF + ridge.

Estimate: 0.40–0.55. Likely worse than #5 because of the hard masking
discontinuity, but could surprise. **Capability demo** + cheapest possible
baseline for "Hebbian outer-product LM".

## Failure modes

- **Hamming radius r is the key hyperparameter.** Too small → no overlap,
  return zeros. Too large → universal pollution. Calibrate to give average
  ~10-50 active locations per query.
- **Stochasticity:** averaging values across multiple active locations
  is naturally a soft distribution; passes the stochasticity filter.
- **Capacity exhaustion at small M:** at M = 10 K, only a tiny fraction
  of the train stream's contexts can be represented. Use M = 10^6 with
  bit-packed addressing if memory is tight.
- **Bit-packed mask compute:** a 10^6 × 256-bit address matrix is 32 MB.
  Mask compute is a popcount-style operation. PyTorch's signed-int matmul
  +threshold approach gives the equivalent in fp16 at higher mem cost but
  faster wall-clock.

What would falsify it: M = 10^6, d_addr = 256, r tuned, val acc ≤ 0.40 →
confirms hard-masking + fixed addresses can't beat paradigm-A. Reject.

## Smallest first experiment

`sdm_lm_v1`:

1. **Address space:** random Gaussian matrix `A: R^d_in → R^d_addr`,
   d_in = 256·128 = 32K (one-hot of 128 bytes), d_addr = 512. Binarize via
   sign. **Frozen.**
2. **Hard locations:** M = 1 M random ±1 binary vectors in R^d_addr,
   sampled iid uniform. **Frozen.**
3. **Content matrix:** C ∈ R^{M × 256} initially zero.
4. **Write phase** (single pass over 5 M training tokens):
   - For each token: address = sign(window features @ A^T).
   - similarity = address @ hard_locations^T (each row in {-d_addr,
     ..., d_addr}).
   - mask = similarity > threshold (calibrated so avg 10 locations
     active).
   - C[mask] += onehot(byte_{t+1}).
5. **Read (`predict`):**
   - same mask computation
   - logits = mean of C[mask] (or sum, then softmax)
   - softmax → 256-d distribution
6. **Sweep:** M ∈ {64K, 256K, 1M}, d_addr ∈ {256, 512, 1024}, r threshold
   (sweep so active count ∈ {5, 10, 50, 100}), ctx_len ∈ {64, 128, 256}.

Streaming wrapper: just maintain rolling window of last 128 bytes,
recompute address on each `observe()`.

## Memory-movement analysis

Train: 5 M tokens × (single mask compute + write) where mask compute is
(1, d_addr) × (d_addr, M) = O(M·d_addr) per token. At M = 10^6, d_addr =
512: 5 × 10^8 flops per token × 5 × 10^6 tokens = 2.5 × 10^15 flops total.
That's a *lot* — likely impractical without batching: with batch B = 1024,
this becomes 5 × 10^11 flops in chunks; total ~3 × 10^14 flops, ~ 1 s on
A100 fp16. HBM traffic dominated by C reads/writes: each token touches
~10 of M = 10^6 rows × 256 cols × 4 bytes = ~10 KB. Over 5M tokens that's
50 GB HBM, sub-second on A100.

So with batched write, train pass ~1 s. Inference is similar per byte but
60 K bytes only → ~10 ms.

Total submission: < 30 s wall-clock, < 3 kJ energy. **Cheapest gradient-free
submission in the portfolio.**

## References

- Kanerva, *Sparse Distributed Memory*, MIT Press 1988 (book).
- Bricken & Pehlevan, NeurIPS 2021: <https://arxiv.org/abs/2111.05498>
- Memorizing Transformers, ICLR 2022: <https://arxiv.org/abs/2203.08913>
- Sparse distributed memory Wikipedia summary:
  <https://en.wikipedia.org/wiki/Sparse_distributed_memory>
