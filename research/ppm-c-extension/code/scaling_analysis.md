# Horizontal scaling comparison: PPM and its alternatives

Two senses of "horizontally scale" matter here, so let me separate them:

- **Scale-out parallelism**: doubling workers / nodes → close-to-doubling throughput.
- **Per-byte parallelism**: within a single byte's update, can work fan out across multiple cores/lanes?

Both depend on a structural property: **does the model's state update commute?** Count-based methods have additive state (counts merge by sum), so they're trivially parallelizable; gradient-based methods don't and have to fall back on classical SGD-scaling techniques.

## Comparison

| Approach | Data-parallel (shard stream + merge) | Per-byte parallel | Model-parallel (shard state) | Communication cost on merge | Best hardware fit |
|---|---|---|---|---|---|
| **Plain PPMd-D** | Yes — counts are additive | No — K sequential probes per byte | Hard — trie is one global structure | High — must merge gigabyte-scale sparse tries | Multi-core CPU, single box |
| **Context mixing (PAQ/cmix)** | Yes — each per-order model independently | **Yes** — K orders evaluate independently per byte | **Natural** — one node per (order, shard) | Medium — K independent smaller-trie merges, themselves parallelizable | Multi-socket NUMA / small cluster |
| **Hash-based feature sharing** | Yes | No (same as PPM) | **Yes** — shard by hash range | **Low** — bounded by fixed hash-table size, fits a single all-reduce | Cluster / distributed memory |
| **Embedding-then-PPM** | Yes for PPM stage; quantizer training is its own story | After quantizer: same as PPM | Yes in code space (which is much smaller than raw context space) | Low — code-space trie is tiny | Two-stage: GPU for quantizer, CPU/GPU for PPM |
| **Small neural readout over PPM features** | Standard sync-SGD scaling on the head; PPM stage scales as PPM | Yes within batch | Yes — standard NN parallelism | Gradient all-reduce — well-trodden | GPU |

## Where the differences come from

1. **Additivity of state is the real lever.** PPM, hash-PPM, context mixing, and embedding-PPM all have `state(A ∪ B) = state(A) + state(B)`. That means **infinite data parallelism with one all-reduce at the end** — no gradient sync, no learning-rate tuning, no mini-batch effects. This is structurally a stronger scaling story than any SGD-based method. The neural-readout hybrid loses this nice property for the head, but the PPM feature-extraction stage keeps it.

2. **Communication cost on merge is the practical bottleneck for scale-out.** This is where the approaches diverge sharply:
   - **PPMd-D**: at K=7 our trie was ~2 GB sparse. Merging two such tries across the network is brutal. Doable but ugly.
   - **Hash-sharing**: merge is `O(hash_table_size)` regardless of data volume. This is what makes hash-sharing the cluster-friendly choice.
   - **Context mixing**: merge cost is `K × per-order-trie`, but the K orders merge in parallel — so wall-time merge is bounded by the *largest* single order, not the sum. And lower-order tries are tiny.
   - **Embedding-PPM**: tiny merges (code space is much smaller than raw context space).

3. **Per-byte parallelism only context-mixing gets for free.** PPM's K context updates are sequential because they walk a node chain. Context mixing breaks that: at each byte, K *independent* models each run their own probe; you can map them to K cores or K SIMD lanes. This converts a memory-latency-bound workload (sequential probes hitting different cache lines) into a memory-bandwidth-bound one (K parallel probes amortizing latency).

4. **The neural readout is the odd one out.** The head needs sync-SGD scaling (well-understood: scales well to ~100s of GPUs with good batch sizes, hits diminishing returns past that). The PPM feature stage scales like any other count-based method. So overall scaling is gated by whichever stage dominates wall time.

## Practical ranking for this benchmark's scale

For ≤ ~32 workers (the realistic budget for a 300 s training experiment):

1. **Context mixing wins on natural parallelism.** K independent models = K-way parallelism for free, on top of data sharding. Per-byte SIMD-friendly. State merges in parallel.
2. **Hash-based sharing wins on communication if you want to go to many nodes.** Fixed-size all-reduce regardless of data scale.
3. **Plain PPMd-D is fine within a single box but ugly across a network.** The trie-merge cost dominates if you try to scale-out.
4. **Embedding-PPM is a two-stage pipeline** — scaling depends on the slowest stage, which is usually quantizer training.
5. **Neural readout** is the only one whose head can't ride the additive-state lever; pays the standard SGD-sync cost.

If you're picking one approach to invest in *because* it scales: **context mixing** is the right answer. It's the only one that gets data-, model-, and per-byte parallelism simultaneously, and the parallelism is structural (independent models) rather than engineered (sharded SGD). That's also why cmix scales linearly across cores on a single box despite being CPU-bound, single-stream code per worker.
