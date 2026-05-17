# Forward-Forward — Exhaustive Investigation Plan

**Goal.** Find a Forward-Forward-family approach that **reaches val char-acc ≥ 0.70 on WikiText-103 char-level within 300 s on A100-80GB**, ranked by training-energy (lower wins). If no FF approach can clear that bar after this investigation, the report justifies a confident discard backed by a wide search.

**Why this is a real undertaking, not a smoke test.** Plain Hinton FF with an FC backbone gave 0.279 — far from 0.70. Closing a 0.42-accuracy gap from a non-backprop method is hard. The pass-1/pass-2 work only touched one corner of the FF design space. An exhaustive search means systematically covering the *axes* of FF, not running a few more random hyperparameter sweeps over the same architecture.

---

## 1. The well-defined task

| | |
|---|---|
| **Dataset** | WikiText-103 raw, byte-level (256-vocab). Use `wikitext.py::load_wikitext103` for train/valid/test. |
| **Hardware** | Modal A100-80GB PCIe (pinned via `task.py::INSTANCE_TYPE`). |
| **Train budget** | ≤ 300 s wall-clock inside `train()`, enforced by `wall_clock_guard`. Eval is not charged. |
| **Eval** | First 60 000 chars of val split, greedy-argmax char-accuracy, via `evaluate()`. |
| **Floor (rule 5)** | val char-acc ≥ 0.70. Submissions below this are DQ. |
| **Ranking metric** | Training energy in joules (NVML, idle-subtracted). Lower wins. |
| **Disallowed** | End-to-end backprop across layers. Pretrained weights of any kind. |
| **Allowed** | Local-per-layer gradient (the FF rule itself). Closed-form linear readouts (ridge etc., these have zero gradient). Embedding tables learned by FF. Any architecture so long as the training rule is FF-family. |
| **Determinism** | Honour `SEED` env var across init, sampling, and any stochastic ops. |

**Success criterion for the investigation.** Either:
- **PASS**: a submission, reproducible from this repo, reaches val char-acc ≥ 0.70 within 300 s. We then report energy and rank against the modded_nanogpt baseline (51,704 J).
- **JUSTIFIED DISCARD**: a structured report covering ≥ 6 distinct FF rule variants × ≥ 4 backbones × controlled diagnostics shows no path within an order of magnitude of the floor. The report names what *was* tested and what was deliberately left out.

The investigation deliverable is a single artifact — not just the winning submission, but a survey-grade write-up that lets future researchers know which FF directions are dead-ends on this benchmark.

---

## 2. What Forward-Forward actually is

### Core algorithm (Hinton 2022, [arxiv:2212.13345](https://arxiv.org/abs/2212.13345))

Replace end-to-end backprop with a **per-layer local objective**:

1. Define a **goodness** `G_l(x) = sum_i a_l[i]^2` on each layer's activations.
2. For each layer, train weights to **push `G_l` above a threshold θ on positive samples** and **below θ on negative samples** via a logistic loss: `L_l = softplus(θ − G_pos) + softplus(G_neg − θ)`.
3. The input to layer `l` is `detach(LayerNorm(a_{l-1}))`. Gradient through layer `l` touches only `W_l`. No cross-layer flow.
4. L2-normalise between layers (strip magnitude) so goodness can't leak trivially up the stack.
5. At inference, score candidates by their goodness summed across layers, or fit a separate readout.

**What this changes vs. backprop.** Constant memory per layer. No global graph. Layers can be trained in parallel or sequentially. The "credit assignment" problem is replaced with a contrastive-discrimination problem at each layer. The cost is that each layer only sees its local objective — there's no end-to-end gradient telling layer 1 to learn features useful for layer 5.

### The FF family — taxonomy by axis

| Axis | Choices (✓ = tested in survey passes 1/2) |
|---|---|
| **A. Goodness function** | sum-of-squares ✓ · mean · log-sum-exp · cosine · L1 · custom learned |
| **B. Loss formulation** | logistic on (G − θ) ✓ · margin · contrastive InfoNCE · sigmoid CE · energy |
| **C. Negative generation** | external unigram ✓ · cross-class concat (Hinton MNIST) · hard self-readout ✓ · top-down generative (Hinton §4) · within-batch contrastive · augmentation-based |
| **D. Layer-1 treatment** | frozen random projection ✓ · trained · learned embedding · conv stem · hashed n-gram |
| **E. Backbone** | FC stack ✓ · causal Conv-1D · dilated conv · MLP-Mixer · recurrent (RFF) · transformer-style block (local FF training inside) · hybrid |
| **F. Layer-norm style** | L2 between layers ✓ · LayerNorm · BatchNorm-free · sphere-normalisation |
| **G. Schedule** | round-robin per-step ✓ · greedy layer-wise (CaFo style) · parallel synchronous · curriculum (short→long context) |
| **H. Predictor / readout** | goodness-softmax over candidates ✓ · linear ridge on concat features ✓ · per-layer ridge ensemble · MLP head (caveat: head-only BP is debatable but allowed) · kernel ridge |
| **I. Context length** | K=24 ✓ · K=64 · K=128 · K=256+ |
| **J. Variants** | plain Hinton ✓ · SymBa-FF · CaFo (Cascaded Forward) · Contrastive FF / FF-Aug · PEPITA · Mono-Forward · Generative FF |
| **K. Capacity** | width 384–512 ✓ · 1024 · 2048 · 4096 |

**Survey-pass coverage.** Passes 1 and 2 fixed A=sum-of-sq, B=logistic, C=unigram (pass 1) + hard-self (pass 2), D=frozen-random, E=FC, F=L2, G=round-robin, H=goodness then ridge, I=24, J=plain, K=384–512. **One slice of a 10-dimensional grid.**

### Named variants from the published literature (verify citations in Phase 1)

The following are reported variants the user should expect a phase-1 literature scan to confirm and characterise. Treat the descriptions as my best recollection; Phase 1 produces the verified taxonomy.

- **SymBa-FF** — adds a symmetric "backward" pass that re-uses the same goodness rule but with the negative-positive pair swapped. Claims of near-BP performance on CIFAR-10.
- **CaFo (Cascaded Forward)** — sequential layer training where each layer's "label" is the previous layer's output via a small KL-divergence head. Layer-wise training, not round-robin.
- **Contrastive FF / FF-CL** — InfoNCE-style within-batch negatives instead of external negatives.
- **PEPITA** — sibling of FF (not strictly FF). Uses an error-modulated second forward pass. Listed because it's a useful comparison point in the "local rules" family.
- **Generative-negative FF** — Hinton's §4 refinement. The model generates its own negatives via top-down feedback.
- **Forward-Forward for NLP** ([arxiv:2307.04205](https://arxiv.org/abs/2307.04205), Gandhi & Gala) — applied FF to GLUE/sentiment, mixed-to-weak results.

Phase 1 deliverable expands this list with verified citations, reported gains, and benchmark-portability assessment.

---

## 3. Investigation strategy: phased screening, not flat sweep

A flat grid over the axes above is ~10⁵ combinations — infeasible. We use **phased screening**: each phase produces a winner that fixes one axis, narrowing the search space for the next phase. Each phase has explicit kill criteria.

```
Phase 1 (no Modal cost):    Literature → verified FF taxonomy
Phase 2 (5 runs):           Diagnostic baselines on pass-2 setup
                            ↓
Phase 3 (6–10 runs):        FF rule variants screened side-by-side
                            ↓                       (kill: any variant < pass-2 by ≥ 0.02)
Phase 4 (8–12 runs):        Backbone variants on the best rule from Phase 3
                            ↓                       (kill: FC remains best)
Phase 5 (6–8 runs):         Negative-quality + schedule on the winner of Phases 3+4
                            ↓
Phase 6 (5–8 runs):         Readout optimisation on the best representation
                            ↓
Phase 7 (10–15 runs):       Scale & combine — push the best stack to the budget,
                            multi-seed, hyperparameter polish
                            ↓
Phase 8 (no Modal cost):    Final report (PASS or JUSTIFIED DISCARD)
```

**Total Modal budget**: ~50–70 runs, ~$30–45 cost, ~3–5 h wall if parallelised at 5 concurrent. Single-stream wall ~25–40 h.

**Pruning logic.** A variant only advances to the next phase if it either (a) clears a numerical bar set against the prior-phase winner, or (b) gives a *mechanistic insight* (e.g. layer-wise probe pattern) that opens a new branch. This is a tournament: most variants get killed, a few survive.

**Parallelism note.** Modal supports concurrent A100 runs. Within a phase, all experiments are independent and run in parallel.

---

## 4. Phase 1 — Literature deep-dive (no Modal cost)

**Deliverable.** `.survey/FF_LITERATURE.md` — verified taxonomy of FF variants from 2022 to now, each entry with:

- Variant name + canonical citation (arxiv ID + first-author surname)
- One-paragraph algorithm summary
- Reported headline result and benchmark
- Difference from plain Hinton FF (which axis it modifies)
- Code availability (github repo if any)
- Estimated portability to byte-level LM at 300 s budget (subjective, justified)

**Search scope.**
- Google Scholar + arxiv search: `"forward-forward" Hinton`, `forward-forward language`, `forward-forward char`, `local learning rule language model`, `backprop-free language model`.
- Citation graph: papers citing Hinton 2022 (~200 as of 2026; not all relevant).
- OpenReview ICLR 2024/2025 backprop-free / local-learning tracks.
- Pull from related Hebbian/local-learning families when they're cited by FF papers.

**Time estimate.** 4–8 hours of focused literature work, done by a subagent with web access. Output is one document, not running experiments.

**Bounded scope.** Limit to 20 candidate variants in the taxonomy. Rank them by portability score; Phase 3 picks the top 6–8 to actually test.

**Kill criterion.** None — this phase always produces output. The kill happens in Phase 3.

---

## 5. Phase 2 — Diagnostic baselines (5 Modal runs)

**Purpose.** Before testing variants, we need the pass-2 setup's mechanistic profile: is the ridge readout doing all the work? Is depth helping? Is width the bottleneck? These results frame how we interpret Phase 3+ outcomes.

All Phase-2 runs use pass-2 hyperparameters as the baseline (5 × 384 FC, K=24, θ=2.0, 14k steps, hard-neg refresh every 500). One axis changes per run.

| ID | Variation | What it measures |
|---|---|---|
| **P2-A** | All 5 layers frozen random Gaussian, no FF. Same ridge on concat(LN(a_2..a_5)). | The "random projection" floor. If FF + ridge ≈ random + ridge, FF is not adding representational value. **The single most important missing measurement from the prior survey.** |
| **P2-B** | Standard pass-2 FF, but fit **per-layer** ridge readouts (one per layer 1..5) and report all five accuracies on a 20K-char diagnostic val chunk. | Is FF building hierarchy? Layer-1 (random) acc vs layer-5 acc. |
| **P2-C** | Standard pass-2 FF at width **1024** with step count cut to fit 250 s. | Capacity scaling slope. |
| **P2-D** | Standard pass-2 FF with K=**64** context. | Context-length scaling. |
| **P2-E** | Standard pass-2 FF, but **bigram one-hot** input encoding (last byte + last bigram concat) instead of K char one-hot. | Whether the input encoding bottlenecks FF. |

**Outputs.** Each run produces (val char-acc, energy, mechanistic notes). The five numbers together give a 2D scaling picture: depth-usefulness × width-response + context-response + input-response.

**Kill criteria.** None for Phase 2 — these are diagnostics. But the readings shape Phase 3's design priors:
- If P2-A ≥ 0.265: every Phase 3 variant must beat random projection by ≥ 0.04, or it's dead before further investigation.
- If P2-B shows monotone hierarchy: backbone depth matters; Phase 4 prioritises deep variants.
- If P2-C slope is steep: width matters; Phase 7 prioritises width over other axes.

---

## 6. Phase 3 — FF rule variants (6–10 runs, parallel)

**Purpose.** Replace the FF rule itself, holding architecture fixed at pass-2's FC backbone (so the rule is the only axis varying).

**Architecture (fixed).** 5 × 384 FC, K=24 context, ridge readout on concat(LN(a_2..a_5)). All non-rule axes match pass 2.

**Candidate variants (top picks from Phase 1; final list set after Phase 1).**

| ID | Rule | Key mechanism |
|---|---|---|
| **P3-1** | Plain Hinton FF (pass-2 control) | Sum-of-squares goodness, logistic loss, external negatives |
| **P3-2** | SymBa-FF | Symmetric pass with swapped pos/neg |
| **P3-3** | CaFo | Sequential layer training with KL head per layer |
| **P3-4** | Contrastive FF | InfoNCE within-batch negatives, no external sampler |
| **P3-5** | Generative-negative FF | Top-down feedback negatives (Hinton §4) |
| **P3-6** | Cosine-goodness FF | Replace `sum(a^2)` with `cos(a, c_l)` where c_l is a learned class-conditional vector |
| **P3-7** | Margin-loss FF | Replace logistic with hinge margin |
| **P3-8** | LSE-goodness FF | Replace sum-of-squares with log-sum-exp |

Run 6–8 of these (final list from Phase 1).

**Pass criterion.** Variant advances to Phase 4 if its val char-acc beats pass-2's 0.2792 by ≥ 0.02 **OR** if its layer-wise probe (re-run B-style on the variant) shows substantially different hierarchy than pass 2 (e.g. monotone increase where pass 2 plateaus).

**Kill criterion.** Variant with val char-acc < 0.27 AND no qualitative mechanistic improvement is killed.

**Expected survivors.** 2–3 variants. Expectation-managed: if all 8 variants land at 0.25–0.29, the conclusion is that **rule choice doesn't matter much** on byte LM — and Phase 4's architecture exploration becomes the load-bearing axis.

---

## 7. Phase 4 — Backbone architecture variants (8–12 runs, parallel)

**Purpose.** Replace the FC backbone with architectures that have stronger inductive biases for sequence data.

**FF rule (fixed).** Winner from Phase 3 (or plain FF if Phase 3 didn't surface a winner).

**Backbones to test.**

| ID | Backbone | Why |
|---|---|---|
| **P4-1** | FC stack (control, matches Phase 3 winner) | Baseline |
| **P4-2** | Causal Conv-1D stack, 5 layers, kernel 5, width 256 | Locality + parameter sharing; bytes have strong local structure |
| **P4-3** | Dilated causal conv, dilations [1, 2, 4, 8, 16], width 256 | Wider receptive field at moderate cost |
| **P4-4** | MLP-Mixer-style: alternate token-mix + channel-mix MLPs | Cheap attention substitute, FF rule applies cleanly per block |
| **P4-5** | Recurrent FF — GRU-style cells trained per-layer by FF | Long context via state, no attention |
| **P4-6** | Convolutional stem + FC FF head (hybrid) | Tries to combine conv inductive bias with the simple FC FF rule |
| **P4-7** | Wide-FF (control): width 1536 FC at fixed step count | Capacity-only comparison vs structured architectures |
| **P4-8** | Causal conv with K=128 context | Compound with longer context (the K=24 limit of pass 1/2 may have been the real bottleneck) |

**Pass criterion.** Backbone advances to Phase 5 if val char-acc beats Phase-3 winner by ≥ 0.03 **OR** if it pairs well with a specific Phase 5 readout improvement (e.g. conv features + per-layer ensemble).

**Kill criterion.** Backbone fails to clear Phase-3 winner by 0.01 → discarded.

**Expected survivors.** 1–3 backbones. Strong prior: causal conv with reasonable receptive field will outperform FC on bytes.

---

## 8. Phase 5 — Negative quality & training schedule (6–8 runs)

**Purpose.** On the (rule × backbone) winner, optimise the parts that aren't the rule or the architecture: how negatives are generated and how layers are scheduled.

**Variants.**

| ID | Variation |
|---|---|
| **P5-1** | External unigram negatives (Phase 4 winner setup, control) |
| **P5-2** | Cross-class one-hot concat negatives (Hinton MNIST style) |
| **P5-3** | Hard self-readout negatives, top-K=5, refresh every 500 steps (pass-2 style) |
| **P5-4** | Generative top-down feedback negatives |
| **P5-5** | Within-batch contrastive negatives |
| **P5-6** | Mixed: 50% unigram + 50% hard-self (pass-2 style) — control |
| **P5-7** | Round-robin schedule (pass-2 control) |
| **P5-8** | Sequential layer-wise (greedy, CaFo-style) — train layer 1 to convergence, freeze, train layer 2, ... |

Run P5-1..5 (negatives) on the Phase-4 winner, then P5-7..8 (schedule) on the best negative-strategy from P5-1..5.

**Pass criterion.** ≥ 0.02 absolute lift over Phase-4 winner.

**Kill criterion.** No lift over Phase 4 → keep Phase 4 winner config, skip P5 additions in Phase 7.

---

## 9. Phase 6 — Readout optimisation (5–8 runs)

**Purpose.** Train the best representation from Phases 3+4+5, then experiment with the readout layer.

**Variants.**

| ID | Readout |
|---|---|
| **P6-1** | Linear ridge on concat(LN(a_2..a_L)) (pass-2 control) |
| **P6-2** | Per-layer ridge ensemble: independent W per layer, average logits at eval |
| **P6-3** | Kernel ridge with random Fourier features (RBF approximation) — closed-form, gradient-free |
| **P6-4** | Quadratic feature ridge: include pairwise products a_l ⊙ a_{l+1} |
| **P6-5** | Bayesian linear regression (gives calibrated probs, may matter for argmax ties) |
| **P6-6** | MLP head trained with backprop on top of frozen FF features (2-layer, width 1024, ReLU). **Run as ablation regardless of submission eligibility.** |
| **P6-7** | Conditional Random Field with FF features as emissions |

**Pass criterion (closed-form readouts P6-1..5, P6-7).** ≥ 0.02 absolute lift over P6-1.

**P6-6 (MLP head ablation) — purpose and framing.** This run is an **ablation, not a candidate submission**. It compares closed-form ridge vs. a small MLP head with intra-head backprop on the *same frozen FF features*. The MLP head can extract nonlinear structure the linear readout cannot, so the gap (P6-6 − best of P6-1..5) tells us:
- If gap is small (< 0.02): the FF representation is the bottleneck, not the readout. Closed-form is fine. Strong evidence FF's ceiling is structural.
- If gap is large (≥ 0.05): the FF representation has more signal than closed-form can extract. Then the leaderboard story becomes "FF + closed-form is rate-limited by the readout, not the rule" — interesting finding but doesn't help us pass rule 5 with a pure-FF submission.
- If P6-6 itself reaches 0.70+: the FF features alone (with a learned head) are competitive. Then we explicitly flag this in the report as "FF with a head trained by backprop reaches 0.70 — but the head's backprop means this is not a pure-FF submission per the spirit of the rules."

In either case the ablation produces a load-bearing piece of evidence. Run it.

---

## 10. Phase 7 — Scale & combine (10–15 runs)

**Purpose.** Take the best stack from Phases 3+4+5+6 and push it to the limit of the 300 s budget. This is the phase that might actually crack 0.70.

**Sub-experiments.**

| ID | Variation |
|---|---|
| **P7-1** | Best stack at width 1024 (control = Phase 4/5 baseline) |
| **P7-2** | Best stack at width 2048 (if fits in 300 s) |
| **P7-3** | Best stack at width 1024 + K=128 context |
| **P7-4** | Best stack at width 1024 + K=128 context + curriculum (short → long context over training) |
| **P7-5** | Best stack with FP16 training to halve memory and allow wider models |
| **P7-6** | Best stack + 3 seeds for variance estimation |
| **P7-7** | Best stack + warm-start ridge from earlier checkpoint, refit at end |
| **P7-8** | Best stack but trained at width 1024 with **early stopping** on val char-acc (if budget allows) |
| **P7-9** | Best stack but with **doubled training steps** if width allows (within 300 s) |
| **P7-10** | Best stack + **L2-LN replaced with cosine output normalisation** (free axis-F experiment) |

**Pass criterion.** Hit val char-acc ≥ 0.70. If yes → declare PASS, file the submission, generate report.

**Kill criterion.** Best Phase-7 result < 0.45 → declare JUSTIFIED DISCARD. The report says "FF, across 6 rules × 4 backbones × ... was tested; the ceiling is X."

**Mid-criterion.** Phase-7 best in [0.45, 0.69] → declare a **soft-discard with footnotes**: FF reaches X on this benchmark, far short of 0.70 but well above prior pass-2 0.279. The report names which combination came closest.

---

## 11. Phase 8 — Final report (no Modal cost)

**Deliverable.** `.survey/FF_FINAL_REPORT.md` with:

- Outcome banner: **PASS**, **SOFT-DISCARD (X.XX ceiling)**, or **JUSTIFIED DISCARD**.
- Full table of all ~50–70 runs: rule × backbone × negatives × readout → (val_acc, energy, wall_time, notes).
- Phase-by-phase narrative of what got killed and why.
- The winning stack's mechanism explanation (if PASS).
- Mechanistic findings: layer-wise hierarchy, capacity slope, context response, rule sensitivity.
- Explicit list of **what we did not test** (with reasons).
- Recommendation for future research-group action.

---

## 12. Budget, parallelism, and scheduling

### Modal cost estimate

| Phase | Runs | Cost @ $0.62/run | Wall (sequential) | Wall (5-parallel) |
|---|---:|---:|---:|---:|
| 2 — Diagnostics | 5 | $3 | 25 min | 5 min |
| 3 — FF rules | 8 | $5 | 40 min | 10 min |
| 4 — Backbones | 10 | $6 | 50 min | 12 min |
| 5 — Negatives + schedule | 7 | $4 | 35 min | 8 min |
| 6 — Readouts | 6 | $4 | 30 min | 7 min |
| 7 — Scale & combine | 12 | $7 | 60 min | 15 min |
| **Total Modal** | **48** | **~$30** | **~4 h** | **~1 h** |
| Phase 1 (lit review) | 0 Modal | 0 | 4–8 h focused | n/a |
| Phase 8 (report) | 0 Modal | 0 | 2–4 h focused | n/a |
| **Investigation total** | **48 runs** | **~$30** | **~10–16 h** | **~7–11 h** |

### Parallelism

- Phase 3 + 4 + 5 + 6 internal runs are mutually independent. Dispatch 5+ at once.
- Phases must be sequential because each phase's winner feeds the next.
- The 5-parallel column assumes a Modal account that allows 5 concurrent A100-80GB jobs. Default Modal usually allows this; verify before scheduling.

### Time-boxing

- Each Modal run ≤ 5 min wall (300 s train + ~60–120 s eval + overhead).
- Phase 1 time-boxed to **8 h** of literature work; if exceeded, ship with whatever taxonomy we have.
- Phase 7 time-boxed to **3 h** of Modal time (~36 runs max); past that, the answer is "doesn't fit this budget."

---

## 13. What this plan does and does not cover

### Covered (explicit)
- The 10 main FF axes (A–K in §2) get at least one experimental variation each.
- All 6+ published FF variants we can verify in Phase 1 get tested in Phase 3.
- At least 4 architecture backbones tested in Phase 4.
- At least 4 negative-generation strategies tested in Phase 5.
- At least 5 readout strategies tested in Phase 6.
- Capacity (width) and context (K) scaling characterised in Phases 2 and 7.

### Deliberately NOT covered
- **End-to-end backprop** anywhere except within the readout head (P6-6, flagged for user decision). Excluding this is the whole point of an FF investigation.
- **Pretrained anything.** Competition rule 1.
- **Architectures outside the FF rule's compatibility.** E.g. full transformer with cross-attention — the local FF rule doesn't have a clean per-layer adaptation for attention queries-keys-values, so this is research-level work outside the scope of "find a working FF approach."
- **Subword tokenisation.** Competition rule 2 (use the wikitext char/byte streams).
- **Budgets beyond 300 s.** The benchmark fixes this.
- **Hyperparameter random search.** We do **targeted** sweeps along named axes; random search over 100s of HPs is excluded by the screening-tournament design.

### Open questions the user needs to decide before Phase 3 dispatches

1. ~~Does an MLP readout head count as backprop violation?~~ **DECIDED 2026-05-17:** Run P6-6 as an **ablation**, not a candidate submission. The diagnostic gap (closed-form vs. MLP head on same FF features) is informative regardless of submission eligibility.
2. ~~Should Phase 1 be done by a subagent?~~ **DECIDED 2026-05-17:** Yes — dispatched as background subagent. **COMPLETED 2026-05-17:** 20 variants verified in `LITERATURE.md`.
3. ~~Is the $30 Modal budget acceptable?~~ **DECIDED 2026-05-17:** Yes — full budget approved.
4. ~~What's the time horizon?~~ **DECIDED 2026-05-17:** Single sitting; check-in with user between phases.
5. ~~Phase 3 variant list~~ **DECIDED 2026-05-17:** Swap P3-6 (cosine-goodness) for **Mono-Forward** (per-layer cross-entropy + linear head, arxiv:2501.09238). HFF in the Phase 3 contingency list already covers the cosine-prototype axis if needed.

---

## 14. File layout for the investigation

```
research/forward-forward-deep/
├── PLAN.md                          ← this file
├── LITERATURE.md                    ← Phase 1 output (20 verified variants)
├── FINAL_REPORT.md                  ← Phase 8 output (created at end)
└── runs/                            ← Phase 2..7 artifacts
    ├── phase2/
    │   ├── P2-A_random_projection/
    │   │   ├── design.md
    │   │   ├── submission.py
    │   │   ├── result.json
    │   │   └── run.log
    │   ├── P2-B_layerwise_probe/
    │   ├── P2-C_width_1024/
    │   ├── P2-D_context_64/
    │   └── P2-E_bigram_input/
    ├── phase3/
    │   ├── P3-1_plain_hinton/
    │   ├── P3-2_symba_ff/
    │   └── ... (one dir per variant)
    ├── phase4/ ... phase5/ ... phase6/ ... phase7/
```

Each run dir has: `design.md` (spec, written before submission), `submission.py` (the actual file), `result.json` (harness output), `run.log` (Modal stdout/stderr).

---

## 15. Next concrete step

The next action depends on your answers to §13's five open questions. The likely path:

1. You answer §13 (5 min).
2. I dispatch a subagent for Phase 1 literature work (4–8 h, runs while we plan Phases 2–7 specs).
3. In parallel, I draft Phase 2 specs (5 design.md files) for your review.
4. On Phase 1 completion: confirm Phase 3 variant list, write Phase 3 specs.
5. Dispatch Phase 2 runs (parallel). 5 runs, ~5 min wall.
6. Read Phase 2 results, adjust Phase 3+ priors.
7. Continue through phases.

If you want to compress: skip Phase 1's full literature review and let me use my existing knowledge of 4–5 named variants as the Phase 3 list. That cuts 4–8 h off the plan and adds maybe 5% risk of missing a relevant variant.

The investigation as designed costs ~$30 Modal credit and ~10–16 h of focused work. Whether it succeeds in finding a 0.70-clearing FF is genuinely unknown — the prior is moderate at best. Whether it produces a defensible discard report is high-confidence yes.
