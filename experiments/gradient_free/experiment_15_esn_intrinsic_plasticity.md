# Experiment 15: ESN + Triesch Intrinsic Plasticity + Ridge

## Hypothesis
The Triesch (2005) intrinsic-plasticity (IP) rule adapts each reservoir neuron's gain and bias so that its output activation matches a target exponential distribution — locally, online, and without any gradient signal from the readout. This unsupervised adaptation *maximizes per-neuron information transmission* and reliably improves ESN downstream performance by 10–30% on time-series tasks (Steil 2007, Schrauwen et al. 2008). Applied to byte-level WikiText with a 16384-unit reservoir + IP + closed-form ridge readout, we expect val char-acc 0.50–0.65, **strictly better than ESN pass 1's 0.347 at comparable energy**. IP is **the highest-leverage gradient-free way to make a reservoir's state diversity match its capacity**, and it has not been tried in this repo.

## Motivation
ESN pass 1 (0.347) and pass 2 (DQ) both used random reservoirs sampled once and frozen. Reservoir state activations in such networks are typically concentrated near 0 (tanh nonlinearity, modest input drive) → most neurons carry redundant information. The Triesch IP rule fixes this by adapting per-neuron parameters (a, b) so the output `y = tanh(a·net + b)` follows a target maximum-entropy exponential distribution with mean μ. This is a **local, unsupervised, gradient-free** adaptation (the parameter update uses only the neuron's own activation and the target distribution) — it satisfies the no-NN, no-backprop constraint by construction.

## Method
**Reservoir** (vanilla single-layer ESN, deliberately simpler than experiment_14): N=16384, density 0.005 (sparser than pass 1/2 to keep matvec cheap), ρ=0.95, leak a=0.3, K=4-byte one-hot input (D_in=1024). W_in dense, W_res sparse CSR. All fp32.

**Intrinsic Plasticity adaptation phase** (Triesch 2005 rule for sigmoid/tanh neurons, adapted from Schrauwen et al. 2008):
```
y = tanh(a · net + b)        # net = W_res · x + W_in · u  (pre-activation)
db = -eta_IP · (-μ + (2 + 1/μ)·y - y²/μ)
da = eta_IP / a + db · net
```
where μ=0.1 is the target mean activation, eta_IP=5e-4. Apply this update for 500K bytes of unsupervised pre-training (no labels used). Per-neuron parameters (a, b) ∈ R^N each, initialized to (1, 0).

**Streaming feature phase** (with frozen IP-adapted (a, b)): run 8 M training bytes through the IP-adapted reservoir, accumulate (Ψ_t = concat(x_t, u_t)) ∈ R^(N + D_in) = R^17408 row-by-row in chunks of 50K.

**Ridge solve**: W_out = (Ψ^T Ψ + λI)^{-1} Ψ^T Y_onehot in fp64.

## Memory-Movement Analysis
- **IP phase**: 500K steps × (sparse matvec ~0.1 ms + tanh + per-neuron-param update). Per step the IP update is N=16384 elementwise ops ≈ negligible. Total: ~80 s if Python loop overhead is the bottleneck (consistent with pass 1's ~7 K steps/s). Mitigate by batching IP-phase as B=64 chunks, same as experiment_14 — ~10 s for IP phase.
- **Streaming phase**: 8 M bytes / B=64 = 125 K steps, ~1 ms/step at density 0.005 → 125 s.
- **Ψ^T Ψ accumulation**: 8 M rows of 17408 fp32. Chunked: 80 blocks of (100K, 17408) → Phi^T Phi per block ~ 60 GFLOPs · 80 ≈ 5 TFLOPs ≈ 25 s on A100 in fp32 then cast.
- **Cholesky (17408, 17408) fp64**: ~30 s (pass 2 spec).
- **Total**: 10 + 125 + 25 + 30 + 30 (slack) = 220 s. Under budget.
- **Arithmetic intensity**: sparse matvec at density 0.005, batched B=64: N=16384, 2·N·N·0.005 = 2.7 MFLOPs/step input, density × matrix-size dominated. Sparse path on A100 is ~1 TFLOP. Step time ≈ 3 µs of GPU; the rest is Python.

## Setup
- N=16384, density 0.005, leak 0.3, ρ=0.95, K=4-byte input → D_in=1024.
- IP target: exponential with mean μ=0.1; eta_IP=5e-4; IP phase = 500K bytes.
- Streaming N_train = 8 M bytes (post-IP), B=64 chunks, washout=1000 bytes/chunk.
- ψ_t = concat(x_t, u_t) ∈ R^17408 (post-IP x_t already adapted via the frozen (a, b)).
- Ridge λ = 1e-3 (relative).
- fp32 states, fp64 normal equations. Seed = `int(os.environ.get("SEED", 0))`.
- Compare against: ESN pass 1 (0.347 / 29.8 kJ); ESN pass 2 (DQ); deep_esn_multiscale (experiment_14, pending).

## Procedure
1. `cp -r submissions/rff_ridge_v1 submissions/esn_ip_d16k`.
2. Sample W_in, W_res; rescale ρ via 30-step power iter. Initialize a=1, b=0.
3. IP phase: stream 500K bytes through reservoir, update (a, b) per-neuron per-step using Triesch rule. *No labels used.*
4. Freeze (a, b). Streaming feature phase: 8 M bytes, B=64 chunks; build XtX, XtY chunked.
5. Cholesky-solve W_out in fp64. Cast to fp32.
6. `CharModel`: maintain reservoir state x and K-byte history ring; `predict()` returns `softmax(W_out · psi)`; `observe(c)` runs the leaky-tanh update with the IP-adapted (a, b).
7. `python submit.py submissions/esn_ip_d16k --yes`.

## Success Criteria
- **Capability**: val char-acc ≥ 0.50, **strictly above ESN pass 1's 0.347**. Validates IP as the missing piece.
- **Strong**: val char-acc ≥ 0.62, energy ≤ 30 kJ.
- **Surprise**: val char-acc ≥ 0.70 — clears floor. First IP-ESN result on byte LM in literature.
- **Refutation**: val char-acc ≤ 0.40 — IP did not unlock useful state diversity; reservoir-computing direction is mechanically capped on bytes. Move on.

## Failure Modes & Diagnostics
- **IP drives (a, b) divergent** (a → ∞ if eta_IP too high): clip a ∈ [0.1, 10], b ∈ [-5, 5] per step. Diagnostic: log percentiles of a, b at IP phase end. Healthy range: a ~ [0.5, 2], b ~ [-2, 2].
- **IP phase mean activation ≠ μ=0.1**: log empirical mean(y) per layer at end of IP. If far from 0.1 (>0.3), increase IP duration to 1M bytes or eta_IP to 1e-3.
- **Numerical conditioning of XtX at 17408 dim**: same risk as ESN pass 2. fp64 + relative ridge should fix it; if Cholesky fails use `torch.linalg.lstsq`.
- **Sparse-batched matvec slower than projected** (the pass-2 killer): density 0.005 (10× sparser than pass 2's 0.02) gives more headroom. If step time > 2 ms at B=64, drop B=32 and N_train=4M.
- **IP makes acc *worse* than no-IP** (counter-evidence to Triesch 2007): document and check the rule implementation against Schrauwen 2008 eq. 7. The rule is brittle to sign errors.

## Estimated Cost
1 Modal A100-80GB run × ~4 min wall ≈ $0.08. A no-IP ablation (same hyperparams, IP phase skipped) to isolate the IP contribution: another $0.08.

## References
- Triesch 2005, "A Gradient Rule for the Plasticity of a Neuron's Intrinsic Excitability", ICANN 2005.
- Schrauwen, Wardermann, Verstraeten, Steil, Stroobandt 2008, "Improving reservoirs using intrinsic plasticity", Neurocomputing 71(7–9):1159–1171.
- Steil 2007, "Online reservoir adaptation by intrinsic plasticity for backpropagation–decorrelation and echo state learning", Neural Networks 20(3):353–364.
- Lukoševičius 2012, "A Practical Guide to Applying Echo State Networks", Neural Networks: Tricks of the Trade.
- `research/gradfree-survey/designs/method_esn-ridge-readout_pass_2.md` — pass 2 failure analysis (sparse-batched matvec throughput cap, informs density choice here).
