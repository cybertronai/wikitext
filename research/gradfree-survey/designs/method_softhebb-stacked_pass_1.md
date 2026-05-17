# SoftHebb Stacked Conv-1D + Ridge Readout (pass 1)

## 1. Hypothesis
A 4-layer causal 1-D conv stack trained layer-wise by the SoftHebb soft-WTA Hebbian rule on raw bytes will produce a hierarchy of unsupervised temporal feature detectors whose linear projection (closed-form ridge to next-byte one-hot) recovers enough local n-gram structure to reach val char-acc in the 0.55-0.65 range, with an honest stretch goal of 0.70. We expect to learn whether image-domain Hebbian inductive biases transfer to byte streams at all, and whether layer depth helps (vs. saturating after layer 2-3 as receptive field grows).

## 2. Model
- Input: one-hot byte embedding, C_in=256, sequence length T (streamed).
- 4 causal 1-D conv layers, all kernel K=5, channel widths [384, 384, 512, 512], dilations [1, 2, 4, 8] -> cumulative causal receptive field = 1 + 4*(1+2+4+8) = 61 bytes.
- Per-layer nonlinearity: channel-wise softmax-WTA, y[c,t] = softmax_c(u[c,t] / tau). No BN, no residuals, no bias.
- Causal left-padding ((K-1)*dilation zeros on the left of the time axis) so output length matches input length.
- Readout: concat per-timestep activations of layers 2, 3, 4 (384+512+512 = 1408 features) -> linear W_out in R^{256 x 1408} + bias, fit by ridge to next-byte one-hot. No nonlinearity in readout. Logits = W_out @ phi(t) + b at inference, softmax over 256 bytes for predict().

## 3. Training procedure
1. Encode train_text to uint8 tensor on GPU. Take first 100M chars (the corpus is ~530MB train; 100M is a representative slice that fits the budget).
2. For layer l = 1..4:
   - Stream chars in non-overlapping windows of length 8192, batch B=32 (262144 chars per step).
   - Forward through frozen layers 1..l-1 (no grad anywhere).
   - Compute u_l = causal_conv1d(x_{l-1}, W_l, dilation_l); y_l = softmax(u_l / tau, dim=channel).
   - SoftHebb update (no autograd): dW_l[c,:,:] = eta_l * sum_t y_l[c,t] * (x_{l-1}[:, t-K_eff:t+1] - u_l[c,t] * W_l[c,:,:]), averaged over batch and time. Apply in fp32, weights stored fp32.
   - Pass over M_chars_per_layer = 100M chars in one sweep (~382 steps). Anneal eta linearly to 0 over the sweep.
   - Freeze W_l.
3. Feature collection: stream the first 40M chars through all 4 layers, collect phi(t) in R^{1408}. Subsample one in every 8 timesteps -> N = 5M feature rows (fits ~28 GB in fp16 on A100-80GB; we use fp32 accumulators for the Gram).
4. Closed-form ridge: solve (Phi^T Phi + lambda I) W_out^T = Phi^T Y, where Y is next-byte one-hot in {0,1}^{N x 256}. Compute Gram in fp32 by streaming chunks of Phi (no full materialization). One Cholesky solve per output is unnecessary -- solve once for all 256 outputs.
5. Wrap into a CharModel: reset() zeros a ring buffer of the last 61 bytes; observe(c) shifts the buffer and runs one streaming forward (cheap, O(layers * channels * K)); predict() applies W_out and softmaxes.

## 4. Hyperparameters
- Layers L = 4; channels [384, 384, 512, 512]; K = 5; dilations [1, 2, 4, 8].
- tau = 1.0 (paper's default; halved to 0.5 only if WTA collapses to one channel).
- eta_l = [0.02, 0.01, 0.01, 0.005], linearly annealed to 0.
- M_chars_per_layer = 100M; batch 32 x 8192.
- Readout features = 1408 (concat layers 2-4).
- Readout N rows = 5M (subsample stride 8).
- Ridge lambda = 1e-2 * trace(Phi^T Phi) / 1408 (scale-aware).
- SEED honored via os.environ["SEED"] for the (small) RNG used in subsampling.

## 5. Expected wall time (A100-80GB)
- Per layer forward+Hebbian update: a causal conv at C_out=512, C_in<=512, K=5, dilation<=8 on 100M tokens is ~512*512*5*1e8 = 1.3e14 MACs. In bf16 at ~150 TFLOP/s sustained -> ~9 s/layer of compute. Hebbian outer-product update adds ~2x cost -> ~20 s/layer. 4 layers -> ~80 s.
- Feature collection (40M chars through 4 frozen layers): ~30 s.
- Phi^T Phi (1408 x 1408 from N=5M rows, fp32, chunked): ~5M * 1408^2 = 1e13 ops -> ~5 s; Cholesky 1408^3 -> <1 s.
- Loading / encoding / overhead: ~20 s.
- Total: ~140 s, comfortable margin under 300 s. Doubling channels to [512,512,768,768] is the slack-use plan if we finish early.

## 6. Success criterion
- Honest target: val_char_acc >= 0.55 on first 60K val chars (would already be a novel positive result for SoftHebb on language).
- Stretch: 0.65. Bar-clear: 0.70 (treated as upside; we will not tune for it).
- Energy: aim < 8 kJ (single-pass training, no optimizer state). The interesting axis here is joules-per-acc, not raw acc.

## 7. Failure modes anticipated
- WTA collapse: one channel wins everywhere -> dead features. Mitigation: monitor channel-usage entropy at end of each layer; if below 0.5 * log(C), raise tau and re-train that layer (one retry allowed).
- Exploding activations: Oja term should self-normalize; if ||W_l|| drifts, project rows to unit norm every 50 steps.
- Poor readout: 1408 features may underfit 256-way distribution. Fallback: also include layer 1 (-> 1792 features). Not enabled by default to keep ridge fast.
- OOM during feature collection: N=5M x 1408 fp16 = ~14 GB, OK. If we widen channels, drop stride to 16 (N=2.5M).
- Short receptive field (61 bytes) vs. PPM/ESN (effectively ~100+): inherent ceiling on acc; not fixable within this method.

## 8. What we will NOT do
- NOT use backprop, BPTT, or any gradient-based optimizer anywhere in the conv stack.
- NOT fine-tune conv weights on the readout cross-entropy (would violate the "local rule only" premise of this branch).
- NOT use BatchNorm / LayerNorm / residual connections (not in the SoftHebb recipe; would confound the result).
- NOT use a non-linear readout (MLP), kernel ridge, or per-layer readouts; one global linear readout only.
- NOT mix in a small Transformer or n-gram backoff; this spec tests SoftHebb features in isolation.
- NOT exceed 100M training chars per layer; if undertrained, that is a finding, not a thing to paper over.

---

Channel counts: [384, 384, 512, 512] (layers 1-4); readout features = 1408 (concat of layers 2-4).
Success criterion: val_char_acc >= 0.55 honest / 0.65 stretch / 0.70 bar-clear upside; energy budget < 8 kJ.
