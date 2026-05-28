# Experiment 06: Frozen Random-Features Layer Inside Gradient-Trained Transformer

## Hypothesis
Replacing **one** MLP block (out of 6) in the modded_nanogpt with a **frozen random-features layer** (a single linear projection W ∈ R^(d, 4d) with no trainable parameters, followed by ReLU², then a *trainable* linear back to d) loses ≤ 0.02 acc vs. baseline while saving ~17% of that block's trainable parameters and reducing per-step compute by skipping the fc-weight backward pass. Tests whether one MLP block out of six carries genuinely task-adapted features or is mostly a random expansion.

## Motivation
The "frozen random projection in the middle of an SGD-trained net" idea is the cheapest possible paradigm-B grad-free component. Crucially **it is not paradigm-A** because the layers around it remain SGD-trained. It is the FF-block / random-feature analog of `hopfield_layer`'s frozen K_mem.

**Prior art directly relevant.** Casanueva-Artís et al. 2025 ("Is Random Attention Sufficient for Sequence Modeling?", arXiv 2506.01115) test the symmetric question — freezing *all* MLPs vs. freezing *all* attention QK — on next-token LM. They find: Frozen-QK on Wikitext perplexity 3.07 vs. 2.78 for fully-trained (≈10% hit), but explicitly conclude *"Freezing the MLPs causes the most performance drop"* on memorization (Frozen-MLP loses ~2/3 of storage capacity). FreezeTST (arXiv 2508.18130) interleaves frozen random-feature reservoir blocks with trainable transformer layers on time series and reports parameter savings at minimal accuracy cost. Together these set the expectation: freezing *all* MLPs is bad; freezing *one of six* is the open empirical question this experiment closes for char-LM.

Note: `rff_linear_head` (exp 02) used random features as the *whole* representation and hit the paradigm-A 0.37 ceiling. This is the opposite: random features as one *component* inside an otherwise SGD-trained net.

## Method
Baseline 6-layer modded_nanogpt. Replace `MLP` at one block (sweep: at block 0, 2, 4) with:

```python
class FrozenRFMLP(nn.Module):
    def __init__(self, dim):
        super().__init__()
        hdim = 4 * dim
        W = torch.randn(hdim, dim) / dim**0.5
        self.register_buffer("W_frozen", W)   # frozen, no grad
        self.proj = Linear(hdim, dim)         # trainable
    def forward(self, x):
        h = F.linear(x, self.W_frozen)
        h = h.relu().square()
        return self.proj(h)
```

The 4d → d output projection remains trainable so the network can choose which random features to use. The fc (d → 4d) is frozen. This halves that block's trainable parameters and skips its weight-grad/backward.

## Memory-Movement Analysis
- Frozen W: 4·d² = 590 KB at d=384, lives in L2 forever. No optimizer state, no gradient buffer.
- Forward pass: identical to a normal MLP fc.
- Backward pass: skips the dW = grad_h.T @ x outer product (saves ~B·T·d·4d = 50 GFLOPs / step / block). Activation grad still flows back through W via x_grad = grad_h @ W.
- Net per-step compute saving: 1 block × ~5% per-step → ~5% energy saving if accuracy holds.

## Setup
- 6-layer modded_nanogpt with one block's MLP replaced by FrozenRFMLP.
- Sweep block index ∈ {0, 2, 4} (or pick block 2 — the same insertion point Hopfield won at).
- All other hyperparameters identical to `modded_nanogpt`.

## Procedure
1. `cp -r submissions/modded_nanogpt submissions/rf_mlp_block2`
2. Add `FrozenRFMLP` class above the existing `MLP`.
3. In `Block.__init__`, take a `frozen_mlp: bool = False` flag and conditionally use `FrozenRFMLP(dim)` instead of `MLP(dim)`.
4. In `GPT.__init__`, pass `frozen_mlp=(i == TARGET_BLOCK)` to each block.
5. Train. Submit.

## Success Criteria
- **Strong**: val ≥ 0.735 at energy ≤ 48 kJ → random features substitute for one MLP block at ~6% energy saving.
- **Pass**: val ≥ 0.72, any energy ≤ 51 kJ → MLP block partially substitutable.
- **Diagnostic**: even a refutation tells us each MLP block's fc is necessary, sharpening "what does SGD actually need to do here."

## Failure Modes & Diagnostics
- Variance scaling of W matters: try W_std ∈ {1/√d, 1/√(2d), 2/√d}; suboptimal init can kill accuracy independent of the hypothesis. Log activation std at init for each block.
- Position matters: block 0 (closest to embed) might need the most adaptation; block 4 might tolerate freezing best. Run all three insertion points if budget allows.

## Estimated Cost
1–3 Modal runs (one if just block 2; three if sweeping insertion point) ≈ $0.40–$1.25.

## References
- Rahimi & Recht 2007 "Random Features for Large-Scale Kernel Machines" — random features as kernel approximations, where the trainable head dictates which features matter.
- Casanueva-Artís et al. 2025 "Is Random Attention Sufficient for Sequence Modeling? Disentangling Trainable Components in the Transformer" (arXiv 2506.01115) — directly tests all-MLPs-frozen and finds large degradation; this experiment's single-block variant is the open question they leave for char-LM.
- "Frozen in Time: Parameter-Efficient Time Series Transformers via Reservoir-Induced Feature Expansion and Fixed Random Dynamics" (arXiv 2508.18130) — interleaves frozen random-feature blocks with trainable transformer layers; closest architectural cousin (different modality).
- Hinton 2022 "Forward-Forward" — block-local training; here we substitute training with no-training, asking what's lost.
