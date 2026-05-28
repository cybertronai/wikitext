# DiffusionBlocks: Blockwise Training for Generative Models via Score-Based Diffusion

**Authors:** Makoto Shing (Sakana AI), Masanori Koyama (University of Tokyo), Takuya Akiba (Sakana AI)

**Venue:** ICLR 2026 (The Fourteenth International Conference on Learning Representations)

**arXiv:** [2506.14202](https://arxiv.org/abs/2506.14202) · **OpenReview:** [pwVSmK71cS](https://openreview.net/forum?id=pwVSmK71cS)

---

## Abstract

Training large neural networks with end-to-end backpropagation creates significant memory bottlenecks, limiting accessibility to state-of-the-art AI research.
This paper proposes **DiffusionBlocks**, a novel training framework that interprets neural network blocks as performing denoising operations in a continuous-time diffusion process.
By partitioning a network into independently trainable blocks and optimizing noise level assignments based on equal cumulative probability mass, the approach achieves significant memory efficiency while maintaining competitive performance compared to traditional backpropagation in generative tasks.
Experiments across five architectures spanning image classification, image generation, and text generation demonstrate memory reduction proportional to the number of blocks while matching or exceeding end-to-end performance across all settings.

---

## 1. Introduction

As neural networks grow following established scaling laws, they become increasingly inaccessible to much of the research community.
Training models with hundreds of billions of parameters requires computational resources available only to select institutions, threatening to concentrate AI advancement within well-resourced organizations.

The fundamental bottleneck is **end-to-end backpropagation**, which requires storing intermediate activations across the entire network, resulting in prohibitive memory demands.
This problem is especially acute for generative AI, where large-scale models are essential for high-quality generation.

Previous layerwise training approaches have underperformed compared to end-to-end backpropagation for two primary reasons:

- They lack principled mechanisms to coordinate information flow between independently trained layers.
- They struggle to balance parameter allocation effectively.

Moreover, these prior approaches have been evaluated mostly on image *classification* tasks, with limited exploration of generative modeling.

Meanwhile, **diffusion models** have revolutionized generative modeling through their mathematically principled approach to distribution transformation.
Recent advances in network conditioning and sampling efficiency have established diffusion models as state-of-the-art across multiple domains.

### Key Idea

DiffusionBlocks reconceptualizes neural network training by interpreting each network block as implementing a discretized step of a **continuous-time reverse diffusion process**.
The core innovation is a principled mapping between network blocks and noise-level ranges based on **equal cumulative probability mass**, ensuring each block faces an equally challenging learning problem.
This enables independent block training without requiring gradient communication between blocks.

### Contributions

1. A diffusion-inspired blockwise training framework achieving true block independence in continuous time, where each block can be trained without requiring gradients from other blocks.
2. An **equi-probability partitioning strategy** that optimally allocates learning difficulty across blocks based on cumulative probability mass, ensuring balanced parameter utilization.
3. Comprehensive empirical validation across **five architectures** (ViT, DiT, Masked Diffusion, AR Transformer, Recurrent-depth Transformer) spanning image classification, image generation, and text generation — demonstrating B-fold memory reduction with competitive or superior performance throughout.
4. An extension to **recurrent-depth (looped) Transformers**, replacing K-iteration backpropagation through time with a single forward pass during training.

---

## 2. Background

### 2.1 Score-Based Diffusion Models

Let $z_0 \in \mathbb{R}^d \sim p_\text{data}$ denote a clean data sample.
Under the Variance-Exploding (VE) formulation, we perturb $z_0$ with Gaussian noise whose standard deviation $\sigma(t)$ increases monotonically with time $t \in [0, 1]$:

$$z_t = z_0 + \sigma(t)\epsilon, \quad \epsilon \sim \mathcal{N}(0, I)$$

This gives marginal distribution $p_t(z_t) = \int p_\text{data}(z_0)\, p_t(z_t | z_0)\, dz_0$.

The **Probability Flow ODE (PF-ODE)** shares the same marginals as the stochastic process but follows deterministic trajectories:

$$\frac{dz_t}{dt} = -\dot{\sigma}(t)\sigma(t) \nabla_z \log p_t(z_t)$$

By setting $\sigma(t) = t$ and reparameterizing directly in terms of noise levels, this simplifies to:

$$\frac{dz_\sigma}{d\sigma} = -\sigma \nabla_z \log p_\sigma(z_\sigma)$$

The score function is approximated via a denoiser network $D_\theta(z_\sigma, \sigma)$ that predicts the clean data:

$$\nabla_z \log p_\sigma(z_\sigma) \approx \frac{D_\theta(z_\sigma, \sigma) - z_\sigma}{\sigma^2}$$

The denoiser is trained with a weighted L2 loss:

$$\mathcal{L}(\theta) = \mathbb{E}\left[ w(\sigma) \| D_\theta(z_\sigma, \sigma) - z_0 \|_2^2 \right]$$

### 2.2 Neural Network Block Structure

Consider a deep network with $L$ layers, producing intermediate activations $\{z^{(l)}\}_{l=0}^{L}$.
End-to-end backpropagation requires storing all these activations, with memory scaling linearly with depth and batch size.

When partitioning into $B$ blocks, each block $i$ groups consecutive layers together.
The core challenge: defining appropriate training objectives for each block without end-to-end supervision.

### 2.3 Residual Connections as Euler Steps

The connection between residual networks and continuous-time ODEs is well-established: residual updates $z^{(l)} = z^{(l-1)} + g_{\theta_l}(z^{(l-1)})$ correspond to Euler discretizations of ODEs.

Applying Euler discretization to the reverse diffusion PF-ODE at noise levels $\sigma_0 > \sigma_1 > \cdots > \sigma_N$ gives:

$$z_{\sigma_l} = z_{\sigma_{l-1}} + \frac{\Delta\sigma_l}{\sigma_{l-1}} \underbrace{\left(z_{\sigma_{l-1}} - D_\theta(z_{\sigma_{l-1}}, \sigma_{l-1})\right)}_{=:\, g_{\theta_l}(z_{\sigma_{l-1}})}$$

where $\Delta\sigma_l = \sigma_{l-1} - \sigma_l > 0$.

Each denoising step naturally takes the form of a **residual update**, matching the structure of modern architectures with skip connections.
This explains why DiffusionBlocks is architecturally restricted to networks with explicit residual connections: ResNets, U-Nets, and Transformer blocks with residual paths are all well-suited.
Architectures without skip connections would require implicit ODE solvers, which are computationally more complex.

---

## 3. Method

### 3.1 Diffusion-Based Blockwise Training

The reinterpretation is straightforward:

- The **network input** corresponds to noise ($z_{\sigma_\text{max}} \sim \mathcal{N}(0, \sigma_\text{max}^2 I)$).
- The **network output** corresponds to clean data ($z_0 \sim p_\text{data}$).
- Each **block** performs partial denoising over an assigned noise-level range.

For a network with $L$ layers partitioned into $B$ blocks, block $i$ is responsible for noise range $[\sigma_i, \sigma_{i+1}]$.
Block $i$ is trained with the following objective:

$$\mathcal{L}(\theta_i) = \mathbb{E}\left[ w(\sigma) \| D_{\theta_i}(z_\sigma, \sigma, x) - y \|_2^2 \right], \quad \sigma \sim p_\sigma^{(i)}$$

For language modeling, the L2 loss is replaced with cross-entropy after appropriate normalization.

Each block-specific denoiser is self-contained (it includes input embedding, transformer layers, and output projection), making blocks **truly independent**.
During training, only the activations of the active block need to be stored, resulting in approximately **B-fold memory reduction**.

### 3.2 Equi-Probability Block Partitioning

Different noise levels present varying degrees of difficulty.
Intermediate noise levels are most challenging and impactful for learning; very low or very high noise levels are comparatively easier.

To optimize parameter utilization, block boundaries are assigned so that each block handles an **equal amount of cumulative probability mass** under the (log-normal) noise distribution:

$$\sigma_i = \exp\!\left( P_\text{mean} + P_\text{std} \cdot \Phi^{-1}(p_i) \right)$$

where $p_i = \text{CDF}_\text{min} + \frac{i}{B}(\text{CDF}_\text{max} - \text{CDF}_\text{min})$ and $\Phi^{-1}$ is the inverse standard normal CDF.

This ensures:

$$\int_{\sigma_i}^{\sigma_{i+1}} p_\sigma(\sigma)\, d\sigma = \frac{1}{B}$$

The equi-probability boundaries concentrate in the challenging intermediate noise region, in contrast to naive **uniform partitioning** (equal intervals in log-space), which over-allocates parameters to easy regions.

### 3.3 Controlled Block Overlap

To mitigate potential discontinuities at block boundaries, each block's training range is expanded to:

$$[\sigma_i / \alpha,\; \sigma_{i+1} \cdot \alpha], \quad \text{where } \alpha = (\sigma_{i+1}/\sigma_i)^\gamma$$

Here, $\gamma$ is the overlap coefficient.
This ensures smoother transitions during inference by allowing each block to learn from samples slightly outside its primary range.
Across all experiments, $\gamma = 0.1$ provides the best balance between block independence and transition smoothness (see ablations in Section 4.3).

### 3.4 Implementation Details

The implementation follows the EDM framework (Karras et al., 2022) including its preconditioning strategy.
Training selects a block uniformly at random at each step, samples a noise level within that block's range, applies the block-specific denoiser, and updates only that block's parameters.
Inference assigns each ODE solver step to the appropriate block based on the current noise level.

---

## 4. Experiments

DiffusionBlocks is validated across **three task domains and five architectures**, demonstrating that the framework is architecture-agnostic and generalizes well beyond the generative tasks in the workshop preprint.

### Summary of Results

| Task | Architecture | Dataset | DiffusionBlocks | End-to-End |
|---|---|---|---|---|
| Image Classification | ViT | CIFAR-100 | competitive | baseline |
| Image Generation | DiT | ImageNet-256 | FID 15.55 | FID 16.62 |
| Image Generation | DiT | CIFAR-10 | FID 41.39 | FID 41.87 |
| Text Generation | Masked Diffusion | text8 | competitive | baseline |
| Text Generation | AR Transformer | OpenWebText | competitive | baseline |

In all cases, DiffusionBlocks matches or exceeds end-to-end training while using approximately 1/B the training memory.

### 4.1 Image Generation (DiT)

**Setup:** CIFAR-10 and ImageNet-256 using Diffusion Transformer (DiT) architectures (DiT-S/2 and DiT-L/2), partitioned into 4 blocks.
All models trained with classifier-free guidance, label dropout probability 0.1.
ImageNet images are compressed using a pretrained VAE before training.

**Results (FID, lower is better):**

| Method | CIFAR-10 | ImageNet-256 |
|---|---|---|
| End-to-End BackProp | 41.87 | 16.62 |
| **DiffusionBlocks (ours)** | **41.39** | **15.55** |

DiffusionBlocks achieves better FID on both datasets while requiring only 1/4 the memory.
A significant secondary advantage: inference is approximately **3× faster**, since each diffusion step only requires a forward pass through the relevant block rather than the entire network.

### 4.2 Language Modeling

**Setup:** One Billion Words Benchmark (LM1B), Llama-style architecture with 12 transformer layers partitioned into 4 blocks.
Evaluation via MAUVE score on conditional generation, following the SEDD protocol.

**Results (MAUVE, higher is better):**

| Method | MAUVE ↑ |
|---|---|
| End-to-End BackProp | 0.595 |
| **DiffusionBlocks (ours)** | **0.779** |

The improvement in MAUVE score is substantial: 0.779 vs. 0.595, despite training with only 1/4 the memory.

### 4.3 Ablation Studies

All ablations use CIFAR-10 with the same architecture and hyperparameters.

#### Block Partitioning Strategy

Block overlap is disabled in this ablation to isolate the effect of partitioning.

| Strategy | FID ↓ |
|---|---|
| Uniform | 68.06 |
| **Equi-Probability (ours)** | **45.50** |

Equi-probability partitioning provides a consistent advantage by concentrating block capacity where denoising is most difficult.

#### Block Overlap Coefficient

| Overlap Coefficient $\gamma$ | FID ↓ |
|---|---|
| 0.00 (none) | 45.50 |
| 0.05 | 42.98 |
| **0.10** | **41.39** |
| 0.15 | 42.84 |
| 0.20 | 56.69 |

Without overlap, discontinuities between independently trained blocks degrade performance.
Excessive overlap ($\gamma \geq 0.15$) also hurts, likely due to conflicting learning objectives.
$\gamma = 0.1$ is the optimal point.

#### Block Count

| Blocks | FID ↓ | Layers/Step | Relative Speed |
|---|---|---|---|
| B=1 (End-to-End) | 41.87 | 12 | 1.0× |
| B=2 | 38.58 | 6 | 2.0× |
| B=3 | 41.39 | 4 | 3.0× |
| **B=4** | **41.39** | 3 | 4.0× |
| B=6 | 53.74 | 2 | 6.0× |

There is a clear quality-efficiency trade-off.
$B = 3$ or $B = 4$ appear to be practical sweet spots, providing substantial efficiency gains while preserving competitive generation quality.
Beyond $B = 6$, individual blocks become too small (2 layers each) to perform effective denoising, and quality degrades significantly.

### 4.4 Extension: Recurrent-Depth (Looped) Transformers

DiffusionBlocks naturally extends to **recurrent-depth models** (also called Looped Transformers), which apply the same network $K$ times iteratively to progressively refine their output.
Standard training of such models requires **backpropagation through time (BPTT)** over all $K$ iterations, which is a major computational bottleneck.

The DiffusionBlocks perspective resolves this directly: the iterative refinement dynamics of a looped Transformer match the "gradual progress toward the target" that each block's denoising objective captures.
As a result, $K$-iteration BPTT training can be replaced with a **single forward pass** during training, while the original $K$-iteration procedure is preserved at inference.
This yields substantial computational savings without sacrificing performance.

---

## 5. Related Work

### Diffusion and Score-Based Generative Models

Diffusion models and score-based generative models define processes that gradually transform simple distributions into complex ones through sequences of denoising steps.
DiffusionBlocks leverages these mathematical foundations for network training itself, rather than for generation alone.

### Layerwise / Blockwise Training

Prior approaches (Synthetic Gradients, Feedback Alignment, Forward-Forward, Target Propagation, Blockwise SSL) share the goal of avoiding end-to-end backpropagation but face two fundamental limitations: lack of principled theoretical foundations for inter-block coordination, and limited effectiveness on generative modeling tasks.
DiffusionBlocks addresses both by grounding each block's objective in continuous-time diffusion theory, where each block's denoising objective naturally aligns with the global generative goal.

### Memory-Efficient Implicit Depth Models

Neural ODEs achieve constant memory backpropagation via the adjoint sensitivity method, but still require end-to-end backpropagation through a single monolithic network.
Deep Equilibrium Models (DEQs) achieve memory efficiency through implicit computation, but focus on equilibrium rather than generative modeling.
Neither enables genuine inter-block independence.

### Concurrent Work: NoProp

The closest concurrent work is NoProp (Li et al., 2025), which also interprets neural network training through diffusion principles.
However, NoProp's continuous-time variant employs a single network $\hat{u}_\theta(z_t, x, t)$ that must handle all noise levels $t \in [0, 1]$, requiring end-to-end backpropagation — more like Neural ODEs than blockwise training.
DiffusionBlocks achieves genuine blockwise independence in continuous time by partitioning the noise range into $B$ disjoint intervals.
Furthermore, NoProp focuses on classification tasks, while DiffusionBlocks demonstrates superior performance on generative modeling tasks (image generation and language modeling), directly comparing against conventional end-to-end backpropagation.

---

## 6. Conclusion

DiffusionBlocks enables independent neural network block training by interpreting blocks as denoising operations at specific noise levels in a continuous-time diffusion process.
The framework achieves:

- **B-fold memory reduction** during training
- **Competitive or superior performance** across five architectures spanning image classification, image generation, and text generation
- **Faster inference** (~3× for image generation) by routing each diffusion step to only the relevant block
- **Single-forward-pass training** for recurrent-depth (looped) Transformers, replacing expensive K-iteration BPTT

With $B = 4$, this translates to a 4× memory reduction with better FID and MAUVE scores than the full-network baseline.

---

## Appendix A: Limitations and Future Directions

### Limitations

1. **Architecture restriction:** DiffusionBlocks requires explicit residual connections (ResNets, U-Nets, Transformers), excluding feedforward networks and other non-residual architectures.
2. **Autoregressive inference overhead:** For language modeling, generating $K$ tokens with $M$ diffusion steps requires $O(KM)$ forward passes, compared to $O(K)$ for standard autoregressive generation.

### Future Directions

- **Alternative diffusion formulations:** Variance Preserving (VP), flow matching, stochastic interpolants, and bridge matching are all candidates for blockwise adaptation.
- **Theoretical understanding:** Why DiffusionBlocks outperforms end-to-end backpropagation is an open question; likely candidates are implicit regularization through diffusion-structured constraints and more efficient parameter utilization via equi-probability allocation.
- **Fast inference for language modeling:** Adapting recent fast diffusion samplers (DPM-Solver++, UniPC) could significantly reduce inference costs.
- **Block-parallel generation:** Simultaneous token generation could address sequential bottlenecks in autoregressive tasks.
- **Architecture and multimodal extensions:** Investigating optimal block designs per noise range and extending to Mixture-of-Experts models could broaden applicability.

---

## Appendix B: Mathematical Background

### B.1 Variance Exploding Diffusion (VE)

A perturbed sample at noise level $\sigma$ is defined as $z_\sigma = z_0 + \sigma\epsilon$, $\epsilon \sim \mathcal{N}(0, I)$.
The forward SDE is $dz_\sigma = \sqrt{2\sigma}\, dw$ and the reverse SDE is:

$$dz_\sigma = 2\sigma \nabla_z \log p_\sigma(z_\sigma)\, d\sigma + \sqrt{2\sigma}\, d\bar{w}$$

### B.2 Probability Flow ODE

The deterministic PF-ODE sharing the same marginal distributions as the SDE is:

$$\frac{dz_\sigma}{d\sigma} = -\sigma \nabla_z \log p_\sigma(z_\sigma)$$

### B.3 Score Estimation and Denoising Score Matching

The score is approximated as $\nabla_z \log p_\sigma(z_\sigma) \approx \frac{D_\theta(z_\sigma, \sigma) - z_\sigma}{\sigma^2}$.
Training loss:

$$\mathcal{L}(\theta) = \mathbb{E}\left[ w(\sigma) \| D_\theta(z_\sigma, \sigma) - z_0 \|_2^2 \right], \quad w(\sigma) = \frac{\sigma^2 + \sigma_\text{data}^2}{(\sigma \cdot \sigma_\text{data})^2}$$

### B.4 Noise Level Scheduling

Following EDM, noise levels are sampled from a log-normal distribution: $\log(\sigma) \sim \mathcal{N}(P_\text{mean},\, P_\text{std}^2)$.
This concentrates probability mass in intermediate noise regions, which contribute most to learning quality.

---

## Appendix C: Algorithms

### Training

```
Algorithm 1: DiffusionBlocks Training

Input: Dataset D, number of blocks B, noise range [σ_min, σ_max], log-normal params P_mean, P_std

1. Compute block boundaries {σ_0, ..., σ_B} via equi-probability partitioning
2. Initialize block parameters {θ_0, ..., θ_{B-1}}
3. While not converged:
   a. Sample block index i ~ Uniform(0, B-1)
   b. Sample data point (x, y) ~ D
   c. Sample noise level σ from block i's range [σ_i, σ_{i+1}]
   d. Sample ε ~ N(0, I); compute z_σ = y + σ·ε
   e. Compute loss:
      - Image generation: L = w(σ) · ||D_θᵢ(z_σ, σ, x) − y||²
      - Language modeling: L = w(σ) · CrossEntropy(Normalize(D_θᵢ(z_σ, σ, x)), y)
   f. Update only θ_i to minimize L
```

### Inference

```
Algorithm 2: DiffusionBlocks Inference

Input: Input x, trained blocks {θ_0, ..., θ_{B-1}}, boundaries {σ_0, ..., σ_B}, N inference steps

1. Discretize noise levels {σ^(0), ..., σ^(N)} via EDM schedule
2. Initialize z^(0) = σ^(0) · ε, ε ~ N(0, I)
3. For j = 0 to N-1:
   a. Determine block index i such that σ^(j) ∈ [σ_i, σ_{i+1})
   b. z^(j+1) = ODESolverStep(z^(j), σ^(j), σ^(j+1), D_θᵢ, x)
4. Return z^(N)
```

---

## Appendix D: Experimental Details

### D.1 Image Generation

**Architecture:** DiT-S/2 (CIFAR-10) and DiT-L/2 (ImageNet-256), partitioned into 4 blocks.
**Training:** AdamW, lr=1e-4, batch size 512 (CIFAR-10) / 1024 (ImageNet), 100 epochs, label dropout 10%.
**ImageNet:** Images resized to 256×256 and compressed via Stability AI's SDXL VAE before training.
**Inference:** Classifier-free guidance scale 2.0, Euler sampling, 50k samples evaluated.
**FID:** Computed as clean-FID against test sets; minimum over 3 runs reported.
**Fair comparison:** Total layer updates are matched between DiffusionBlocks and the end-to-end baseline across all experiments.

### D.2 Language Modeling

**Architecture:** Llama-style model with 12 transformer layers, 768 hidden dimensions, 12 attention heads, partitioned into 4 blocks of 3 layers each.
**Training:** LM1B dataset, AdamW, lr=3e-4, batch size 256, 10 epochs, context length 256 tokens.
**Attention masks:** Vectorized Training from Block Diffusion (Arriola et al., 2025) processes clean and noisy sequences jointly using specialized attention masks, avoiding multiple forward passes and KV cache overhead.
**Evaluation:** 1000 test sequences (≥100 tokens each), first 50 tokens as prompt, generate 5 samples of 50 tokens using 50 Euler diffusion steps; MAUVE computed between 5000 generated and 1000 reference sequences.

---

## Selected References

- Karras et al. (2022). *Elucidating the Design Space of Diffusion-Based Generative Models* (EDM). NeurIPS 2022.
- Peebles & Xie (2023). *Scalable Diffusion Models with Transformers* (DiT). ICCV 2023.
- Song et al. (2021). *Score-Based Generative Modeling through Stochastic Differential Equations*. ICLR 2021.
- Li et al. (2025). *NoProp: Training Neural Networks without Back-Propagation or Forward-Propagation*. arXiv:2503.24322.
- Ho et al. (2020). *Denoising Diffusion Probabilistic Models*. NeurIPS 2020.
- Siddiqui et al. (2024). *Blockwise Self-Supervised Learning at Scale*. TMLR 2024.
```bibtex
@inproceedings{shing2026diffusionblocks,
  title     = {DiffusionBlocks: Block-wise Neural Network Training via Diffusion Interpretation},
  author    = {Makoto Shing and Masanori Koyama and Takuya Akiba},
  booktitle = {The Fourteenth International Conference on Learning Representations},
  year      = {2026},
  url       = {https://openreview.net/forum?id=pwVSmK71cS}
}
```
