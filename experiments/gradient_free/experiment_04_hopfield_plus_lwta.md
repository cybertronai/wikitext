# Experiment 04: Hopfield Layer + LWTA MLP — Cross-Pollination of Both Winning Mechanisms

## Hypothesis
A 4-layer modded-nanogpt body where (a) the MLP activations use LWTA-k=4 instead of ReLU² AND (b) a frozen Hopfield retrieval layer sits between blocks 2 and 3 reaches val acc ≥ 0.735 at energy ≤ 38 kJ — strictly better than either `hopfield_layer` (40.2 kJ / 0.7293) or `lwta_k4` (46.2 kJ / 0.7238) alone. Both mechanisms add structured sparsity / retrieval to a smaller model.

## Motivation
The leaderboard has **two distinct gradient-free-flavored wins**: LWTA (k=4 winner-take-all activations) and Hopfield (frozen-memory retrieval). They affect orthogonal layers (MLP vs between-blocks) and should compose. This is the highest-EV cross-pollination experiment because both ingredients are already known to clear the floor.

## Method
Combine the two best-known submissions:
1. Take `submissions/hopfield_layer/submission.py` as scaffold (4 layers + Hopfield).
2. Replace the `MLP.forward` ReLU² nonlinearity with LWTA-k=4 from `submissions/lwta_k4/submission.py`.
3. Keep the Hopfield layer between blocks 2 and 3, M=4096.
4. Train.

## Memory-Movement Analysis
- LWTA: reduces MLP write bandwidth by 4× post-activation (3/4 of activations are zero). Hidden state (B, T, 4d) goes through scatter mask — same write traffic in dense Phase-1 mode but the proj output is sparser → potential for sparse output gemm in a Phase-2 kernel (out of scope here, dense mask sufficient).
- Hopfield: unchanged from exp 11. Sits between blocks, K_mem in L2.
- Net effect: both reduce per-layer effective capacity, both add specialization. No layer-wise FLOP increase; expected energy ≈ between LWTA and Hopfield alone, ≈ 38–42 kJ.

## Setup
- 4-layer body, LWTA-k=4 in MLPs, Hopfield M=4096 between blocks 2-3.
- All other hyperparameters from `hopfield_layer`.
- Baselines: `hopfield_layer` (40.2 kJ / 0.7293), `lwta_k4` (46.2 kJ / 0.7238), `modded_nanogpt` (51.7 kJ / 0.7374).

## Procedure
1. `cp -r submissions/hopfield_layer submissions/hopfield_lwta`
2. Add LWTA function (copy from `lwta_k4/submission.py`):
```python
LWTA_K = 4
def lwta_k(x, k):
    assert x.size(-1) % k == 0
    g = x.reshape(*x.shape[:-1], -1, k)
    winner = g.argmax(dim=-1, keepdim=True)
    mask = torch.zeros_like(g).scatter_(-1, winner, 1.0)
    return (g * mask).reshape(*x.shape)
```
3. Modify `MLP.forward`:
```python
def forward(self, x):
    x = self.fc(x)
    x = lwta_k(x, LWTA_K)   # was: x.relu().square()
    return self.proj(x)
```
4. Submit.

## Success Criteria
- **Strong**: val ≥ 0.735, energy ≤ 38 kJ → both mechanisms compose without interference and the combination beats baseline modded_nanogpt on acc/energy frontier.
- **Pass**: val ≥ 0.72, energy < 42 kJ → matches Hopfield-only baseline on the energy frontier with the LWTA capacity penalty absorbed.
- **Refutation**: val < 0.71 OR energy > 46 kJ → mechanisms interfere (e.g., LWTA-sparse activations break the Hopfield query distribution).

## Failure Modes & Diagnostics
- LWTA + Hopfield query: LWTA's sparse, hard-WTA activations may produce degenerate Hopfield queries (most entries zero). Mitigation: insert Hopfield *before* the first LWTA-MLP, not after — try insertion point ∈ {0, 1, 2}.
- LWTA breaks Muon's spectral assumptions inside the MLP fc layer: monitor MLP grad norms; if they explode, fall back to AdamW on those params (the `lwta_k4` submission already had this working).

## Estimated Cost
1–2 Modal runs (insertion point ablation if first run is borderline) ≈ $0.40–0.85.

## References
- `submissions/hopfield_layer/submission.py`
- `submissions/lwta_k4/submission.py`
- Srivastava 2013 "Compete to Compute" (LWTA original); Ramsauer 2020.
