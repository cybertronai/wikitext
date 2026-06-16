# nanogpt_comp_muon_mp1

`modded_nanogpt` fork with Tilde Research's isotropic Compositional Muon on attention `V`/`O` pairs. This variant uses `CM_MP = 1.0`.

Mechanism:

- `attn.v.weight` and `attn.proj.weight` are updated by an OV-only Compositional Muon step.
- The OV step uses the cheap isotropic partner-whitening approximation from Tilde's release.
- Q/K and MLP 2-D weights stay on ordinary Muon.
- Embeddings, LM head, biases, and RMSNorm gains stay on AdamW.

Hypothesis: partner-dependent OV scaling improves nanoGPT training efficiency without the extra cost of full Gram inverse roots.

Knobs:

- `CM_MP`: CM learning-rate multiplier, default `1.0`.
- `CM_DAMPING`: partner-norm damping, default `1e-2`.
- `N_STEPS`: override training steps.
- `SMOKE=1`: tiny local smoke configuration.
- `SEED`: reproducibility hook.

Reference: https://blog.tilderesearch.com/blog/compositional-muon
