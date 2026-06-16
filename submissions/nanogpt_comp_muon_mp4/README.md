# nanogpt_comp_muon_mp4

`modded_nanogpt` fork with Tilde Research's isotropic Compositional Muon on attention `V`/`O` pairs plus row-uniform Muon on the MLP expansion matrix. This first-touch variant uses `CM_MP = 1.0`, `MLP_MP = 1.0`, and a shortened 1600-step schedule.

Mechanism:

- `attn.v.weight` and `attn.proj.weight` are updated by an OV-only Compositional Muon step.
- The OV step uses the cheap isotropic partner-whitening approximation from Tilde's release.
- `mlp.fc.weight` uses a cheap Aurora/NorMuon-inspired row-uniform Muon update for tall matrices.
- Q/K and `mlp.proj.weight` stay on ordinary Muon.
- Embeddings, LM head, biases, and RMSNorm gains stay on AdamW.

Hypothesis: after OV CM, the next structural bottleneck is Muon's tall-matrix leverage anisotropy in the ReLU^2 MLP expansion. Equalizing per-neuron update mass may reduce early dead units and improve pass-threshold energy without full Aurora overhead.

Result:

- Total energy: `42,058.8 J` (`34,234.8 J` GPU + `7,824.0 J` CPU)
- Training duration: `185.0s`
- Validation accuracy: `0.7286`
- Native CE: `1.3182` bits/char, pass under `CE_MAX = 1.35`
- GPU: NVIDIA A100-SXM4-80GB

Knobs:

- `CM_MP`: CM learning-rate multiplier, default `1.0`.
- `CM_DAMPING`: partner-norm damping, default `1e-2`.
- `MLP_MP`: MLP row-uniform Muon learning-rate multiplier, default `1.0`.
- `N_STEPS`: override training steps, default `1600`.
- `SMOKE=1`: tiny local smoke configuration.
- `SEED`: reproducibility hook.

References:

- https://blog.tilderesearch.com/blog/compositional-muon
- https://blog.tilderesearch.com/blog/aurora
