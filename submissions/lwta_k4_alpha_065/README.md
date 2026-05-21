# lwta_k4_alpha_065 — LWTA-k=4 NN + W31 n-gram at α=0.65

**Paradigm:** Stack two 2026-05-19 wins on top of clean W31 hybrid:
* `lwta_k4_plus_w31` (α=0.7): 12,102 J / 0.7332
* `alpha_065` (ReLU^2, α=0.65): 15,307 J / 0.7387

`lwta_k4_plus_w31` won at α=0.7, but `alpha_065` showed α=0.65 is a better
mixing weight when the NN is well-trained. With LWTA-k=4 (sparser activations,
only 25% of MLP active), the NN may benefit even more from extra n-gram weight
at predict time (35% n-gram vs 30%).

**Mechanism:**
* W31 GPU order-12 KN n-gram (verbatim build path on GPU via int64-packed keys).
* d=256 / L=4 / 1200 steps modded-nanogpt with LWTA-k=4 in MLP (Muon + AdamW).
* α=0.65 hybrid: `p_final = 0.65*p_nn + 0.35*p_kn`.

**L2-clean:** Yes. KN tables built via `torch.unique`-equivalent sort on GPU.
NN training fully GPU (Muon, AdamW, attention, MLP all on CUDA). No CPU
multiprocessing.

**Hypothesis:** 11-13 kJ / 0.733-0.740. Target: 0.738+ acc.

**Expected DQ risk:** Low. lwta_k4_plus_w31 (parent) passed cleanly at α=0.7.

## Smoke test

```bash
.venv/bin/python -c "
import sys, importlib.util
sys.path.insert(0, '/Users/naka/src/sutro/wikitext')
spec = importlib.util.spec_from_file_location('sub', '/Users/naka/src/sutro/wikitext/submissions/lwta_k4_alpha_065/submission.py')
sub = importlib.util.module_from_spec(spec); spec.loader.exec_module(sub)
from wikitext import evaluate, load_wikitext103, CharModel
t = load_wikitext103('/Users/naka/src/sutro/wikitext/fixtures/tiny', 'train')
v = load_wikitext103('/Users/naka/src/sutro/wikitext/fixtures/tiny', 'valid')
model = sub.train(t)
assert isinstance(model, CharModel)
r = evaluate(model, v[:50])
print(f'SMOKE PASS: chars={r.n_chars} acc={r.accuracy:.3f}')
"
```
