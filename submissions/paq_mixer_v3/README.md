# paq_mixer_v3

PAQ-style multi-order context mixing — Run 3 of the N8 adaptive budget.

## Why this exists

- v1: 3,244 J / 0.7121 — PASS but 76% above W31 (1,847 J).
- v2: 2,378 J / 0.7121 — PASS, fast-materialise saved 27% J.
- v2 still 29% above W31 J leader.

Run 3 targets: drop top-order (K=12 → K=11). The k=12 materialise was
27.8s and the most expensive step in v2. Skipping it saves ~700 J on
Modal at the cost of ≤0.5pp acc (order-12 contributes minimally since
only ~30% of bytes find a 12-byte match).

## Changes from v2

- **MAX_ORDER = 11** (was 12). Builds tables for ctx_len 0..10.
- Everything else identical to v2.

## Expected Modal numbers

- v2 Modal: 2,378 J / 0.7121 / 104.3s.
- v3 target: 1,650-1,850 J / 0.706-0.712.
- Beat W31 (1,847 J): plausible if v3 lands at ~1,800 J at acc ≥ 0.706.

## Adaptive-budget context

Run 2 → Run 3 trajectory shows substantial improvement (26% J cut).
Budget extends to 5 runs per adaptive-explore rule. Run 4 candidate:
push to K=10 or add a 3rd-layer mixer if Run 3 still doesn't beat W31.

## Author

`@worker-paq-mixer`
