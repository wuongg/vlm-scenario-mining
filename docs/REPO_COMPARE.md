# Repo #1 ↔ repo #2 (one page)

Same discipline, different layer. Method signature is the product, not the sensor.

| signature | repo #1 ([ev-charging-intel](https://github.com/wuongg/ev-charging-intel)) | repo #2 (this eval) |
|---|---|---|
| Layer | Decision: rolling-horizon LP under site-capacity | Data: T0 → detector → VLM over a driving index |
| Pre-registered eval | VOI / policy table written before the official cells | v0.3 executed; `headline_metrics.csv` frozen; interpretation_gate |
| Frozen inputs | 2019 bank + plant I9 + T2C planner | GT n=60, seed=7, SHA `a380a034…e2215c`; no redraw for scene-0061 |
| Manifest / freeze | PROTOCOL + freeze hashes | mini structure (no self-SHA); map v1.3 Zenodo; CODA-LM Test dir SHA |
| Cost / frontier | VOI 182% at β=0.25; min-cap 79.6 kW | 3 points at $0; corpus API held (D17/D20) |
| Claim-audit | README numbers from `results/*.csv` via `make_report.py` | 2/4 GT-bounded; 0/4 unsupported on GT-60; 0757 + logprob stub |
| Honest failures | plant-bug record (I9); P3=P1 when 150 kW does not bind | 0757 oscillation; logprob stub; matching saturation; CODA phrase zeros |

Do not say repo #2 “beats” repo #1. Say: the same controls were applied to a driving index and survived a kill-switch.
