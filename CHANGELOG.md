# Changelog

All notable changes to this project are documented here.
Format: what changed and why.

## [Unreleased]

- Day-8 official GT-60 first run (grid provenance): ped_cross recall@tIoU {0.3,0.5,0.7} + Wilson/n_eff + per-scene + condition + FP + claim-audit. Routing skeleton (`eval/routing.py`) with frozen U=1·%auto−5·FN-auto.
- D21: CODA2022-test 500/500 on disk. Official object-R 0.345 (387/1123). Do not quote the Day-7 missing-image figure. D22 stitch duration cap 30 s / 15 windows: corpus 221→45, overmerge 0.
- Kill-switch 3/3 pass — v1.0-track continues. D20 ship 2-point frontier (corpus API not retried). `results/cost_frontier.csv` frozen (draft grid numbers until Day 9).
- Pre-reg v0.2.3 appendix: corpus API held at GT-60. Detector event vs box P/R split (D19). `eval/metrics.py` n_eff-Wilson + kill-switch script. Phrase keep (D18).
- Session-3 GT 60/60; canary-lite freeze (D17) blocked corpus API. missing_none_flag → schema_violation (0757). Draft ped_cross stitch/recall + quadrant on GT-60.

## [v0.3-mined] — 2026-10-26

- GT-60 mined on detector (60/60, 4,111 boxes) + API tier (55/60 parse_ok, iteracompute `qwen/qwen3.8-27b`, $0).
- Pre-reg v0.2.2 backend switch DashScope → IteraCompute (D16). Canary structural checks required every session.
- Canary 4-check grounding = class-claim bbox on a single middle frame (video 503; 6-image 400 `too_many_images`).
- D14: GT-60 detector uses 6 frames/window; corpus uses 1 middle frame. Overnight two-pass + checkpoint/heartbeat.
- D15: T0 cut_in requires closure slope + ego-lane occupancy. 105 → 0 flags; rule stays enabled.
- Session-2 GT: 35/60 labeled (last-write-wins). Anchor alarm 1/5 flip (C3) → range re-label.
- Tighten pedestrian_cross join to v2 (ped ground in crossing polygon). JOIN-FP 2/5 → 1/3.
- T0 Mode A/B + closure slope-fit + sanitized-manifest assert. CODA-LM Test 500 loader (object-level, denom 500).
- Detector D12: combined caption, 960×540, one middle frame per window.

## [v0.2-eval-frozen] — 2026-10-26

- Swap taxonomy to the spec 9+1 (safety vs compliance) so GT freeze does not encode the Day-1 reconstructed class list.
- Fix GT drop-rate (camera-frame z) and remap `load_map_layers` through `polygon_token` so the ped_crossing join can run.
- Freeze pre-reg v0.2: headline = {pedestrian_cross}, GT n=60 SHA `a380a034…e2215c`, local_vlm = local-only after the RAM gate failed.

## [v0.1-day1] — 2026-10-26

- Scaffold the P0 repo (configs, ingest, geometry fixture, VLM schema, labeling tool, keyword index, CI) so Day 2 can freeze GT without renaming modules.
- Pin every numeric knob in `configs/*.yaml` and refuse magic strings; integrity and geometry gates run on synthetic fixtures in CI so nuScenes never enters GitHub Actions.
- Add a keyword-only fallback backend (not a baseline) plus detector/API skeletons so the mini-pipeline and `vsm search` have a real, schema-valid path on Day 1.
