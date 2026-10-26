# Next steps + measured counts

Proposal numbers are **not** the paper numbers. Measured on nuScenes mini after `integrity_gate` + `build_frames_table` on 2026-10-26.

## Headline (frozen)

Primary event-level cell = recall@tIoU=0.5 on `{pedestrian_cross}` only (pre-reg v0.2.3). Day-8 first-run grid: union **1.0** (10/10) — freeze Day 9.

CODA-LM Test 500 is the primary **quantitative object-level** table (D13, frozen fallback: 8/9 classes <3 positives). Mini is pedestrian_cross event-level + qualitative.

## Measured

| quantity | proposal | measured | note |
|---|---|---|---|
| scenes | 10 | **10** | integrity_gate asserts 10 |
| keyframes (samples) | ~404 | **404** | inside band [390, 420] |
| frames rows (kf × 3 cams) | 1,200 | **1,212** | 404 × 3 |
| windows | 360 | **360** | N=floor((K−6)/3)+1 per (scene, cam) |
| SG scenes | unspecified | **6** | 3 hollandvillage + 2 queenstown + 1 onenorth; Boston 4 |
| night frames / windows | 3 night scenes, all hollandv | **363 frames / 108 windows** | confound holds |
| GT drop-rate mean | — | **0.799** | D8. FRONT 0.705 / LEFT 0.903 / RIGHT 0.790 |
| geometry max pixel error | <2 px | **0.00 px** | synthetic oracle |
| GT labeling | "26 h" (wrong) | **~24 s/window → 60 ≈ 25–30 min** | session-1 n=10, mean 24.1 s. GT is not the bottleneck. |
| moto ≥3 frames | — | **72 windows** | T0 `lane_split_moto` support |
| join v2 JOIN-FP | — | **1/3 = 33%** | D11; below 40% fallback. v1 was 2/5 = 40% |
| scene-0061 share of GT-60 | — | **15/60** | do not redraw; leave-one-scene-out (appendix v0.2.1) |
| T0 flags / 360 | — | **92 flags (T0-enabled 92 / T0-disabled 0)** | D15 cut_in 105→0 (tightened, still enabled). sidewalk_mount 33; lane_intrusion 21; pedestrian_cross 21; lane_split_moto 17 |
| DINO steady-state | — | **13.45 s/frame combined** | load 4.4 s once. 960×540 = 13.85 s (no win). per-phrase 105 s. 360 middles ≈ 81 min; 2160 ≈ 8.3 h |

## Design deltas (ask, not silently patch)

1. `architecture_final_v5_mvp.md` and `architecture_final_v4.md` were **not in the workspace**. Taxonomy is the spec 9+1 (D7).
2. Frames table is keyframes at fps=2, not 1/12 s. Gate uses `ts_gap_s=0.5 ± 0.12`.
3. Mini ships **13** JSON tables; gate asserts the **11** load-bearing ones.
4. Map pin = Zenodo SHA `9dbc80a0…be5a144`.
5. CODA-LM Test pin dir SHA `55e94b29…608217` (annotations). Images are CODA2022-test, not in that snapshot.
6. nuScenes `sample.json` has no `data` map — join via calibrated_sensor → sensor.

## Day 2 (done)

- Headline `{pedestrian_cross}`. GT list n=60 SHA `a380a034…e2215c`. Session 1: 10 labeled (1 pedestrian_cross+doubt, 1 none+doubt, 8 none).
- local_vlm RAM gate failed → local-only (D10).

## Day 3 (done; tag stayed v0.2-eval-frozen)

- T0 Mode A/B on 360 → `results/t0_flags.parquet`. Sanitized-manifest assert.
- Join v2 + 10 re-handchecks. DINO config D12 (combined 960×540, 1 frame/window).
- CODA-LM path on: load 500 (1,123 GT objects), overlap 0. Detector smoke 50: object-level recall 0.177 (not official).
- 6 Day-3 commits pushed (`e6a3b38`).

## Day 7 (done)

- Kill-switch **3/3 pass — v1.0-track continues** (sunk-cost). Days 8–10 = official eval + routing skeleton + README. No v1.0 tag today.
- D20: **SHIP 2-point frontier**. No corpus-API retry. `results/cost_frontier.csv`: detector ($0, R=1.0, P=0.196, 9944 s) + API-GT60 ($0, R=0.3, P=0.75, 765 s) + union ($0, R=1.0, P=0.196, 10708 s). Draft grid until Day 9.
- Corpus detector **360/360** (4457 boxes, 5159 s). Stitch ped_cross 221→60 events (overmerge 15) — **superseded Day 8 / D22**.
- CODA-500 on Day 7 was 50/500 images. The missing-image figure is a **denominator artifact** (missing = miss). **Do not quote it as CODA recall.** See Day 8 / D21.
- GT-60 schema **ok** 60/60 (2-axis, event_id, latency, doubt).

## Day 8 (done)

- D21: fetched remaining CODA2022-test images (HF `CODA2022-test.zip`, 428 MB) → **500/500** jpgs. Official object-R **0.345 (387/1123)**, 0 missing, 5578 s. VRU 0.784 / vehicle 0.762; barrier/cone/other/light 0 under the combined caption. Smoke 0.177 on the first 50 is appendix only. Do not quote the Day-7 missing-image figure as recall. Pre-reg stays v0.2.3.
- D22: stitch guard `max_event_duration_s=30`, `max_windows=15`, gap≤1. Corpus **221→45** (delta −176), overmerge **0**. (Old 60 = 45 + 15 cap-6 fragments.) GT-60 detector **51→39**, overmerge 0. Index headline = 45.
- scene-1077 T0 silent: night hollandv, 0 ped boxes, 14 moto (not ≥3-frame straddle), ped_cross v2 join not met, cut_in 0. Not a map-lookup crash. `results/error_taxonomy.json`.
- Routing skeleton: U=1·%auto − 5·FN-auto. Family-level GT-60 sweep. τ* spatial=0.8 (U=0.367); maneuver/compliance=0.3 (U=0.85, n_gt=0). Step-function (API has no logit). Plot `results/day8_routing_skeleton.png`.
- Official GT-60 first run (grid, not Day-9 freeze): union ped_cross recall@tIoU 0.3/0.5/0.7 = **1.0** (10/10), Wilson n_eff=10 ICC=0 → [0.72, 1.0] diagnostic. Per-scene min/med/max = 1.0 (5 scenes with GT pos). API 0.3 (3/10). Claim-audit: 2/4 GT-bounded hallucination, 0/4 detector-unsupported (n=4, raw counts only). CSV `results/official_eval.csv`.
- D20 2-point frontier unchanged. Box-level split stays a note, not a 4th point.
- No v1.0 tag.

## Remaining (Days 9–12)

- Day 9: freeze the official table (same cells as Day 8 unless a bug). Interpretation gate.
- Day 10–11: README T1/T2 skeleton; still no v1.0 tag until that lands.
- Day 12: pretty plots (today's `day8_*.png` are drafts). Tag v1.0 after README + interpretation_gate.

## Day 6 (done — commit `0e5e22d`)

- Pre-reg **v0.2.3** appendix: corpus API held at GT-60 (D17). Detector-only corpus. Day 7 retry-or-ship → D20 ship.
- Detector 41 FP: 15 sampled (seed 13) → standing 7 / object-FP 7 / boundary 1. Box-level VRU P=0.440 R=0.212 vs event-level P=0.196 R=1.0 (GT-60, τ=0.30, IoU>0.5).
- Frontier detector-only = ($0, R=1.0, P=0.196). API tier contributes claims, not coverage; union ≈ detector on this corpus.
- 0757 oscillation evidence in `results/0757_oscillation.json`. Phrase keep (D18).
- `eval/metrics.py` implemented (n_eff-Wilson, per-scene range, condition-breakdown, interpretation_gate). `scripts/killswitch_check.py` local pass.
- CODA 500 starts after the corpus detector pass finishes (checkpointed, resume via `data/cache/detector/coda/`).

## Day 5 (done — commits `b1e1d94` `5e0a650`)

- Tag **v0.3-mined** pushed (`93749b0`). Gitleaks: git history + targeted src/scripts/results clean; `results/env` gitignored. full-tree scan skipped: data/raw is not tracked; history + code scans clean.
- Canary-lite session-2: **freeze** after re-collect (D17). 0757 class-set claim→none; 0916 parse_ok flip. Corpus API **not started**. GT-60 API tokens remain 76,424 / 30M.
- GT **60/60**. Session-3 anchors **0/5 flip**. C1/C2 not rewritten from the model.
- Draft ped_cross recall@0.5 (grid, not official): API 0.3 (3/10); detector/union 1.0 (51 pred, 41 FP). Stitch detector 51→39 (delta −12), over-merge 0. API tier contributes claims, not coverage; union ≈ detector on this corpus.
- FP taxonomy GT-60 API: schema-forced 5, perception-limit 2, C1 disagreement 1.
- Quadrant: agree-true 1 / T0-only 5 / VLM-only 3 / both-miss-GT 4.

## Day 4 (done — tagged v0.3-mined)

- D14: corpus = 1 middle frame; GT-60 = full 6 frames. Sequential overnight, checkpoint 50, atomic write, heartbeat.
- D15: cut_in tightened (slope + ego-lane ≥25% ≥2). **105 → 0**. status=enabled. Coverage splits T0-enabled / T0-disabled.
- GT: **35/60** labeled (session-2 +25, last-write-wins). Session-2 anchors: **1/5 class-flip** (C3) → STOP + re-read card + range re-label none.
- Detector: GT-60 full-6 **done** (60/60, 4111 boxes, 0 partial JSON, 4785 s ≈ 80 min). Corpus 360 middles running.
- API: IteraCompute `qwen/qwen3.8-27b` (pre-reg **v0.2.2**, D16). Canary 5/5 schema JSON; 1 middle image/request; grounding on claims only. GT-60 mined 60/60 (parse_ok 55; schema_violation 3; loop_detected 2). Tokens 70,260 + 6,164 = **76,424** ($0 / 30M mo). Corpus API skipped (headroom).

## Effort (corrected)

| task | old guess | now |
|---|---|---|
| GT labeling 60 | ~26 h | **25–30 min** (24 s/window) |
| Detector 2,160 native per-phrase forwards | overnight | **infeasible on CPU** — D12 cuts to 360 resized combined forwards |
| T0 360 | — | CPU, map+bbox, no torch |
