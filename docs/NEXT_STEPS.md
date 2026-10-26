# Next steps + measured counts

Proposal numbers are **not** the paper numbers. Measured on nuScenes mini after `integrity_gate` + `build_frames_table` on 2026-10-26.

## Headline (frozen)

Primary event-level cell = recall@tIoU=0.5 on `{pedestrian_cross}` only (pre-reg v0.2.1).

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
| T0 flags / 360 | — | **197 flags, 153 windows** | lane_split_moto 17; pedestrian_cross 21; cut_in 105 (loose 2D); lead/oncoming/turn/red=0 |
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

## Day 3 (this tag stays v0.2-eval-frozen; next tag = v0.3-mined)

- T0 Mode A/B on 360 → `results/t0_flags.parquet`. Sanitized-manifest assert.
- Join v2 + 10 re-handchecks. DINO config D12 (combined 960×540, 1 frame/window).
- Smoke E2E 5 labeled windows. **DASHSCOPE_API_KEY still needed before Day 4.**
- CODA-LM path on: load 500 (1,123 GT objects), overlap check (no `nuscenes_sample_tokens.json`, 0 mini-token leak → held-out stands). Detector smoke 50 images: object-level recall **0.177** (vehicle 0.68 / vru 0.88 / cone+barrier 0 — phrases do not cover them). Not the official 500-image number.
- Smoke E2E 5 labeled windows: parse_ok **1.0**. Search works. **DASHSCOPE_API_KEY still missing — needed before Day 4.**

## Effort (corrected)

| task | old guess | now |
|---|---|---|
| GT labeling 60 | ~26 h | **25–30 min** (24 s/window) |
| Detector 2,160 native per-phrase forwards | overnight | **infeasible on CPU** — D12 cuts to 360 resized combined forwards |
| T0 360 | — | CPU, map+bbox, no torch |
