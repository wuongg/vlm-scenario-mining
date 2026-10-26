# Decisions

Non-obvious choices. One adopted fix or one rejected reason per line.
Source of truth for names: `docs/repo_design_v1.md`. Architecture docs `architecture_final_v5_mvp.md` and `architecture_final_v4.md` (DESIGN_LOG) were **not in the workspace on Day 1**.

## Header

| id | date | status | scope | summary |
|----|------|--------|-------|---------|
| D1 | 2026-10-26 | ADOPTED | CONFIG | Extra core deps numpy/pyarrow/duckdb/shapely — required by frozen signatures, not listed in the Day 1 core bullet |
| D2 | 2026-10-26 | ADOPTED | INGEST | Keyframe table gap is 0.5 s (2 Hz), not 1/12 s; 1/12 s is native camera rate if sweeps are ingested |
| D3 | 2026-10-26 | REPLACED | TAXONOMY | Day-1 reconstruction from mining-rule names — superseded by D7 spec 9+1 |
| D7 | 2026-10-26 | ADOPTED | TAXONOMY | Spec 9+1: safety_maneuver / safety_spatial / compliance / none. Drop cut_out+overtaking. Add oncoming (wrong-way-only) + turn_lane_drift. Rename ped_cross→pedestrian_cross, red_light→red_light_violation. uncertain is a schema slot, not a class. |
| D8 | 2026-10-26 | ADOPTED | INGEST | Day-1 drop-rate 0.584 omitted camera-frame z. 11233/27112 behind-camera boxes were kept after sign-flip. Filter now uses z>near per (box, this camera). Remeasured mean 0.799 (FRONT 0.705 / LEFT 0.903 / RIGHT 0.790). Not a wrong-camera join. |
| D9 | 2026-10-26 | ADOPTED | EVAL | Headline = {pedestrian_cross} only. Spec fallback 3-class is not claimed: cut_in/lead_braking have <3 verified positives. |
| D10 | 2026-10-26 | ADOPTED | VLM | local_vlm RAM gate failed (host <16 GB). Comparison row is local-only; do not improvise a 2B number later. |
| D11 | 2026-10-26 | ADOPTED | EVAL | JOIN-FP 2/5 = 40% hit the frozen fallback. Tighten pedestrian_cross to ped ground-in-crossing (v2). Re-handcheck 10: 1/3 JOIN-FP (33% < 40%). |
| D12 | 2026-10-26 | ADOPTED | VLM | GD-tiny CPU: combined caption at 960×540, 1 middle frame/window. Per-phrase native is not a corpus path. Sweep box-threshold on the resized input. |
| D13 | 2026-10-26 | ADOPTED | EVAL | Frozen fallback 6/9 empty → CODA-LM Test 500 is the primary quantitative object-level table. Mini stays pedestrian_cross event-level + qualitative. |
| D14 | 2026-10-26 | ADOPTED | DETECT | corpus = 1 middle frame; GT-60 = full 6 frames; rationale = temporal evidence where scored |
| D15 | 2026-10-26 | ADOPTED | T0 | cut_in tightened (closure slope AND ego-lane ≥25% ≥2 frames). 105 → 0 flags. status remains enabled (<30). |
| C1 | 2026-10-26 | CASE | LABEL | definition_doubt, no rule-change: standing ped in lot aisle |
| C2 | 2026-10-26 | SUPERSEDED | LABEL | session-1 borderline ped_cross on 0061 FRONT 2939662460 — flipped in session-2 (C3) |
| C3 | 2026-10-26 | CASE | LABEL | session-2 anchor alarm: 1/5 class-flip. STOP + re-read card + range re-label none |
| D16 | 2026-10-26 | ADOPTED | VLM | Backend switch DashScope → IteraCompute. Canary 4-check. Grounding on claims only. |
| C4 | 2026-10-26 | CASE | LABEL | session-3 anchors 0/5 class-flip. C1/C2 kept (not rewritten from VLM). |
| D17 | 2026-10-26 | ADOPTED | VLM | Canary-lite session-2 freeze: 0757 class-set claim→none; 0916 parse_ok flip after missing_none_flag validator. Corpus API not started. Token-only alarms logged, not freeze. |
| D18 | 2026-10-26 | ADOPTED | DETECT | Keep combined caption; do not add `person .`. traffic/none 1387 and vehicle/none 1023 are documented generic leftovers, not a phrase bug. |
| D19 | 2026-10-26 | ADOPTED | EVAL | Detector event-level ped_cross R=1.0 P=0.196 vs VRU box-level P=0.440 R=0.212 (IoU>0.5, τ=0.30, GT-60). Object recall ≠ event precision; join turns boxes into events. |
| D20 | 2026-10-26 | ADOPTED | EVAL | Ship 2-point frontier. No corpus-API retry. Reseller oscillation (0757) still in force; union ≈ detector. |
| D4 | 2026-10-26 | ADOPTED | GEOM | Synthetic oracle (hand-computed pixels) is the CI fixture; real-devkit compare is a local gate after download |
| D5 | 2026-10-26 | ADOPTED | INGEST | Mini tgz uses structure/gzip/tar integrity, not a self-computed SHA; map v1.3 and CODA-LM Test pin SHA after download |
| D6 | 2026-10-26 | ADOPTED | INDEX | DuckDB FTS stemmer='none'; Python BM25 fallback if the fts extension cannot load |

## Log

- ADOPTED D1: `project_box_global_to_pixel`, parquet, `create_duckdb`, and `load_map_layers(...) -> Polygon` cannot run without numpy/pyarrow/duckdb/shapely. Names unchanged; only the lockfile grows.
- ADOPTED D2: `frames.parquet` is keyframes (fps=2). Mini keyframe gaps include 0.40/0.45/0.55 s (12 Hz tick slip + rounding). `ts_gap_eps_s=0.12` (one tick + margin). Data is not rewritten. repo_design's "1/12 s" is the camera tick, not the keyframe period.
- REPLACED D3: Day-1 class list was reconstructed from mining-rule identifiers because DESIGN_LOG was missing. That list is wrong vs the spec 9+1.
- ADOPTED D7 (REPLACE D3): `configs/taxonomy.yaml` now matches the spec. Families are safety_maneuver (cut_in, lead_braking, oncoming, turn_lane_drift), safety_spatial (lane_split_moto, sidewalk_mount, lane_intrusion, pedestrian_cross), compliance (red_light_violation), none. Multi-label axes stay 1 maneuver + 0-1 spatial-slot + optional pedestrian_cross. `signal_state` is an attribute and does not match. No GT was relabeled (none existed).
- ADOPTED D8: `extract_gt_boxes` already projected every sample ann through the camera being processed (nuScenes anns are camera-agnostic). The 0.584 mean was low because `filter_boxes_in_image` was called without z, so behind-camera corners that sign-flipped into the image were counted as kept. After z-filter the mean is 0.799. In-front-only denominator is 0.604 (FOV crop of boxes already in front). Dropped boxes remain not-a-VLM-miss.
- ADOPTED D4: CI must not import nuscenes-devkit or read `data/`. Fixture uses a second, deliberately-simple pinhole oracle plus mutation tests.
- ADOPTED D5: User rule: no self-SHA on the mini tgz. Map expansion and CODA-LM Test record sha256 in `data/manifest.yaml`.
- ADOPTED D6: FTS extension needs a network install; tests must still pass. Same EN documents, two query backends.
- REJECTED: video scrubber, dashboard, MCP, few-shot corpus, GT freeze — Day 1 gold-plate / Day 2 work.
- REJECTED: installing torch on the default extra — CI stays CPU-only; `[detect]` and `[local]` are opt-in with a 30-minute timebox.
- ADOPTED D9: Prevalence audit (map join `ped_cross_v1` n=41; 10 hand-checks; 20-window quick scan; T0 not implemented). Only pedestrian_cross clears the <3 rule. 2/5 join-positives were painted-crossing + curb-ped FPs — join is an upper bound.
- ADOPTED D10: `scripts/run_ram_test.py` → ram_ok=false. Pre-reg backend #6 records local-only today.
- REJECTED: using the 105-count 2D cut_in heuristic as n_pos — it is not T0 and is not a class label.
- ADOPTED D11: Join v1 (ego-path buffer ∩ ped_crossing AND any ped) is an upper bound. Fallback text is "tighten join (require ped bbox ∩ crossing)". Implemented as `ped_cross_v2_ped_ground_in_crossing`: unproject ped bbox bottom-center onto z=0 and test polygon.contains. Same 10 window_ids, disjoint from the 60. v2 positives 3/10; JOIN-FP 1/3 (scene-0103 painted crossing + sidewalk peds). 33% < 40% → do not reframe the headline. Pseudo-GT remains an upper bound.
- ADOPTED D12: Steady-state (load once = 4.4 s, then 20 CAM_FRONT forwards): combined-native **13.45 s/frame**, combined-960×540 **13.85 s/frame** (resize is not the lever — GD-tiny's CPU forward dominates), per-phrase 960×540 **105 s/frame** (8 phrases). Day-1 34 min / 3 windows was per-phrase × 6 frames, not model reload. Config: `caption_mode=combined`, `input_wh=[960,540]` (kept for the brief; no speed win), `frames_per_window=1` (middle keyframe). Estimates: 2,160 combined forwards ≈ 8.3 h; 360 middle-frame forwards ≈ **81 min**. Trade-off: miss a class visible only on the first/last frame of the 3 s window. Box-threshold sweep must be re-run under this caption/frame policy.
- ADOPTED D13: Prevalence 8/9 classes <3 verified positives (and 6/9 empty on the 10-label session). Pre-reg fallback: CODA-LM Test 500 becomes the primary *quantitative object-level* (group-level) table; mini is pedestrian_cross event-level + qualitative. Denominator frozen 500; parse-fail = miss. `nuscenes_sample_tokens.json` is absent from the CODA-LM Test snapshot; 0 mini-token overlap → held-out wording stands.
- CASE C1 (not a rule-change): window `scene-0916_CAM_FRONT_1538984236412460`. Trigger: pedestrian standing in a parking-lot aisle among parked motos. Decision: `class_set=["none"]`, `definition_doubt=true`. Card EXCLUDE for pedestrian_cross is waiting/standing with no trajectory into the roadway — lot aisle is not a crossing. Do not widen the class.
- CASE C2 (SUPERSEDED by C3): window `scene-0061_CAM_FRONT_1532402939662460`. Session-1: `pedestrian_cross` + definition_doubt (onenorth zebra / standing worker). Session-2 blinded anchor flipped it.
- ADOPTED D14: Forwards allocation. Corpus pass = 1 middle keyframe/window (~81 min, 360 forwards). GT-60 pass = full 6 frames/window (360 forwards ≈ ~81 min more) because GT-60 is where event-level scoring happens — temporal evidence belongs where the window is scored. Overnight sequential ~2.7 h CPU. Checkpoint every 50 windows, atomic parquet write, heartbeat. Knobs in `configs/model.yaml`: `corpus_frames_per_window=1`, `gt60_frames_per_window=6`.
- ADOPTED D15: T0 `cut_in` was a loose 2D lateral-enter heuristic (105 flags / 360, Day-2 discarded as n_pos). Tightened predicate: lateral enter AND `scale_slope_flag` (closure) AND bbox∩ego-lane ≥25% for ≥2 frames. Remeasure: **105 → 0**. 0 < 30 → keep `status=enabled` (do not disable). Coverage report splits T0-enabled vs T0-disabled even when the disabled bucket is empty. Flags still written; none are marked disabled today.
- ADOPTED D16: DashScope unavailable phía user. API backend = IteraCompute `qwen/qwen3.8-27b` (`https://api.iteracompute.com/v1`, `ITERACOMPUTE_API_KEY`). Pre-reg bumped to **v0.2.2** (not in-place). This is not qwen3-vl-plus. Free 30M tok/month, $0, no auto-overage. Canary 2026-10-26 on the 5 frozen windows, current prompt, T=0 / top_p=1 / `enable_thinking=false`:
  - (1) schema JSON: **5/5 pass** (fence-wrapped; strip is 0-call).
  - (2) bbox_2d [0,1000): **present on the 1/1 window that emitted a class claim** (`[405,505,435,680]`). None-rows have no box. Reseller rejects video-file (503) and 6-image lists (400 `too_many_images`) → **1 middle image/request**. T2 bbox audit is claim-level on that single frame, not video-grounding.
  - (3) none slot: 3/5 clean none; 0916 invented `pedestrian_cross` on a GT-none lot-aisle (C1); 0757 returned a per-frame empty list (no `none` flag). None/uncertain work, but the model can fabricate a class on standing-ped scenes.
  - (4) latency mean **12.7 s**; usage from `response.usage`: 1171 prompt + 24–183 completion / call (5-window total 6441 tokens).
  Decoding parity: T=0 and top_p=1 were accepted (no fallback values). Canary structural checks are **required** every API session (quantization may vary). Concurrency 5. 429 `free_model_monthly_limit` = stop.
- CASE C3 (session-2 anchor alarm, not a rule-change): 5 blinded anchors at start of session 2. **1/5 class-flip** on `scene-0061_CAM_FRONT_1532402939662460` (`pedestrian_cross` → `none`). Protocol: STOP → re-read card (TRIGGER = in-window crossing) → re-label the range. Adjacent GT-60 windows `…2938162460` and `…2941262460` labeled `none` (post-zebra van+cones, no ped in 6 frames). Other 4 anchors no-flip. Card unchanged. Last-write-wins loader replaces the session-1 row.
- ADOPTED: `assert_t0_no_gt_access` — deleting `sample_annotation` from a manifest leaves T0 flags identical. T0 reads `objects_json` + map polygons + `calibrated_sensor` only.
- REJECTED: redrawing the n=60 GT list because scene-0061 holds 15/60. Formula is frozen. Report per-scene range + leave-one-scene-out for scene-0061 (pre-reg v0.2.1 appendix).
- CASE C4 (session-3 anchors, not a rule-change): 5 blinded relabels. **0/5 class-flip.** C1 lot-aisle and C2/C3 zebra kept; not rewritten from IteraCompute output. 25 new windows labeled → GT 60/60.
- ADOPTED D17: Canary-lite session-2 vs `canary_baseline.json` (class-set ±1 / token >20% / parse_ok flip; no hashes). Recollect fired. Hard freeze: 0757 class-set `claim`→`none`; 0916 parse_ok flip (`schema_violation` after missing_none_flag). Token-only deltas logged (T=0 still varies 24–183). Corpus API **not started**; GT-60 API records kept. Ledger: freeze, $0.
- ADOPTED D18: Phrase-confusion traffic/none + vehicle/none are generic combined-caption boxes with no specific phrase match. Keep documented. Do not add `person .` — that would retune the frozen τ sweep on GT-60.
- ADOPTED D19: 15/41 event-FP sample (seed 13, contact sheet): event-FP-standing 7 / object-FP 7 / boundary 1. Standing+boundary = 8/15 event-semantics FPs; object-FP is the other half (van/cones/UPS/empty lot). Headline still: object-level detector recall is not event-level ped_cross precision; the join rule is what turns boxes into events. Box-level VRU P=0.440 R=0.212 vs event-level P=0.196 R=1.0. Frontier detector-only = ($0, R=1.0, P=0.196). τ sweep does not move this point (box threshold already frozen).
- ADOPTED D20 (Day 7, sunk-cost): SHIP 2-point frontier. Do not retry corpus API. Canary-lite would be required first; 0757 still oscillates at T=0. GT-60 API (4 claims) is enough for claim-audit. CSV `results/cost_frontier.csv`.
