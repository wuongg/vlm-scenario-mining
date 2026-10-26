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
