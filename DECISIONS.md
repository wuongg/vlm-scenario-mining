# Decisions

Non-obvious choices. One adopted fix or one rejected reason per line.
Source of truth for names: `docs/repo_design_v1.md`. Architecture docs `architecture_final_v5_mvp.md` and `architecture_final_v4.md` (DESIGN_LOG) were **not in the workspace on Day 1**.

## Header

| id | date | status | scope | summary |
|----|------|--------|-------|---------|
| D1 | 2026-10-26 | ADOPTED | CONFIG | Extra core deps numpy/pyarrow/duckdb/shapely — required by frozen signatures, not listed in the Day 1 core bullet |
| D2 | 2026-10-26 | ADOPTED | INGEST | Keyframe table gap is 0.5 s (2 Hz), not 1/12 s; 1/12 s is native camera rate if sweeps are ingested |
| D3 | 2026-10-26 | ADOPTED | TAXONOMY | 9+1 classes reconstructed from `repo_design_v1.md` mining-rule identifiers + DATA_SOURCE caveats; swap `configs/taxonomy.yaml` if DESIGN_LOG ontology differs |
| D4 | 2026-10-26 | ADOPTED | GEOM | Synthetic oracle (hand-computed pixels) is the CI fixture; real-devkit compare is a local gate after download |
| D5 | 2026-10-26 | ADOPTED | INGEST | Mini tgz uses structure/gzip/tar integrity, not a self-computed SHA; map v1.3 and CODA-LM Test pin SHA after download |
| D6 | 2026-10-26 | ADOPTED | INDEX | DuckDB FTS stemmer='none'; Python BM25 fallback if the fts extension cannot load |

## Log

- ADOPTED D1: `project_box_global_to_pixel`, parquet, `create_duckdb`, and `load_map_layers(...) -> Polygon` cannot run without numpy/pyarrow/duckdb/shapely. Names unchanged; only the lockfile grows.
- ADOPTED D2: `frames.parquet` is keyframes (fps=2). Mini keyframe gaps include 0.40/0.45/0.55 s (12 Hz tick slip + rounding). `ts_gap_eps_s=0.12` (one tick + margin). Data is not rewritten. repo_design's "1/12 s" is the camera tick, not the keyframe period.
- ADOPTED D3: DESIGN_LOG ontology section was missing. Class ids equal mining rule names (`cut_in`, `lead_braking`, `lane_split_moto`, `lane_intrusion`, `sidewalk_mount`) plus `cut_out`, `overtaking`, `ped_cross`, `red_light`, and `none`.
- ADOPTED D4: CI must not import nuscenes-devkit or read `data/`. Fixture uses a second, deliberately-simple pinhole oracle plus mutation tests.
- ADOPTED D5: User rule: no self-SHA on the mini tgz. Map expansion and CODA-LM Test record sha256 in `data/manifest.yaml`.
- ADOPTED D6: FTS extension needs a network install; tests must still pass. Same EN documents, two query backends.
- REJECTED: video scrubber, dashboard, MCP, few-shot corpus, GT freeze — Day 1 gold-plate / Day 2 work.
- REJECTED: installing torch on the default extra — CI stays CPU-only; `[detect]` and `[local]` are opt-in with a 30-minute timebox.
