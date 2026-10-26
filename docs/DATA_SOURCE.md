# Data source

## nuScenes mini (P0 corpus)

- Direct URL (public, no account): https://www.nuscenes.org/data/v1.0-mini.tgz
- License: [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/)
- Expected: 10 scenes. Sample/keyframe count is measured at ingest and written to `docs/NEXT_STEPS.md` (proposal 10 / ~404 / 360 windows).
- Cameras used: front-triad `CAM_FRONT`, `CAM_FRONT_LEFT`, `CAM_FRONT_RIGHT`.
- Map expansion pin: **v1.3**. Official portal is account-gated; Day 1 also tries `https://www.nuscenes.org/data/nuScenes-map-expansion-v1.3.zip` then the Zenodo fallback in `configs/ingest.yaml`. SHA-256 is recorded in `data/manifest.yaml` after the first local download.
- Zenodo [record 15667707](https://zenodo.org/records/15667707) (`10.5281/zenodo.15667707`) is the public copy of **nuScenes-map-expansion-v1.3.zip** (398.5 MB; folders `basemap` / `expansion` / `prediction`). Cross-checked against the official [nuscenes-devkit map-expansion notes](https://github.com/nutonomy/nuscenes-devkit) (v1.3 = lidar basemap + 11 semantic layers; extract those three folders into `maps/`). Local size 398,535,531 bytes matches the Zenodo listing.
- Singapore maps in mini: `singapore-onenorth` (1 scene), `singapore-hollandvillage` (3), `singapore-queenstown` (2) plus `boston-seaport` (4). **SG scene count = 6.** Do not copy a proposal number into a paper table.

## Caveats that bind eval

- **`red_light_violation` has no GT** on nuScenes (no signal-state labels). The class is a *violation event*, not a lamp state. `signal_state` is an attribute (on|off|unclear). Unsupported-claim audit may mention it; GT-bounded hallucination must not.
- **Night confound:** mini has 3 Night scenes, all in `singapore-hollandvillage`. A night-condition claim is a map claim until a larger split exists. See `configs/ingest.yaml` `conditions.night_map_confound`.
- **GT bbox drop-rate** from `box_in_image` is a separate column (`gt_drop_rate`). Dropped boxes are not VLM misses.

## CODA-LM Test 500 (pre-downloaded Day 1, not in the P0 headline)

- Annotations: https://huggingface.co/datasets/KaiChen1998/coda-lm (Test split, n=500)
- Images: CODA2022 test, [CODA download](https://coda-dataset.github.io/download.html)
- License: follow CODA / CODA-LM terms (research). SHA-256 of the annotation archive is pinned in `data/manifest.yaml`.
- Role: optional corner-case probe after kill-switch, not a Day-1 metric.
