# Next steps + measured counts

Proposal numbers are **not** the paper numbers. Measured on nuScenes mini after `integrity_gate` + `build_frames_table` on 2026-10-26.

## Measured (Day 1)

| quantity | proposal | measured | note |
|---|---|---|---|
| scenes | 10 | **10** | integrity_gate asserts 10 |
| keyframes (samples) | ~404 | **404** | inside band [390, 420] |
| frames rows (kf × 3 cams) | 1,200 | **1,212** | 404 × 3; proposal undershot by 12 |
| windows | 360 | **360** | N=floor((K−6)/3)+1 per (scene, cam) matches |
| SG scenes | unspecified | **6** | 3 hollandvillage + 2 queenstown + 1 onenorth; Boston 4 |
| night frames / windows | 3 night scenes, all hollandv | **363 frames / 108 windows** | all `singapore-hollandvillage` — confound holds |
| GT drop-rate mean | — | **0.799** (Day-1 0.584 was a bug) | per (box, this camera) + z>near. FRONT 0.705 / LEFT 0.903 / RIGHT 0.790. 11,233 behind-cam boxes were kept after sign-flip without z. In-front-only denom = 0.604. Not a VLM miss. See D8. |
| geometry max pixel error | <2 px | **0.00 px** | synthetic oracle; mutations all fail as required |

## Design deltas to ask, not silently patch

1. `architecture_final_v5_mvp.md` and `architecture_final_v4.md` (DESIGN_LOG) were **not in the workspace**. Taxonomy 9+1 was reconstructed from mining-rule names in `repo_design_v1.md`. Swap `configs/taxonomy.yaml` if DESIGN_LOG differs.
2. repo_design ts-invariant says gap=1/12 s. Frames table is keyframes at fps=2. Mini keyframe gaps include 0.40/0.45/0.55 s (12 Hz tick slip). Gate uses `ts_gap_s=0.5 ± 0.12`. Data is not rewritten.
3. Mini ships **13** JSON tables; the gate asserts the **11** load-bearing ones in `configs/ingest.yaml` `tables_required`.
4. Official map URL `nuscenes.org/data/nuScenes-map-expansion-v1.3.zip` returned an HTML login page (200). Zenodo fallback is the pin. SHA-256 `9dbc80a095b6b28d9b79fc9a43471a750dc92ca78c6d0db288fd92b34be5a144`.
5. HuggingFace `archive/main.zip` for CODA-LM is HTML. Day 1 used `huggingface_hub.snapshot_download` of `CODA-LM/Test/*` (n=500). Dir SHA `55e94b2924731a3c5a0c8a78391209d61141f00879b0e7794c2a764558608217`.
6. Day-1 user brief asked for backends + keyword index + pre-reg draft, which repo_design §7 places on Days 3 and 6. Implemented as **skeletons that run**, not the full cascade.
7. nuScenes `sample.json` has no `data` map and `sample_data.json` has no `channel` — join via calibrated_sensor → sensor. Not a rename; the JSON dump != the Python NuScenes object.

## Day 2 (done)

- Headline frozen: `{pedestrian_cross}` only. 41 map-join positives (BOS 17 / HV-night 15 / ON 9 / QT 0). 8 classes out-of-headline (<3 verified pos).
- GT list n=60 SHA `a380a0340051e72d8c41695256d949e49c058ee63f2a3c1c7153ebec68e2215c`. Session 1 labeled 10 (1 pedestrian_cross with doubt, 1 none with doubt, 8 none). Mean latency 24.1 s.
- local_vlm RAM gate failed → local-only branch.
- T0 rules still Day 3.

## Day 3

- `vlm/prompt+parse` + detector/api smoke on 5 windows.
- `mining/rules` Mode A/B + cascade E2E.
