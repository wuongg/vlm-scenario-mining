# VLM Scenario Mining

[![CI](https://github.com/wuongg/vlm-scenario-mining/actions/workflows/ci.yml/badge.svg)](https://github.com/wuongg/vlm-scenario-mining/actions/workflows/ci.yml) [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE) [![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](pyproject.toml)

Three-stage labeling cascade over a frame-level index of driving data.

<!-- interpretation_gate: 1.0 -->

## T1 — TL;DR

<!-- ≤150 words + GIF. Placeholder until Day 12 `make assets`. -->

## T2 — One page

### Pitch

Three-stage labeling cascade (rules → open-vocab detector → VLM) over a frame-level index of driving data. Pre-registered eval on nuScenes mini; CODA-LM Test 500 is the object-level fallback.

### Capability manifest

9 transfer / 5 not / 1 partial. Source: [`docs/CAPABILITY_MANIFEST.md`](docs/CAPABILITY_MANIFEST.md).

| bucket | items |
|---|---|
| Transfer (9) | pre-reg eval; T0→detector→VLM cascade; keyword index + negative-space; geometry fixture + join v2; kill-switch; $0 frontier; claim-audit; condition-breakdown; CODA-500 object table |
| Not (5) | fine-tune/LoRA; MCP; dense embedder; stacking/crop-verify; Platt calibration |
| Partial (1) | family-level routing skeleton (dummy confs until a logit source) |

### Results (frozen Day 9)

Headline class `{pedestrian_cross}` recall@tIoU=0.5 = **1.0** (10/10, detector = union). Precision 0.196. API 0.3 / 0.75 on 4 claims (qualitative only). CODA object-R **0.345** (387/1123). Frontier: detector ($0, R=1.0, P=0.196) · API-GT60 ($0, R=0.3, P=0.75) · union ($0, R=1.0, P=0.196). Box-level VRU P=0.440 / R=0.212 is a note, not a fourth point.

Window-level matching saturates (all 10 TPs have tIoU≥0.98; 7 same-window, 3 cross-camera time-overlap). The join rule is the story: event R=1.0 / P=0.196 vs box R=0.212 / P=0.440. Wilson [0.72, 1.0] on n_eff=10 is a conservative diagnostic; the honest statement is 10/10 on the 60-window grid, clustered by 10 scenes.

CSV: [`results/headline_metrics.csv`](results/headline_metrics.csv) · [`results/cost_frontier.csv`](results/cost_frontier.csv) · [`results/coda_official.csv`](results/coda_official.csv)

### What I deliberately did not build

Fine-tune, LoRA, MCP, dense embedder (bge-m3 + RRF), stacking / crop-verify, Platt calibration, video scrubber, dashboard, few-shot corpus, torch on the default extra. Routing exists as a P1 skeleton after the kill-switch; it is not gold-plated.

### Rejected findings

Adversarial design review: **37 findings** logged in [`DECISIONS.md`](DECISIONS.md) (**25 ADOPTED**, **8 REJECTED**, **3 CASE**, **1 REPLACED**). Public rejects:

| rejected | why |
|---|---|
| video scrubber, dashboard, MCP, few-shot corpus | Day-1 gold-plate / Day-2 work |
| torch on the default extra | CI stays CPU-only |
| 105-count 2D cut_in as n_pos | not T0, not a class label |
| redraw n=60 because scene-0061 is 15/60 | formula frozen; report per-scene + leave-one-out |
| add `person .` to the caption | would retune the frozen τ sweep (D18) |
| corpus-API retry after canary freeze | D17 freeze + D20 ship 2-point, no retry |

### Data & Licensing

nuScenes mini, [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/). Map expansion v1.3 (Zenodo `10.5281/zenodo.15667707`). CODA-LM Test 500 annotations (HuggingFace KaiChen1998/coda-lm) + CODA2022-test images; research terms. This repo is MIT; the data licenses are not. Night on mini is hollandv-only — a night claim is a map claim.

## T3 — Docs

- `docs/DATA_SOURCE.md`
- `docs/EVAL_PRE_REGISTRATION.md`
- `docs/NEXT_STEPS.md`
- `docs/DEFINITION_CARDS.md`
- `docs/CAPABILITY_MANIFEST.md`
- `docs/SCRIPT_90S.md`
- `docs/repo_design_v1.md`

## Reproduction

```bash
make install
make test
make lint
```
