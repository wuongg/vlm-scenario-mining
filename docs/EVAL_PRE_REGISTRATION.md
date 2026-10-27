# Eval pre-registration v0.3 — FROZEN 2026-10-27

Changes after this file = version bump + changelog. Do not edit in place.

**Primary table executed as pre-registered** (Day 9 freeze). Headline cell = pedestrian_cross recall@tIoU=0.5 = 1.0 (10/10, detector=union). CSV `results/headline_metrics.csv`.

v0.3 adds only the execution appendix (matching saturation, claim-audit n=4 qualitative, Wilson diagnostic wording, CODA structural zeros, routing dummy confs). Headline, GT list, seeds, tIoU, kill-switch, backends, and prior appendices are unchanged.

v0.2.3 adds only the corpus-API appendix (D17). Headline, GT list, seeds, tIoU, kill-switch, backends, and prior appendices are unchanged.

v0.2.2 changes only the backend / decoding-parity rows (item 6, short items 1 and 4). Headline, GT list, seeds, tIoU, kill-switch, and the v0.2.1 scene-0061 appendix are unchanged.

**Why:** DashScope was unavailable on the user side. T1/T2 API tier moves to IteraCompute (`qwen/qwen3.8-27b`, OpenAI-compatible reseller, free 30M tok/month). Dated 2026-10-26. This model is **not** qwen3-vl-plus.

v0.2.1 added only the Appendix note on scene-0061 allocation concentration.

Numbers live in `configs/eval.yaml`. GT list SHA-256: `a380a0340051e72d8c41695256d949e49c058ee63f2a3c1c7153ebec68e2215c`.

## Eight load-bearing items

| # | Item | Frozen value |
|---|---|---|
| 1 | Primary metric | Recall @ tIoU=0.5, greedy score-match, window unit. **Headline class-set = {pedestrian_cross} only.** Sweep {0.3,0.5,0.7} is sensitivity, not a fishing license. |
| 2 | tIoU sweep | {0.3, 0.5, 0.7}; AP is secondary. tIoU=0.7 is valid only if the event is ≥2.5 s; timestamps quantized ±0.25 s. |
| 3 | GT size / allocation | n=60. Formula: `quota_m = 5 + leftover * (W_m / W_tot)` with `W_i = 1 * 2.0[if pseudo+] * 1.2[if night]`, sample without replacement, seed `gt_list=7`. ≥5 windows/map (measured 16/15/15/14). Hand-check 10 ids excluded. SHA above. |
| 4 | Out-of-headline rule | <3 corpus positives → coverage row, not headline. Applied: cut_in, lead_braking, oncoming, turn_lane_drift, lane_split_moto, lane_intrusion, sidewalk_mount, red_light_violation. |
| 5 | Seeds | `gt_list=7`, `label_interleave=13`, `search_demo=0`. |
| 6 | Backends | detector (GD-tiny, frozen τ sweep) + api_vlm = **iteracompute / `qwen/qwen3.8-27b`** (OpenAI-compatible reseller; free tier 30M tok/month; quantization may vary per deployment — **canary structural checks REQUIRED every session**). fallback = CI-only, **not a baseline**. **local_vlm = local-only branch** (RAM gate failed Day 2; not in the comparison). DashScope/`qwen3-vl-plus` is not in the comparison. |
| 7 | Kill-switch | 2026-11-01 18:00 +07. Checklist in `configs/eval.yaml`. |
| 8 | Confounds / claim-audit | Night=hollandv only. `red_light_violation` has no signal GT. drop-rate ≠ VLM miss. **GT-bounded hallucination** = false class on a GT window. **detector-unsupported claim rate** = corpus-wide claims with no detector box (includes detector misses). API bbox audit is **claim-level only** (bbox_2d seen on 1/1 canary claim; none-rows have no box; 1 middle frame, not video). |

### Interpretation rule

Primary table = 1 headline class × recall@0.5. README Day-12 number must equal that cell (`interpretation_gate`). Sweep numbers are appendix.

### n_eff-Wilson

`deff = 1 + (m̄ − 1) * ICC`, `n_eff = n / deff`, Wilson interval on `n_eff`. ICC from one-way ANOVA of per-scene recall (10 scenes). Report per-scene min/med/max. **Limitation: n_scenes=10** — treat the interval as a diagnostic, not a population CI.

### Labeling protocol

Session cap 75'. Interleave scene order with seed 13. `definition_doubt` is a real field. 5 borderline anchors = 1 window each of the highest-ambiguity pairs + 1 none + 1 rare (when a rare class appears). Alarm: ≥1 class-flip on an anchor → STOP → re-read card → re-label the range. Decision log: case-level note vs rule-change+backfill. **Minimum-viable GT n≥40** is pre-declared; n=60 is the target.

## Narrative fallbacks

| If this happens | Fallback (pre-registered) |
|---|---|
| A class has <3 GT positives | Out of headline; appendix prevalence only |
| API 403/quota TERMINAL | Report detector only; stop spend. local_vlm stays local-only. IteraCompute `free_model_monthly_limit` 429 is the same stop (no auto-overage). |
| Geometry fixture red on real mini | Do not mine; quarantine scenes |
| Night claim is map-confounded | Condition-breakdown only |
| Detector extra cannot install <30 min | Skeleton stays; fallback is not a baseline |
| Map expansion SHA mismatch | Quarantine Mode A; Mode B `geometry_mode=B_mapless` |
| VLM loses to T0 on a class | Flip the claim to "where does semantics pay" — T0 is the headline for that class |
| GT-bounded hallucination >30% | Mandatory breakdown: schema-error vs perception-error. Do not quote a single rate. |
| 6/9 classes empty on mini | CODA-LM Test 500 becomes the primary *quantitative object-level* table; mini stays qualitative / pedestrian_cross only |
| Join-FP on pedestrian_cross >40% of hand-checks | Tighten join (require ped bbox ∩ crossing) before scoring; current 2/5 pos hand-checks are JOIN-FP |

## Twelve short items

1. **Decoding parity** — T=0, top_p=1 (accepted on IteraCompute canary 2026-10-26). `enable_thinking=false` required (empty content otherwise). Video-file + fps **not** used: reseller cap is 1 image/request (video 503; 6-image 400 `too_many_images`). One repair (fence strip / truncation ×1.5).
2. **Tag ladder** — v0.2-eval-frozen → v0.3-mined → v1.0 only after kill-switch pass.
3. **Stitch → match** — stitch predictions first; match on stitched events; GT event_id is human, stitcher never writes GT.
4. **Model ID** — dated snapshot in `configs/model.yaml` (`qwen/qwen3.8-27b` IteraCompute 2026-10-26; GD-tiny 2024-05-28). Do not write qwen3-vl-plus.
5. **Seeds** — 7 / 13 / 0 as above.
6. **Stopping rule** — kill-switch 18:00 Day 7 or API TERMINAL, whichever first.
7. **tIoU 0.7** — valid only if event ≥2.5 s; timestamps ±0.25 s quantization.
8. **Hand-checks** — 10 ped_cross join checks, disjoint from the 60 (listed in `results/prevalence_summary.json`).
9. **Stratify metadata** — map then light; night confound recorded.
10. **Truncated flag** — `truncated=true` when finish_reason=length or tokens ≥0.95×max; parse_ok forced false.
11. **Subfail taxonomy** — refused / truncated / loop_detected / valid_but_empty / schema_violation.
12. **Condition columns** — light, blur_quartile, density_bin, visibility_bin on every frame; night=hollandv.

Primary claims in the README (Day 12) must equal the primary table.

## Appendix — scene-0061 concentration (v0.2.1)

Allocation oversample concentrates 15/60 in scene-0061; per-scene range reported; headline recall also reported leave-one-scene-out for scene-0061.

This is a reporting guard, not a redraw. The n=60 list and seed=7 formula stay frozen.

## Appendix — corpus API held at GT-60 (v0.2.3)

Corpus API tier held at GT-60 only due to backend instability (canary alarms D17). Corpus tier = detector-only. Frontier has 2 API points (GT-60) + detector corpus point.

**Day 7 (D20, sunk-cost):** SHIP 2-point frontier. No retry. GT-60 API already yields 4 claims for claim-audit; corpus API would only add coverage that union ≈ detector already has; reseller still oscillates (`results/0757_oscillation.json`). Corpus tier = detector-only. Frontier = (detector $0, event-R 1.0 / P 0.196) + (API GT-60, R 0.3 / P 0.75) + box-level split (VRU P=0.440 / R=0.212). CSV: `results/cost_frontier.csv`.

This is a reporting guard, not a backend swap and not a GT redraw.

## Appendix — Day-9 execution (v0.3)

Primary table ran as pre-registered. Deviations (reporting, not redraws):

1. **Matching saturation (D25).** Detector/union recall is 1.0 at tIoU {0.3, 0.5, 0.7}. Per-TP tIoU on the 10 matches is ≥0.98 (7 same-window identity; 3 cross-camera time-overlap). Window-level matching saturates at this GT granularity; event-level tIoU needs finer GT windows (single-window events). Matcher not tightened.
2. **Claim-audit n=4.** Raw counts only (2 GT-bounded hallucination, 0 detector-unsupported). `n_pos<5` → qualitative only, insufficient positives for a rate.
3. **Wilson n_eff=10, ICC=0.** Interval [0.72, 1.0] is a conservative diagnostic. Honest statement: 10/10 on the 60-window grid, clustered by 10 scenes.
4. **CODA structural zeros (D24).** Combined caption does not emit barrier/cone phrases — structural zero, documented. Not a missing-image hole. Official object-R 0.345 (387/1123) on 500/500.
5. **Routing dummy confs (D26).** IteraCompute accepts `logprobs=True` but returns 1 stub token vs 138 completion tokens — not verdict-token confidence. Self-consistency not run (D17 freeze / 0757 oscillation). Family τ frontier stays a step. Limitation, not a third backend.
