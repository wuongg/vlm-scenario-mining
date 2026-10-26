# Eval pre-registration (DRAFT — Day 1)

Eight load-bearing items. **Headline class-set waits on `prevalence_audit` (Day 2).** Numbers live in `configs/eval.yaml`.

| # | Item | Frozen on Day 1 | Still DRAFT |
|---|---|---|---|
| 1 | Primary metric | Recall @ tIoU=0.5, greedy score-match, window unit | Class-set in the headline |
| 2 | tIoU sweep | {0.3, 0.5, 0.7}; AP is secondary | — |
| 3 | GT size / allocation | n=60; stratified by map then light; ≥5 windows/map | Which 60 window_ids + SHA |
| 4 | Out-of-headline rule | <3 positives → drop from headline | Applied after prevalence |
| 5 | Seeds | `gt_list=7`, `label_interleave=13` | — |
| 6 | Backends in the comparison | detector (GD-tiny, frozen τ sweep) + api_vlm (qwen3-vl-plus, T=0, top_p=1) + fallback (CI-only, **not a baseline**) | Which of api/local actually runs |
| 7 | Kill-switch | Day 7 18:00 local; checklist in `configs/eval.yaml` | Pass/fail |
| 8 | Confounds | Night=hollandv; red_light has no GT; drop-rate ≠ VLM miss | Condition breakdown tables |

## Narrative fallbacks (skeleton)

| If this happens | Fallback (pre-registered, not a fishing license) |
|---|---|
| A class has <3 GT positives | Out of headline; still in appendix prevalence |
| API 403/quota TERMINAL | Report detector + (optional) local; stop spend |
| Geometry fixture red on real mini | Do not mine; quarantine scenes |
| Night claim is map-confounded | Condition-breakdown only |
| Detector extra cannot install <30 min | Skeleton stays; fallback is not promoted to a baseline |
| Map expansion SHA mismatch | Quarantine mapful Mode A; Mode B flagged `geometry_mode=B_mapless` |

Primary claims in the README (Day 12) must equal the primary table. `interpretation_gate` will assert that.
