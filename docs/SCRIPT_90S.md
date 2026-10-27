# 90-second script (2-act)

Speak this. Do not demo a dashboard. No GIF in this file.

## Act 1 — repo #1 (design, ~25 s)

A cascade over driving frames only works if the *join* is honest: a pedestrian box is not a crossing event. Pre-register the headline class, freeze the GT list, and refuse to promote empty classes.

## Act 2 — this repo (eval, ~65 s)

nuScenes mini, 360 windows, GT n=60 pre-registered. Cascade is T0 rules → Grounding-DINO-tiny → IteraCompute `qwen/qwen3.8-27b` (not qwen3-vl-plus). Kill-switch passed 3/3.

Headline `{pedestrian_cross}` recall@tIoU=0.5 is **1.0** (10/10) at precision **0.196**. That is not a strong detector: any VRU box fires an event, and window-level matching saturates (tIoU ≥ 0.98; three TPs are even cross-camera). The story is the join: box-level VRU is P=0.440 / R=0.212.

API: 4 claims, recall 0.3 / precision 0.75 — qualitative only (`n_pos<5`). Two of those four are GT-bounded hallucinations.

CODA-LM Test 500, images complete: object-R **0.345** (387/1123). Barrier/cone zeros are phrase-coverage, not missing files.

Frontier is three $0 points. Routing is a skeleton: step at dummy confidences; τ* spatial 0.8 is not a calibrated policy. The reseller's logprobs flag is a stub.

One README number equals one primary cell. That is the product.
