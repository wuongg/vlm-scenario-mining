# Capability manifest (reconstructed)

DESIGN_LOG (`architecture_final_v5_mvp.md` / v4) was **not in the workspace on Day 1**. This list is the P0 transfer as executed, not a recovered v5 table.

## Transfer (9)

1. Pre-registered eval (8 load-bearing items, n=60, seeds, tIoU sweep).
2. Three-stage cascade T0 → open-vocab detector → VLM.
3. Keyword index + negative-space coverage report.
4. Geometry fixture + map join v2 (ped ground-in-crossing).
5. Kill-switch with a dated checklist (passed 3/3).
6. Cost frontier at $0 (detector / API-GT60 / union).
7. Claim-audit: GT-bounded hallucination vs detector-unsupported.
8. Condition-breakdown with the night=hollandv confound recorded.
9. CODA-LM Test 500 as the frozen object-level fallback table.

## Not transferred (5)

1. Fine-tune / LoRA.
2. MCP server.
3. Dense embedder (bge-m3 + RRF).
4. Stacking / crop-verify pass-2.
5. Platt calibration.

## Partial (1)

1. Family-level routing skeleton (`eval/routing.py`). Utility frozen. 2-point frontier with a step at dummy confidences; per-class τ deferred (logprob stub). τ* spatial=0.8 is a skeleton result, not a calibrated policy.
