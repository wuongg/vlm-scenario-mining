"""Parse/repair. Fence strip is 0 extra model calls. Truncation repair is one-shot."""

from __future__ import annotations

import json
import re
from typing import Any

from vsm.config import cfg_dict
from vsm.constants import CLASS_IDS, GEOMETRY_B, PROVENANCE_GRID
from vsm.vlm.schema import ObjectClaim, Subfail, VLMFinding, WindowOutput, class_family

_MAX = int(cfg_dict("model")["decoding"]["max_tokens"])
_FENCE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)


def _prompt_hash() -> str:
    from vsm.vlm.prompt import prompt_version_hash

    return prompt_version_hash()


def _strip_fence(raw: str) -> str:
    text = (raw or "").strip()
    m = _FENCE.search(text)
    return (m.group(1) if m else text).strip()


def classify_subfail(raw: str) -> Subfail | None:
    """Loop = repeated 8-gram. Empty after strip → valid_but_empty."""

    text = _strip_fence(raw)
    if not text:
        return Subfail.valid_but_empty
    lower = text.lower()
    if any(tok in lower for tok in ("i cannot", "i can't", "refused", "as an ai")):
        return Subfail.refused
    tokens = re.findall(r"\w+", lower)
    if len(tokens) >= 24:
        grams = [" ".join(tokens[i : i + 8]) for i in range(len(tokens) - 7)]
        if grams and grams.count(max(set(grams), key=grams.count)) >= 3:
            return Subfail.loop_detected
    try:
        json.loads(text)
    except json.JSONDecodeError:
        if "{" in text or "[" in text:
            return Subfail.schema_violation
    return None


def parse_output(raw: str, finish_reason: str, max_tokens: int, window_id: str = "parse") -> WindowOutput:
    """Strip fence with 0 extra calls. truncated if finish_reason=length or tokens ≥0.95×max."""

    truncated = finish_reason == "length"
    text = _strip_fence(raw)
    sub = Subfail.truncated if truncated else classify_subfail(raw)
    findings: list[VLMFinding] = []
    none_ = False
    uncertain = False
    parse_ok = True
    try:
        payload: Any = json.loads(text) if text else {}
    except json.JSONDecodeError:
        payload = {}
        if sub is None:
            sub = Subfail.schema_violation
    if isinstance(payload, list):
        empty = all(not (item.get("findings") if isinstance(item, dict) else item) for item in payload)
        if empty:
            # 0757: empty frame-list with no none/uncertain flag → schema_violation
            payload = {"none": False, "uncertain": False, "findings": []}
        else:
            payload = {"findings": [x for x in payload if isinstance(x, dict)]}
    if isinstance(payload, dict):
        label = str(payload.get("window_label") or payload.get("label") or "").lower()
        none_ = bool(payload.get("none") or payload.get("none_") or label == "none")
        uncertain = bool(payload.get("uncertain") or label == "uncertain")
        raw_findings = payload.get("findings") or []
        if isinstance(raw_findings, list):
            for rec in raw_findings:
                if not isinstance(rec, dict):
                    continue
                cls = str(rec.get("class") or rec.get("claim") or "")
                if cls not in CLASS_IDS:
                    continue
                bbox = rec.get("bbox_2d") or [0, 0, 1, 1]
                try:
                    box = tuple(int(v) for v in bbox[:4])
                    findings.append(
                        VLMFinding(
                            frame_idx=int(rec.get("frame_idx") or 0),
                            objects=[ObjectClaim(cls=cls, bbox_2d=box, lane_position="unknown")],  # type: ignore[arg-type]
                            claim=cls,
                            supporting_frames=[int(rec.get("frame_idx") or 0)],
                            confidence=float(rec.get("confidence") or 0.5),
                            family=class_family(cls),
                            signal_state=str(rec.get("signal_state") or "unclear"),  # type: ignore[arg-type]
                            attributes={},
                        )
                    )
                except (TypeError, ValueError):
                    sub = Subfail.schema_violation
        if not findings and not none_ and not uncertain:
            sub = Subfail.schema_violation  # missing_none_flag (0757)
    if sub is not None:
        parse_ok = False
        if sub == Subfail.valid_but_empty:
            none_ = True
            findings = []
    if truncated:
        parse_ok = False
        sub = Subfail.truncated
    del max_tokens
    return WindowOutput(
        window_id=window_id,
        findings=findings,
        none_=none_ and not findings,
        uncertain=uncertain,
        parse_ok=parse_ok,
        parse_subfail=sub,
        finish_reason=finish_reason,
        truncated=truncated,
        prompt_version_hash=_prompt_hash(),
        backend="parse",
        geometry_mode=GEOMETRY_B,
        provenance=PROVENANCE_GRID,
    )


def repair_once(output: WindowOutput, error_kind: str, raw: str = "") -> tuple[WindowOutput, str]:
    """Fence→strip (0 call). truncation→caller must retry max_tokens×1.5 once. Yield logged by caller."""

    if error_kind == "fence":
        repaired = parse_output(raw, output.finish_reason, _MAX, window_id=output.window_id)
        return repaired, "strip"
    if error_kind == "truncation":
        return output, "max_tokens_x1.5"
    return output, "none"
