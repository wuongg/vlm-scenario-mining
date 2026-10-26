"""Keyword CI-only backend.

NOT A BASELINE. Do not report numbers from this module as a detector or VLM result.
"""

from __future__ import annotations

import json
from typing import Any

import polars as pl

from vsm.constants import COLS, FALLBACK_NAME, GEOMETRY_B, PROVENANCE_GRID
from vsm.vlm.schema import ObjectClaim, VLMFinding, WindowOutput, class_family


def _objects(row: dict) -> list[dict]:
    raw = row.get(COLS.OBJECTS_JSON.value) or "[]"
    if isinstance(raw, list):
        return raw
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        return []
    return payload if isinstance(payload, list) else []


class FallbackBackend:
    """Keyword lookup on objects_json. NOT A BASELINE."""

    name = FALLBACK_NAME

    def mine_window(self, frames: Any, fps: float, prompts: dict[str, str]) -> WindowOutput:
        """Return WindowOutput from class keywords already sitting on the frames table."""

        del prompts
        if fps <= 0:
            raise ValueError("fps must be passed and > 0")
        rows = frames.to_dicts() if isinstance(frames, pl.DataFrame) else list(frames)
        window_id = "synth"
        if rows:
            scene = rows[0].get(COLS.SCENE_NAME.value) or rows[0].get(COLS.SCENE_TOKEN.value)
            cam = rows[0].get(COLS.CAMERA.value)
            ts0 = rows[0].get(COLS.TS_US.value)
            window_id = f"{scene}_{cam}_{ts0}"
        findings: list[VLMFinding] = []
        for i, row in enumerate(rows):
            for obj in _objects(row):
                cls_name = str(obj.get("class") or obj.get("cls") or "")
                mapped = _map_class(cls_name)
                if mapped is None:
                    continue
                bbox = obj.get("bbox_2d") or [0, 0, 1, 1]
                bbox_i = tuple(int(v) for v in bbox[:4])
                findings.append(
                    VLMFinding(
                        frame_idx=int(row.get(COLS.FRAME_ID.value, i)),
                        objects=[
                            ObjectClaim(
                                cls=mapped,
                                bbox_2d=bbox_i,  # type: ignore[arg-type]
                                lane_position="in_lane" if mapped == "lane_split_moto" else "unknown",
                            )
                        ],
                        claim=mapped,
                        supporting_frames=[int(row.get(COLS.FRAME_ID.value, i))],
                        confidence=float(obj.get("conf") or 0.5),
                        family=class_family(mapped),
                        signal_state="unclear",
                        attributes={"source": "fallback_keyword", "raw_class": cls_name},
                    )
                )
        none_ = len(findings) == 0
        return WindowOutput(
            window_id=window_id,
            findings=findings,
            none_=none_,
            uncertain=False,
            parse_ok=True,
            parse_subfail=None,
            finish_reason="stop",
            completion_tokens=0,
            truncated=False,
            prompt_version_hash="fallback-keyword",
            backend=self.name,
            geometry_mode=GEOMETRY_B,
            provenance=PROVENANCE_GRID,
            model_id="keyword",
            detector_supported=False,
        )

    def detect_objects(self, frames: Any) -> pl.DataFrame:
        """Pass through objects_json. NOT A BASELINE."""

        rows = frames.to_dicts() if isinstance(frames, pl.DataFrame) else list(frames)
        out = []
        for row in rows:
            for obj in _objects(row):
                out.append(
                    {
                        "frame_id": row.get(COLS.FRAME_ID.value),
                        "class": obj.get("class"),
                        "bbox_2d": obj.get("bbox_2d"),
                        "logit": obj.get("conf", 0.0),
                        "source": "fallback",
                    }
                )
        return (
            pl.DataFrame(out)
            if out
            else pl.DataFrame(
                schema={
                    "frame_id": pl.Int64,
                    "class": pl.Utf8,
                    "bbox_2d": pl.List(pl.Int64),
                    "logit": pl.Float64,
                    "source": pl.Utf8,
                }
            )
        )


def _map_class(raw: str) -> str | None:
    key = raw.lower()
    if "motorcycle" in key or "moto" in key:
        return "lane_split_moto"
    if "pedestrian" in key or "ped" in key:
        return "ped_cross"
    if "traffic light" in key or "red" in key:
        return "red_light"
    if key in {"car", "truck", "bus"}:
        return None
    return None
