"""Pydantic contracts for findings and window outputs. Numbers from configs, not literals in callers."""

from __future__ import annotations

from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator, model_validator

from vsm.constants import CLASS_IDS, FAMILIES, QWEN_BBOX_MAX

SignalState = Literal["on", "off", "unclear"]
LanePosition = Literal["in_lane", "adjacent", "off_lane", "unknown"]


class Subfail(str, Enum):
    refused = "refused"
    truncated = "truncated"
    loop_detected = "loop_detected"
    valid_but_empty = "valid_but_empty"
    schema_violation = "schema_violation"


class ObjectClaim(BaseModel):
    """One object mention. bbox_2d is inclusive-exclusive in [0, 1000)."""

    cls: str
    bbox_2d: tuple[int, int, int, int]
    lane_position: LanePosition = "unknown"

    @field_validator("bbox_2d")
    @classmethod
    def _bbox_in_qwen_range(cls, value: tuple[int, int, int, int]) -> tuple[int, int, int, int]:
        x1, y1, x2, y2 = value
        lim = QWEN_BBOX_MAX
        for v in (x1, y1, x2, y2):
            if v < 0 or v >= lim:
                raise ValueError(f"bbox_2d values must be in [0, {lim}), got {value}")
        if x2 < x1 or y2 < y1:
            raise ValueError(f"bbox_2d must be ordered (x1,y1,x2,y2), got {value}")
        return value


class VLMFinding(BaseModel):
    """One structured claim inside a window."""

    frame_idx: int
    objects: list[ObjectClaim] = Field(default_factory=list)
    claim: str
    supporting_frames: list[int] = Field(default_factory=list)
    confidence: float = Field(ge=0.0, le=1.0)
    family: str
    signal_state: SignalState = "unclear"
    attributes: dict[str, Any] = Field(default_factory=dict)

    @field_validator("family")
    @classmethod
    def _known_family(cls, value: str) -> str:
        if value not in FAMILIES:
            raise ValueError(f"unknown family {value!r}; expected one of {list(FAMILIES)}")
        return value


class WindowOutput(BaseModel):
    """Backend output for one window. none_ is the schema field (none is reserved)."""

    window_id: str
    findings: list[VLMFinding] = Field(default_factory=list)
    none_: bool = False
    uncertain: bool = False
    parse_ok: bool = True
    parse_subfail: Subfail | None = None
    finish_reason: str = "stop"
    completion_tokens: int = 0
    truncated: bool = False
    prompt_version_hash: str = "day1-none"
    backend: str
    geometry_mode: str
    provenance: str
    model_id: str = ""
    decoding_params_hash: str = ""
    detector_supported: bool = False

    @model_validator(mode="after")
    def _none_xor_findings(self) -> WindowOutput:
        if self.none_ and self.findings:
            raise ValueError("none_ cannot be combined with non-empty findings")
        if self.truncated and self.parse_subfail is None:
            self.parse_subfail = Subfail.truncated
        if self.parse_subfail is not None:
            self.parse_ok = False
        return self


def class_family(cls_id: str) -> str:
    """Return the family name for a taxonomy class id."""

    for fam, members in FAMILIES.items():
        if cls_id in members:
            return fam
    if cls_id in CLASS_IDS:
        return "safety_maneuver"
    raise ValueError(f"unknown class {cls_id!r}")
