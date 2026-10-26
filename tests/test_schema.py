"""VLMFinding validation, bbox range, none XOR findings, truncated flag."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from vsm.vlm.schema import ObjectClaim, Subfail, VLMFinding, WindowOutput


def _finding(**kwargs) -> VLMFinding:
    base = dict(
        frame_idx=0,
        objects=[ObjectClaim(cls="cut_in", bbox_2d=(10, 10, 40, 80), lane_position="adjacent")],
        claim="cut_in",
        supporting_frames=[0, 1],
        confidence=0.7,
        family="safety_maneuver",
        signal_state="unclear",
        attributes={},
    )
    base.update(kwargs)
    return VLMFinding(**base)


def test_bbox_rejects_1000() -> None:
    with pytest.raises(ValidationError):
        ObjectClaim(cls="car", bbox_2d=(0, 0, 1000, 10), lane_position="in_lane")


def test_bbox_accepts_999() -> None:
    obj = ObjectClaim(cls="car", bbox_2d=(0, 0, 999, 999), lane_position="in_lane")
    assert obj.bbox_2d[2] == 999


def test_none_xor_findings() -> None:
    with pytest.raises(ValidationError):
        WindowOutput(
            window_id="w",
            findings=[_finding()],
            none_=True,
            uncertain=False,
            parse_ok=True,
            backend="fallback",
            geometry_mode="B_mapless",
            provenance="grid",
        )


def test_truncated_sets_subfail() -> None:
    out = WindowOutput(
        window_id="w",
        findings=[],
        none_=True,
        uncertain=False,
        parse_ok=True,
        truncated=True,
        finish_reason="length",
        backend="api_vlm",
        geometry_mode="B_mapless",
        provenance="grid",
    )
    assert out.parse_subfail == Subfail.truncated
    assert out.parse_ok is False


def test_unknown_family_rejected() -> None:
    with pytest.raises(ValidationError):
        _finding(family="not_a_family")
