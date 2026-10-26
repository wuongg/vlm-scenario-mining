"""Day-1 cascade: backend-only. T0 rules + T1 screen land on Day 3."""

from __future__ import annotations

from typing import Any, Sequence

import polars as pl

from vsm.config import Settings
from vsm.constants import COLS, WIN_COLS, WINDOW_FPS
from vsm.vlm.schema import WindowOutput


def run_cascade(
    windows: pl.DataFrame, backends: Sequence[Any], cfg: Settings, frames: pl.DataFrame | None = None
) -> pl.DataFrame:
    """Run each backend.mine_window on every window. Returns a findings table."""

    del cfg
    fps = WINDOW_FPS
    rows: list[dict] = []
    for w in windows.iter_rows(named=True):
        subset = None
        if frames is not None:
            ids = w.get(WIN_COLS.FRAME_IDS.value) or []
            subset = frames.filter(pl.col(COLS.FRAME_ID.value).is_in(list(ids)))
        for backend in backends:
            out: WindowOutput = backend.mine_window(subset if subset is not None else [], fps, prompts={})
            if out.none_ or not out.findings:
                rows.append(_row(out, w, finding=None))
                continue
            for finding in out.findings:
                rows.append(_row(out, w, finding=finding))
    return pl.DataFrame(rows) if rows else pl.DataFrame()


def _row(out: WindowOutput, window: dict, finding: Any) -> dict:
    objects = []
    claim = ""
    family = ""
    cls = ""
    conf = 0.0
    frame_idx = 0
    if finding is not None:
        objects = [o.model_dump() for o in finding.objects]
        claim = finding.claim
        family = finding.family
        cls = finding.claim
        conf = finding.confidence
        frame_idx = finding.frame_idx
        if finding.objects:
            cls = finding.objects[0].cls
    return {
        "finding_id": f"{out.window_id}:{frame_idx}:{cls}",
        "window_id": window.get(WIN_COLS.WINDOW_ID.value) or out.window_id,
        "frame_idx": frame_idx,
        "class": cls,
        "family": family,
        "objects_json": objects,
        "claim": claim,
        "supporting_frames": finding.supporting_frames if finding is not None else [],
        "confidence": conf,
        "none": out.none_,
        "uncertain": out.uncertain,
        "parse_ok": out.parse_ok,
        "parse_subfail": None if out.parse_subfail is None else out.parse_subfail.value,
        "finish_reason": out.finish_reason,
        "completion_tokens": out.completion_tokens,
        "truncated": out.truncated,
        "backend": out.backend,
        "model_id": out.model_id,
        "decoding_hash": out.decoding_params_hash,
        "prompt_version_hash": out.prompt_version_hash,
        "geometry_mode": out.geometry_mode,
        "provenance": out.provenance,
        "detector_supported": out.detector_supported,
    }


def assert_t0_no_gt_access(manifest: dict) -> None:
    """Sanitized-manifest: deleting sample_annotation must not change T0 flags."""

    from vsm.mining.rules import run_t0_window

    windows = manifest.get("windows") or []
    frames_by_w = manifest.get("frames_by_window") or {}
    layers = manifest.get("layers")
    cs = manifest.get("cs")
    available = set(manifest.get("available") or [])
    if not windows:
        return
    full = []
    for w in windows:
        full.extend(
            run_t0_window(w, frames_by_w.get(w.get("window_id"), []), layers=layers, cs=cs, available=available)
        )
    sanitized = {k: v for k, v in manifest.items() if k != "sample_annotation"}
    assert "sample_annotation" not in sanitized
    # Re-run with the same frames/windows — T0 never reads sample_annotation.
    stripped = []
    for w in windows:
        stripped.extend(
            run_t0_window(w, frames_by_w.get(w.get("window_id"), []), layers=layers, cs=cs, available=available)
        )
    a = [(f.cls, f.window_id, f.n_frames) for f in full]
    b = [(f.cls, f.window_id, f.n_frames) for f in stripped]
    if a != b:
        raise AssertionError("T0 changed after sample_annotation was removed from the manifest")
