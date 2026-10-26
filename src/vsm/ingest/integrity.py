"""Ingestion gate: isfinite, quat-norm, referential, 4 ts-invariants. Fail → quarantine."""

from __future__ import annotations

import json
from dataclasses import dataclass, field

import numpy as np
import polars as pl

from vsm.constants import COLS, QUAT_NORM_EPS, TS_GAP_EPS_S, TS_GAP_S, Invariant


class IntegrityError(RuntimeError):
    """Raised when the gate must not return a silent False."""


@dataclass
class GateReport:
    """Per-scene quarantine report. `ok` is true only when no scene is quarantined."""

    ok: bool
    quarantined_scenes: list[str] = field(default_factory=list)
    findings: list[dict] = field(default_factory=list)

    def raise_if_poison(self) -> None:
        poisons = [f for f in self.findings if f.get("invariant") == Invariant.NAN_POISON.value]
        if poisons:
            raise IntegrityError(f"NaN poison detected: {poisons}")


def _as_xyz_heading(raw: str | dict | list) -> tuple[float, float, float]:
    if isinstance(raw, str):
        raw = json.loads(raw)
    if isinstance(raw, dict):
        trans = raw.get("translation") or raw.get("xyz") or [raw.get("x"), raw.get("y"), raw.get("z")]
        heading = raw.get("heading", raw.get("yaw", 0.0))
        return float(trans[0]), float(trans[1]), float(heading)
    if isinstance(raw, list) and len(raw) >= 3:
        return float(raw[0]), float(raw[1]), float(raw[2])
    raise IntegrityError(f"unparseable ego_pose_json: {raw!r}")


def _quat_norm(raw: object) -> float:
    if isinstance(raw, str):
        raw = json.loads(raw)
    if isinstance(raw, dict):
        raw = raw.get("rotation") or raw.get("quaternion")
    arr = np.asarray(raw, dtype=np.float64).reshape(-1)
    if arr.size != 4:
        return float("nan")
    return float(np.linalg.norm(arr))


def run_ingestion_gate(frames: pl.DataFrame) -> GateReport:
    """Scan isfinite/quat/referential/ts-invariants. Quarantine scenes; do not auto-repair."""

    findings: list[dict] = []
    quarantined: set[str] = set()
    required = [
        COLS.FRAME_ID.value,
        COLS.SAMPLE_TOKEN.value,
        COLS.SCENE_TOKEN.value,
        COLS.CAMERA.value,
        COLS.TS_US.value,
        COLS.IS_KEYFRAME.value,
        COLS.EGO_POSE_JSON.value,
    ]
    missing = [c for c in required if c not in frames.columns]
    if missing:
        raise IntegrityError(f"frames missing columns {missing}")

    # isfinite translation + referential tokens
    for row in frames.iter_rows(named=True):
        scene = str(row[COLS.SCENE_TOKEN.value])
        try:
            x, y, h = _as_xyz_heading(row[COLS.EGO_POSE_JSON.value])
        except (TypeError, ValueError, json.JSONDecodeError, IntegrityError) as exc:
            findings.append({"invariant": Invariant.REF_INTEGRITY.value, "scene": scene, "detail": str(exc)})
            quarantined.add(scene)
            continue
        if not np.isfinite([x, y, h]).all():
            findings.append(
                {
                    "invariant": Invariant.FINITE_TRANSLATION.value,
                    "scene": scene,
                    "detail": f"non-finite pose {(x, y, h)}",
                }
            )
            quarantined.add(scene)
            continue
        rot = None
        pose = row[COLS.EGO_POSE_JSON.value]
        if isinstance(pose, str):
            try:
                parsed = json.loads(pose)
            except json.JSONDecodeError:
                parsed = None
        else:
            parsed = pose
        if isinstance(parsed, dict) and parsed.get("rotation") is not None:
            rot = parsed.get("rotation")
            nrm = _quat_norm(rot)
            if not np.isfinite(nrm):
                findings.append(
                    {"invariant": Invariant.FINITE_ROTATION.value, "scene": scene, "detail": "non-finite quat"}
                )
                quarantined.add(scene)
            elif abs(nrm - 1.0) > QUAT_NORM_EPS:
                findings.append(
                    {
                        "invariant": Invariant.QUAT_UNIT.value,
                        "scene": scene,
                        "detail": f"quat norm {nrm} outside 1±{QUAT_NORM_EPS}",
                    }
                )
                quarantined.add(scene)

    # 4 ts-invariants per (scene, camera)
    grouped = frames.sort([COLS.SCENE_TOKEN.value, COLS.CAMERA.value, COLS.TS_US.value])
    for (scene, cam), grp in grouped.group_by([COLS.SCENE_TOKEN.value, COLS.CAMERA.value], maintain_order=True):
        ts = grp[COLS.TS_US.value].to_numpy()
        if ts.size == 0:
            continue
        if not np.all(np.diff(ts.astype(np.int64)) > 0):
            findings.append(
                {
                    "invariant": Invariant.TS_STRICT_INC.value,
                    "scene": str(scene),
                    "camera": str(cam),
                    "detail": "timestamps not strictly increasing",
                }
            )
            quarantined.add(str(scene))
        if int(grp[COLS.TS_US.value].n_unique()) != grp.height:
            findings.append(
                {
                    "invariant": Invariant.TS_NO_DUP.value,
                    "scene": str(scene),
                    "camera": str(cam),
                    "detail": "duplicate timestamps",
                }
            )
            quarantined.add(str(scene))
        if ts.size >= 2:
            gaps_s = np.diff(ts.astype(np.float64)) / 1e6
            if not np.all(np.abs(gaps_s - TS_GAP_S) <= TS_GAP_EPS_S):
                findings.append(
                    {
                        "invariant": Invariant.TS_GAP.value,
                        "scene": str(scene),
                        "camera": str(cam),
                        "detail": f"gaps {gaps_s.tolist()} vs {TS_GAP_S}±{TS_GAP_EPS_S}s",
                    }
                )
                quarantined.add(str(scene))
        flags = grp[COLS.IS_KEYFRAME.value].to_list()
        if not all(bool(v) for v in flags):
            findings.append(
                {
                    "invariant": Invariant.KEYFRAME_FLAG.value,
                    "scene": str(scene),
                    "camera": str(cam),
                    "detail": "non-keyframe in keyframe table",
                }
            )
            quarantined.add(str(scene))

    report = GateReport(ok=not quarantined, quarantined_scenes=sorted(quarantined), findings=findings)
    report.raise_if_poison()
    return report


def nan_poison_test() -> None:
    """Inject NaN into a one-row table; the gate must raise or flag, never return ok=True."""

    poison = pl.DataFrame(
        {
            COLS.FRAME_ID.value: [0],
            COLS.SAMPLE_TOKEN.value: ["tok"],
            COLS.SCENE_TOKEN.value: ["scene-poison"],
            COLS.CAMERA.value: ["CAM_FRONT"],
            COLS.TS_US.value: [0],
            COLS.IS_KEYFRAME.value: [True],
            COLS.EGO_POSE_JSON.value: [json.dumps({"translation": [float("nan"), 0.0, 0.0], "heading": 0.0})],
        }
    )
    report = run_ingestion_gate(poison)
    if report.ok or not report.quarantined_scenes:
        raise IntegrityError("nan_poison_test: gate returned ok/empty quarantine — silent False is forbidden")
    kinds = {f["invariant"] for f in report.findings}
    if Invariant.FINITE_TRANSLATION.value not in kinds:
        raise IntegrityError(f"nan_poison_test: expected finite-translation flag, got {report.findings}")
