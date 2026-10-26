"""T0 rules. Mode A = BEV map; Mode B = bbox pixel-band, geometry_mode degraded."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Literal

import polars as pl
from shapely.geometry import Point

from vsm.config import REPO_ROOT, cfg_dict
from vsm.constants import COLS, GEOMETRY_A, GEOMETRY_B, IMAGE_H, IMAGE_W, WIN_COLS
from vsm.geometry.projection import pixel_to_ground_xy, qwen_bottom_center_px
from vsm.mining.closure import relative_closure, scale_slope_flag

_VEH = ("car", "truck", "bus", "trailer", "construction", "vehicle")
_PED = ("pedestrian", "human.pedestrian", "bicycle", "cycle")


def _objects(raw: Any) -> list[dict]:
    if not raw:
        return []
    if isinstance(raw, list):
        return raw
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        return []
    return payload if isinstance(payload, list) else []


def _is_vehicle(name: str) -> bool:
    return any(tok in name.lower() for tok in _VEH)


def _is_ped(name: str) -> bool:
    key = name.lower()
    if "motorcycle" in key:
        return False
    return any(tok in key for tok in _PED)


def _is_moto(name: str) -> bool:
    return "motorcycle" in name.lower()


def _ego_xy(ego_json: str) -> tuple[float, float] | None:
    try:
        payload = json.loads(ego_json)
    except json.JSONDecodeError:
        return None
    xyz = payload.get("xyz") or payload.get("translation")
    if not xyz or len(xyz) < 2:
        return None
    return float(xyz[0]), float(xyz[1])


Mode = Literal["A", "B"]
_RULES = cfg_dict("rules")
_JOIN = _RULES["join"]
JOIN_RULE_PED_V2 = str(_JOIN["pedestrian_cross"])
_STRADDLE = float(_JOIN["straddle_m"])
_INTRUSION_FRAC = float(_JOIN["intrusion_frac"])
_INTRUSION_N = int(_JOIN["intrusion_min_frames"])
_SIDEWALK_FRAC = float(_JOIN["sidewalk_frac"])
_MOTO_N = int(_JOIN["moto_min_frames"])
_BAND_LO = float(_JOIN["pixel_band_lo"])
_BAND_HI = float(_JOIN["pixel_band_hi"])
_CTR_LO = float(_JOIN["center_band_lo"])
_CTR_HI = float(_JOIN["center_band_hi"])


@dataclass
class Flag:
    cls: str
    window_id: str
    geometry_mode: str
    n_frames: int
    note: str = ""
    t0_support: str = "full"


@dataclass
class Rule:
    capabilities: list[str]
    fn: Callable[..., list[Flag]]
    family: str
    expected_fp_mode: str
    t0_support: str = "full"


def _parse_ego(raw: str) -> dict:
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        payload = {}
    if "rotation" not in payload:
        payload["rotation"] = [1.0, 0.0, 0.0, 0.0]
    if "translation" not in payload:
        xyz = payload.get("xyz") or [0.0, 0.0, 0.0]
        payload["translation"] = [float(xyz[0]), float(xyz[1]), 0.0]
    return payload


def _cx_norm(bbox: list) -> float:
    x1, _y1, x2, _y2 = [float(v) for v in bbox[:4]]
    return ((x1 + x2) / 2.0) / 1000.0


def _boxes(row: dict, pred: Callable[[str], bool]) -> list[list[float]]:
    out = []
    for obj in _objects(row.get(COLS.OBJECTS_JSON.value)):
        if pred(str(obj.get("class") or "")):
            bbox = obj.get("bbox_2d") or []
            if len(bbox) >= 4:
                out.append([float(v) for v in bbox[:4]])
    return out


def _ground_points(row: dict, pred: Callable[[str], bool], cs: dict | None) -> list[tuple[float, float]]:
    ego = _parse_ego(str(row.get(COLS.EGO_POSE_JSON.value) or "{}"))
    pts: list[tuple[float, float]] = []
    if cs is None:
        return pts
    for bbox in _boxes(row, pred):
        u, v = qwen_bottom_center_px(bbox, IMAGE_W, IMAGE_H)
        xy = pixel_to_ground_xy(u, v, ego, cs)
        if xy is not None:
            pts.append(xy)
    return pts


def _nearest_lane(pt: Point, lanes: list) -> Any | None:
    if not lanes:
        return None
    return min(lanes, key=lambda p: p.distance(pt) if not p.contains(pt) else 0.0)


def rule_lane_split_moto(boxes: list[dict], mode: Mode, **ctx: Any) -> list[Flag]:
    """Straddle ±0.75 m (A) or pixel-band (B), ≥3 frames, moving."""

    window_id = str(ctx.get("window_id") or "")
    lanes = (ctx.get("layers") or {}).get("lane") or []
    cs = ctx.get("cs")
    n_hit = 0
    cxs: list[float] = []
    for row in boxes:
        if mode == "A" and lanes and cs is not None:
            pts = _ground_points(row, _is_moto, cs)
            for x, y in pts:
                lane = _nearest_lane(Point(x, y), lanes)
                if lane is None:
                    continue
                d = float(lane.exterior.distance(Point(x, y)))
                # straddle: near the painted boundary, not deep inside a lane
                if d <= _STRADDLE:
                    n_hit += 1
                    break
        else:
            moto = _boxes(row, _is_moto)
            if any(_BAND_LO <= _cx_norm(b) <= _BAND_HI for b in moto):
                n_hit += 1
                cxs.extend(_cx_norm(b) for b in moto)
    moving = (max(cxs) - min(cxs) > 0.02) if len(cxs) >= 2 else True
    if n_hit >= _MOTO_N and moving:
        geom = GEOMETRY_A if mode == "A" else GEOMETRY_B
        return [Flag("lane_split_moto", window_id, geom, n_hit)]
    return []


def rule_lane_intrusion(boxes: list[dict], mode: Mode, **ctx: Any) -> list[Flag]:
    """bbox ∩ ego-lane ≥25% for ≥2 frames — positive predicate."""

    window_id = str(ctx.get("window_id") or "")
    lanes = (ctx.get("layers") or {}).get("lane") or []
    cs = ctx.get("cs")
    n_hit = 0
    for row in boxes:
        if mode == "A" and lanes and cs is not None:
            ego_xy = _ego_xy(str(row.get(COLS.EGO_POSE_JSON.value) or "{}"))
            ego_lane = _nearest_lane(Point(*ego_xy), lanes) if ego_xy else None
            if ego_lane is None:
                continue
            pts = _ground_points(row, lambda n: _is_vehicle(n) or _is_moto(n), cs)
            inside = sum(1 for x, y in pts if ego_lane.contains(Point(x, y)))
            if pts and inside / len(pts) >= _INTRUSION_FRAC:
                n_hit += 1
        else:
            objs = _boxes(row, lambda n: _is_vehicle(n) or _is_moto(n))
            if any(_CTR_LO <= _cx_norm(b) <= _CTR_HI for b in objs):
                n_hit += 1
    if n_hit >= _INTRUSION_N:
        geom = GEOMETRY_A if mode == "A" else GEOMETRY_B
        return [Flag("lane_intrusion", window_id, geom, n_hit)]
    return []


def rule_sidewalk_mount(boxes: list[dict], mode: Mode, **ctx: Any) -> list[Flag]:
    """≥50% of vehicle ground-points on walkway (A) or off-road pixel band (B)."""

    window_id = str(ctx.get("window_id") or "")
    walks = (ctx.get("layers") or {}).get("sidewalk") or []
    cs = ctx.get("cs")
    n_hit = 0
    for row in boxes:
        vehs = _boxes(row, _is_vehicle)
        if not vehs:
            continue
        if mode == "A" and walks and cs is not None:
            pts = _ground_points(row, _is_vehicle, cs)
            if not pts:
                continue
            on = sum(1 for x, y in pts if any(w.contains(Point(x, y)) for w in walks))
            if on / len(pts) >= _SIDEWALK_FRAC:
                n_hit += 1
        else:
            if any(_cx_norm(b) < 0.12 or _cx_norm(b) > 0.88 for b in vehs):
                n_hit += 1
    if n_hit >= 1:
        geom = GEOMETRY_A if mode == "A" else GEOMETRY_B
        return [Flag("sidewalk_mount", window_id, geom, n_hit)]
    return []


def rule_cut_in(boxes: list[dict], mode: Mode, **ctx: Any) -> list[Flag]:
    """Lateral enter AND closure slope AND ego-lane occupancy ≥25% ≥2 frames. Not 2D-only."""

    window_id = str(ctx.get("window_id") or "")
    lanes = (ctx.get("layers") or {}).get("lane") or []
    cs = ctx.get("cs")
    early: list[float] = []
    late: list[float] = []
    seq: list[list[list[float]]] = []
    half = max(len(boxes) // 2, 1)
    n_lane = 0
    for i, row in enumerate(boxes):
        vehs = _boxes(row, _is_vehicle)
        seq.append(vehs)
        for b in vehs:
            cx = _cx_norm(b)
            if i < half:
                early.append(cx)
            else:
                late.append(cx)
        if mode == "A" and lanes and cs is not None:
            ego_xy = _ego_xy(str(row.get(COLS.EGO_POSE_JSON.value) or "{}"))
            ego_lane = _nearest_lane(Point(*ego_xy), lanes) if ego_xy else None
            if ego_lane is not None:
                pts = _ground_points(row, _is_vehicle, cs)
                inside = sum(1 for x, y in pts if ego_lane.contains(Point(x, y)))
                if pts and inside / len(pts) >= _INTRUSION_FRAC:
                    n_lane += 1
        elif any(_CTR_LO <= _cx_norm(b) <= _CTR_HI for b in vehs):
            n_lane += 1
    if not early or not late:
        return []
    lateral = any(c < 0.28 or c > 0.72 for c in early) and any(_CTR_LO <= c <= _CTR_HI for c in late)
    closed = scale_slope_flag(seq).fired
    if lateral and closed and n_lane >= _INTRUSION_N:
        geom = GEOMETRY_A if mode == "A" else GEOMETRY_B
        return [Flag("cut_in", window_id, geom, n_lane, note="tight_v2_slope_and_lane")]
    return []


def rule_lead_braking(boxes: list[dict], mode: Mode, **ctx: Any) -> list[Flag]:
    """Closure slope on 1/h of the center-band lead. relative_closure subtracts ego odometry."""

    del mode
    window_id = str(ctx.get("window_id") or "")
    seq: list[list[list[float]]] = []
    traj: list[tuple[float, float]] = []
    for row in boxes:
        leads = [b for b in _boxes(row, _is_vehicle) if _CTR_LO <= _cx_norm(b) <= _CTR_HI]
        seq.append(leads)
        xy = _ego_xy(str(row.get(COLS.EGO_POSE_JSON.value) or "{}"))
        if xy:
            traj.append(xy)
    flag = scale_slope_flag(seq)
    rel = relative_closure(seq, traj)
    if flag.fired and rel < _JOIN["lead_slope_max"]:
        return [Flag("lead_braking", window_id, GEOMETRY_B, flag.n, note=f"slope={flag.slope}")]
    return []


def rule_pedestrian_cross(boxes: list[dict], mode: Mode, **ctx: Any) -> list[Flag]:
    """v2: ego-path buffer ∩ crossing AND ped ground point inside that polygon."""

    window_id = str(ctx.get("window_id") or "")
    crossings = (ctx.get("layers") or {}).get("ped_crossing") or []
    cs = ctx.get("cs")
    pts = []
    for row in boxes:
        xy = _ego_xy(str(row.get(COLS.EGO_POSE_JSON.value) or "{}"))
        if xy is not None:
            pts.append(xy)
    hit_map = False
    if len(pts) >= 2 and crossings:
        from shapely.geometry import LineString

        path = LineString(pts)
        if path.length > 0:
            hit_map = any(path.buffer(float(_JOIN["ego_path_buffer_m"])).intersects(poly) for poly in crossings)
    n_hit = 0
    for row in boxes:
        if mode == "A" and crossings and cs is not None:
            gpts = _ground_points(row, _is_ped, cs)
            if any(any(c.contains(Point(x, y)) for c in crossings) for x, y in gpts):
                n_hit += 1
        else:
            peds = _boxes(row, _is_ped)
            if any(b[3] > 700 for b in peds):
                n_hit += 1
    if hit_map and n_hit >= 1:
        geom = GEOMETRY_A if mode == "A" else GEOMETRY_B
        return [Flag("pedestrian_cross", window_id, geom, n_hit, note=JOIN_RULE_PED_V2)]
    return []


def rule_oncoming(boxes: list[dict], mode: Mode, **ctx: Any) -> list[Flag]:
    """WRONG-WAY-ONLY. T0 support thin — likely coverage-only. No heading on 2D boxes."""

    del boxes, mode
    window_id = str(ctx.get("window_id") or "")
    return [
        Flag(
            "oncoming",
            window_id,
            GEOMETRY_B,
            0,
            note="T0 support thin — likely coverage-only",
            t0_support="thin",
        )
    ][:0]  # never auto-fire; coverage-only


def rule_turn_lane_drift(boxes: list[dict], mode: Mode, **ctx: Any) -> list[Flag]:
    """Dedicated turn-lane going straight. T0 support thin — likely coverage-only."""

    del boxes, mode, ctx
    return []


def rule_red_light_violation(boxes: list[dict], mode: Mode, **ctx: Any) -> list[Flag]:
    """Presence-only: lamp in objects_json. Do not score. No signal GT."""

    del mode
    window_id = str(ctx.get("window_id") or "")
    n = 0
    for row in boxes:
        for obj in _objects(row.get(COLS.OBJECTS_JSON.value)):
            name = str(obj.get("class") or "").lower()
            if "traffic_light" in name or "traffic light" in name:
                n += 1
    if n:
        return [
            Flag(
                "red_light_violation",
                window_id,
                GEOMETRY_B,
                n,
                note="presence_only — do not score",
                t0_support="presence_only",
            )
        ]
    return []


RULES: dict[str, Rule] = {
    "lane_split_moto": Rule(["map", "calibrated_ego"], rule_lane_split_moto, "safety_spatial", "straddle_fp"),
    "lane_intrusion": Rule(["map", "calibrated_ego"], rule_lane_intrusion, "safety_spatial", "legal_occupant"),
    "sidewalk_mount": Rule(["map", "calibrated_ego"], rule_sidewalk_mount, "safety_spatial", "driveway"),
    "cut_in": Rule(["calibrated_ego"], rule_cut_in, "safety_maneuver", "parallel"),
    "lead_braking": Rule(["calibrated_ego"], rule_lead_braking, "safety_maneuver", "ego_brake"),
    "pedestrian_cross": Rule(["map", "calibrated_ego"], rule_pedestrian_cross, "safety_spatial", "curb_ped"),
    "oncoming": Rule(["map", "calibrated_ego"], rule_oncoming, "safety_maneuver", "legal_oncoming", "thin"),
    "turn_lane_drift": Rule(["map", "calibrated_ego"], rule_turn_lane_drift, "safety_maneuver", "legal_turn", "thin"),
    "red_light_violation": Rule([], rule_red_light_violation, "compliance", "lamp_only", "presence_only"),
}


def choose_mode(available: set[str], rule: Rule) -> Mode:
    """Return A if every required capability is present, else B (degraded)."""

    if set(rule.capabilities).issubset(available):
        return "A"
    return "B"


def load_calibrated_sensors(mini_root: Path | None = None) -> dict[str, dict]:
    """Index calibrated_sensor.json. Does not touch sample_annotation."""

    from vsm.ingest.convert import _find_version_dir, _load_table

    root = Path(mini_root or REPO_ROOT / "data" / "raw" / "v1.0-mini")
    try:
        version = _find_version_dir(root)
        rows = _load_table(version, "calibrated_sensor")
    except FileNotFoundError:
        return {}
    return {str(r["token"]): r for r in rows}


def run_t0_window(
    window: dict,
    frames: list[dict],
    *,
    layers: dict | None,
    cs: dict | None,
    available: set[str],
) -> list[Flag]:
    """Run every T0 rule on one window. Never reads sample_annotation."""

    flags: list[Flag] = []
    wid = str(window.get(WIN_COLS.WINDOW_ID.value) or "")
    ctx = {"window_id": wid, "layers": layers or {}, "cs": cs, "manifest": {}}
    for name, rule in RULES.items():
        mode = choose_mode(available, rule)
        fired = rule.fn(frames, mode, **ctx)
        status = str((_RULES.get("rules") or {}).get(name, {}).get("status") or "enabled")
        if status == "disabled":
            for flag in fired:
                flag.t0_support = "disabled"
                flag.note = (flag.note + " " if flag.note else "") + "T0-disabled"
        flags.extend(fired)
    return flags


def run_t0(
    windows: pl.DataFrame,
    frames: pl.DataFrame,
    *,
    map_root: Path | None = None,
    mini_root: Path | None = None,
    manifest: dict | None = None,
) -> pl.DataFrame:
    """Run T0 Mode A/B on every window. Output one row per flag."""

    del manifest  # T0 must not consult sample_annotation even if a caller passes nusc
    from vsm.geometry.map_layers import load_map_layers

    cs_ix = load_calibrated_sensors(mini_root)
    layer_cache: dict[str, Any] = {}
    rows: list[dict] = []
    frame_ix = {int(r[COLS.FRAME_ID.value]): r for r in frames.iter_rows(named=True)}
    for w in windows.iter_rows(named=True):
        ids = list(w.get(WIN_COLS.FRAME_IDS.value) or [])
        subset = [frame_ix[i] for i in ids if i in frame_ix]
        if not subset:
            continue
        map_name = str(subset[0].get(COLS.MAP_NAME.value) or "")
        if map_name and map_name not in layer_cache:
            try:
                layer_cache[map_name] = load_map_layers(map_name, map_root)
            except (FileNotFoundError, RuntimeError, ValueError, KeyError):
                layer_cache[map_name] = {}
        layers = layer_cache.get(map_name) or {}
        cs_tok = str(subset[0].get("calibrated_sensor_token") or "")
        cs = cs_ix.get(cs_tok)
        available: set[str] = set()
        if layers.get("lane") or layers.get("ped_crossing") or layers.get("sidewalk"):
            available.add("map")
        if cs is not None:
            available.add("calibrated_ego")
        for flag in run_t0_window(w, subset, layers=layers, cs=cs, available=available):
            rows.append(
                {
                    "window_id": flag.window_id,
                    "class": flag.cls,
                    "geometry_mode": flag.geometry_mode,
                    "n_frames": flag.n_frames,
                    "note": flag.note,
                    "t0_support": flag.t0_support,
                    "t0_enabled": flag.t0_support != "disabled",
                    "backend": "t0",
                }
            )
    return (
        pl.DataFrame(rows)
        if rows
        else pl.DataFrame(
            schema={
                "window_id": pl.Utf8,
                "class": pl.Utf8,
                "geometry_mode": pl.Utf8,
                "n_frames": pl.Int64,
                "note": pl.Utf8,
                "t0_support": pl.Utf8,
                "t0_enabled": pl.Boolean,
                "backend": pl.Utf8,
            }
        )
    )


class T0Backend:
    """Cascade adapter. Loads map+cs once; never reads sample_annotation."""

    name = "t0"

    def __init__(self, map_root: Path | None = None, mini_root: Path | None = None) -> None:
        from vsm.geometry.map_layers import load_map_layers

        self._layers: dict[str, Any] = {}
        self._cs_ix = load_calibrated_sensors(mini_root)
        self._map_root = map_root
        self._load_map = load_map_layers

    def mine_window(self, frames: Any, fps: float, prompts: dict[str, str]) -> Any:
        del fps, prompts
        from vsm.constants import PROVENANCE_GRID
        from vsm.vlm.schema import ObjectClaim, VLMFinding, WindowOutput, class_family

        rows = frames.to_dicts() if isinstance(frames, pl.DataFrame) else list(frames)
        if not rows:
            return WindowOutput(
                window_id="t0-empty",
                findings=[],
                none_=True,
                uncertain=False,
                parse_ok=True,
                backend=self.name,
                geometry_mode=GEOMETRY_B,
                provenance=PROVENANCE_GRID,
                model_id="t0",
                detector_supported=False,
            )
        map_name = str(rows[0].get(COLS.MAP_NAME.value) or "")
        if map_name and map_name not in self._layers:
            try:
                self._layers[map_name] = self._load_map(map_name, self._map_root)
            except (FileNotFoundError, RuntimeError, ValueError, KeyError):
                self._layers[map_name] = {}
        layers = self._layers.get(map_name) or {}
        cs = self._cs_ix.get(str(rows[0].get("calibrated_sensor_token") or ""))
        available: set[str] = set()
        if layers.get("lane") or layers.get("ped_crossing") or layers.get("sidewalk"):
            available.add("map")
        if cs is not None:
            available.add("calibrated_ego")
        scene = rows[0].get(COLS.SCENE_NAME.value)
        cam = rows[0].get(COLS.CAMERA.value)
        ts0 = rows[0].get(COLS.TS_US.value)
        window = {
            WIN_COLS.WINDOW_ID.value: f"{scene}_{cam}_{ts0}",
            WIN_COLS.FRAME_IDS.value: [r.get(COLS.FRAME_ID.value) for r in rows],
        }
        flags = run_t0_window(window, rows, layers=layers, cs=cs, available=available)
        findings = []
        for flag in flags:
            findings.append(
                VLMFinding(
                    frame_idx=int(rows[0].get(COLS.FRAME_ID.value) or 0),
                    objects=[ObjectClaim(cls=flag.cls, bbox_2d=(0, 0, 1, 1), lane_position="unknown")],
                    claim=flag.cls,
                    supporting_frames=[int(rows[0].get(COLS.FRAME_ID.value) or 0)],
                    confidence=1.0,
                    family=class_family(flag.cls),
                    signal_state="unclear",
                    attributes={"t0_support": flag.t0_support, "note": flag.note, "n_frames": flag.n_frames},
                )
            )
        geom = flags[0].geometry_mode if flags else GEOMETRY_B
        return WindowOutput(
            window_id=window[WIN_COLS.WINDOW_ID.value],
            findings=findings,
            none_=len(findings) == 0,
            uncertain=False,
            parse_ok=True,
            backend=self.name,
            geometry_mode=geom,
            provenance=PROVENANCE_GRID,
            model_id="t0",
            detector_supported=False,
        )
