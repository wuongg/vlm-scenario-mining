"""Prevalence audit, GT list, labeling-session helpers. Headline freezes from this module."""

from __future__ import annotations

import hashlib
import json
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

import polars as pl

from vsm.config import REPO_ROOT, cfg_dict
from vsm.constants import CLASS_IDS, COLS, WIN_COLS

_EVAL = cfg_dict("eval")
_MIN_PER_MAP = int(_EVAL["allocation"]["min_windows_per_map"])
_OUT = int(_EVAL["allocation"]["out_of_headline_if_positives_lt"])

JOIN_RULE_PED = str(cfg_dict("rules")["join"]["pedestrian_cross"])
JOIN_RULE_PED_V1 = "ped_cross_v1_crossing_layer_ego_path_and_ped_object"
JOIN_RULE_CUTIN = "cut_in_v1_lateral_enter_center_band"
JOIN_RULE_LEAD = "lead_braking_v1_invh_slope"
HEADLINE_CANDIDATES = ("pedestrian_cross", "lead_braking", "cut_in")
VEHICLE_TOKENS = ("car", "truck", "bus", "trailer", "construction", "vehicle")
PED_TOKENS = ("pedestrian", "human.pedestrian", "bicycle", "cycle")


@dataclass
class PrevalenceTable:
    rows: list[dict] = field(default_factory=list)
    out_of_headline: list[str] = field(default_factory=list)
    headline: list[str] = field(default_factory=list)
    notes: dict[str, str] = field(default_factory=dict)

    def to_dataframe(self) -> pl.DataFrame:
        return (
            pl.DataFrame(self.rows)
            if self.rows
            else pl.DataFrame(
                schema={
                    "class": pl.Utf8,
                    "map": pl.Utf8,
                    "light": pl.Utf8,
                    "n_windows": pl.Int64,
                    "n_pos": pl.Int64,
                    "headline": pl.Boolean,
                    "map_support_ok": pl.Boolean,
                }
            )
        )


def _objects(raw: str | None) -> list[dict]:
    if not raw:
        return []
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        return []
    return payload if isinstance(payload, list) else []


def _is_vehicle(name: str) -> bool:
    key = name.lower()
    return any(tok in key for tok in VEHICLE_TOKENS)


def _is_ped(name: str) -> bool:
    key = name.lower()
    if "motorcycle" in key:
        return False
    return any(tok in key for tok in PED_TOKENS)


def _is_moto(name: str) -> bool:
    return "motorcycle" in name.lower() or "cycle.motorcycle" in name.lower()


def _bbox_cx_h(bbox: list) -> tuple[float, float]:
    x1, y1, x2, y2 = [float(v) for v in bbox[:4]]
    return (x1 + x2) / 2.0, max(y2 - y1, 1.0)


def _lin_slope(xs: list[float], ys: list[float]) -> float | None:
    n = len(xs)
    if n < 4:
        return None
    mx = sum(xs) / n
    my = sum(ys) / n
    den = sum((x - mx) ** 2 for x in xs)
    if den < 1e-12:
        return None
    return sum((x - mx) * (y - my) for x, y in zip(xs, ys, strict=False)) / den


def _window_frames(frames: pl.DataFrame, window: dict) -> pl.DataFrame:
    ids = list(window.get(WIN_COLS.FRAME_IDS.value) or [])
    return frames.filter(pl.col(COLS.FRAME_ID.value).is_in(ids)).sort(COLS.TS_US.value)


def _ego_xy(ego_json: str) -> tuple[float, float] | None:
    try:
        payload = json.loads(ego_json)
    except json.JSONDecodeError:
        return None
    xyz = payload.get("xyz") or payload.get("translation")
    if not xyz or len(xyz) < 2:
        return None
    return float(xyz[0]), float(xyz[1])


def _window_has_ped(subset: pl.DataFrame) -> bool:
    for raw in subset[COLS.OBJECTS_JSON.value].to_list():
        for obj in _objects(raw):
            if _is_ped(str(obj.get("class") or "")):
                return True
    return False


def _window_moto_frames(subset: pl.DataFrame) -> int:
    n = 0
    for raw in subset[COLS.OBJECTS_JSON.value].to_list():
        if any(_is_moto(str(obj.get("class") or "")) for obj in _objects(raw)):
            n += 1
    return n


def _cs_index() -> dict[str, dict]:
    from vsm.ingest.convert import _find_version_dir, _load_table

    try:
        version = _find_version_dir(REPO_ROOT / "data" / "raw" / "v1.0-mini")
        return {str(r["token"]): r for r in _load_table(version, "calibrated_sensor")}
    except FileNotFoundError:
        return {}


def _ped_ground_in_crossing(row: dict, crossings: list, cs: dict | None) -> bool:
    """v2 predicate: unprojected ped feet sit inside a ped_crossing polygon."""

    from shapely.geometry import Point

    from vsm.constants import IMAGE_H, IMAGE_W
    from vsm.geometry.projection import pixel_to_ground_xy, qwen_bottom_center_px

    if not crossings or cs is None:
        return False
    try:
        ego = json.loads(str(row.get(COLS.EGO_POSE_JSON.value) or "{}"))
    except json.JSONDecodeError:
        return False
    if "rotation" not in ego:
        return False
    if "translation" not in ego:
        xyz = ego.get("xyz") or [0.0, 0.0, 0.0]
        ego["translation"] = [float(xyz[0]), float(xyz[1]), 0.0]
    for obj in _objects(row.get(COLS.OBJECTS_JSON.value)):
        if not _is_ped(str(obj.get("class") or "")):
            continue
        bbox = obj.get("bbox_2d") or []
        if len(bbox) < 4:
            continue
        u, v = qwen_bottom_center_px(bbox, IMAGE_W, IMAGE_H)
        xy = pixel_to_ground_xy(u, v, ego, cs)
        if xy is None:
            continue
        if any(poly.contains(Point(xy[0], xy[1])) for poly in crossings):
            return True
    return False


def pseudo_gt_pedestrian_cross(
    windows: pl.DataFrame, frames: pl.DataFrame, map_root: Path | None = None
) -> pl.DataFrame:
    """Join rule v2: ped bbox bottom-center unprojected into a ped_crossing polygon.

    Hand-checks of this join must be disjoint from the eval-60 list.
    """

    from vsm.geometry.map_layers import load_map_layers

    cache: dict[str, Any] = {}
    cs_ix = _cs_index()
    rows: list[dict] = []
    for w in windows.iter_rows(named=True):
        subset = _window_frames(frames, w)
        if subset.height == 0:
            continue
        map_name = str(subset[COLS.MAP_NAME.value][0])
        if map_name not in cache:
            try:
                cache[map_name] = load_map_layers(map_name, map_root)
            except (FileNotFoundError, RuntimeError, ValueError, KeyError):
                cache[map_name] = None
        layers = cache[map_name]
        crossings = (layers or {}).get("ped_crossing") or []
        pts = []
        for ego in subset[COLS.EGO_POSE_JSON.value].to_list():
            xy = _ego_xy(str(ego))
            if xy is not None:
                pts.append(xy)
        hit_map = False
        if len(pts) >= 2 and crossings:
            from shapely.geometry import LineString

            path = LineString(pts)
            if path.length > 0:
                buf = path.buffer(float(cfg_dict("rules")["join"]["ego_path_buffer_m"]))
                hit_map = any(buf.intersects(poly) for poly in crossings)
        hit_ped = False
        for row in subset.iter_rows(named=True):
            cs = cs_ix.get(str(row.get("calibrated_sensor_token") or ""))
            if _ped_ground_in_crossing(row, crossings, cs):
                hit_ped = True
                break
        if hit_map and hit_ped:
            rows.append(
                {
                    WIN_COLS.WINDOW_ID.value: w[WIN_COLS.WINDOW_ID.value],
                    "class": "pedestrian_cross",
                    WIN_COLS.TS_START_US.value: w[WIN_COLS.TS_START_US.value],
                    WIN_COLS.TS_END_US.value: w[WIN_COLS.TS_END_US.value],
                    "join_rule_version": JOIN_RULE_PED,
                    "se_rule_checked": False,
                }
            )
    return (
        pl.DataFrame(rows)
        if rows
        else pl.DataFrame(
            schema={
                WIN_COLS.WINDOW_ID.value: pl.Utf8,
                "class": pl.Utf8,
                WIN_COLS.TS_START_US.value: pl.Int64,
                WIN_COLS.TS_END_US.value: pl.Int64,
                "join_rule_version": pl.Utf8,
                "se_rule_checked": pl.Boolean,
            }
        )
    )


def heuristic_cut_in(subset: pl.DataFrame) -> bool:
    """Lateral-enter then center-band. Conservative; not T0."""

    tracks: list[list[tuple[float, float]]] = []
    for raw in subset[COLS.OBJECTS_JSON.value].to_list():
        frame_pts = []
        for obj in _objects(raw):
            if not _is_vehicle(str(obj.get("class") or "")):
                continue
            bbox = obj.get("bbox_2d") or [0, 0, 1, 1]
            frame_pts.append(_bbox_cx_h(bbox))
        tracks.append(frame_pts)
    if len(tracks) < 4:
        return False
    half = len(tracks) // 2
    early = [p for fr in tracks[:half] for p in fr]
    late = [p for fr in tracks[half:] for p in fr]
    if not early or not late:
        return False
    early_side = any(cx < 280 or cx > 720 for cx, _h in early)
    late_center = any(380 <= cx <= 620 for cx, _h in late)
    grew = max(h for _cx, h in late) > max(h for _cx, h in early) * 1.1
    return early_side and late_center and grew


def heuristic_lead_braking(subset: pl.DataFrame) -> bool:
    """1/h slope of the largest center-band vehicle. Conservative; not T0."""

    xs: list[float] = []
    inv: list[float] = []
    hs: list[float] = []
    for i, raw in enumerate(subset[COLS.OBJECTS_JSON.value].to_list()):
        cands = []
        for obj in _objects(raw):
            if not _is_vehicle(str(obj.get("class") or "")):
                continue
            cx, h = _bbox_cx_h(obj.get("bbox_2d") or [0, 0, 1, 1])
            if 380 <= cx <= 620:
                cands.append((h, 1.0 / h))
        if not cands:
            continue
        h, inv_h = max(cands, key=lambda t: t[0])
        xs.append(float(i))
        inv.append(inv_h)
        hs.append(h)
    slope = _lin_slope(xs, inv)
    if slope is None:
        return False
    # 1/h falling = lead growing. Require growth and a steep-enough slope.
    return slope < -0.012 and hs[-1] > hs[0] * 1.12 and hs[0] < 450


def build_pseudo_gt(windows: pl.DataFrame, frames: pl.DataFrame, map_root: Path | None = None) -> pl.DataFrame:
    """Corpus-wide pseudo-GT used only for prevalence.

    pedestrian_cross = map join (versioned). cut_in / lead_braking 2D heuristics are
    candidates only — they are NOT counted as n_pos (too loose / T0 is Day 3).
    """

    return pseudo_gt_pedestrian_cross(windows, frames, map_root)


def _attach_meta(windows: pl.DataFrame, frames: pl.DataFrame) -> pl.DataFrame:
    if COLS.MAP_NAME.value in windows.columns and COLS.CONDITION_LIGHT.value in windows.columns:
        return windows
    lookup = frames.select(
        [COLS.FRAME_ID.value, COLS.MAP_NAME.value, COLS.CONDITION_LIGHT.value, COLS.SCENE_NAME.value]
    )
    maps: list[str] = []
    lights: list[str] = []
    scenes: list[str] = []
    for w in windows.iter_rows(named=True):
        ids = list(w.get(WIN_COLS.FRAME_IDS.value) or [])
        sub = lookup.filter(pl.col(COLS.FRAME_ID.value).is_in(ids))
        maps.append(str(sub[COLS.MAP_NAME.value][0]) if sub.height else "unknown")
        lights.append(str(sub[COLS.CONDITION_LIGHT.value][0]) if sub.height else "day")
        scenes.append(str(sub[COLS.SCENE_NAME.value][0]) if sub.height and COLS.SCENE_NAME.value in sub.columns else "")
    return windows.with_columns(
        pl.Series(COLS.MAP_NAME.value, maps),
        pl.Series(COLS.CONDITION_LIGHT.value, lights),
        pl.Series("scene_name", scenes),
    )


def prevalence_audit(windows: pl.DataFrame, pseudo: pl.DataFrame, maps: pl.DataFrame | None = None) -> PrevalenceTable:
    """Stratify by map (≥5 windows/map) and light. Classes with <3 corpus positives leave the headline."""

    joined = windows
    map_col = COLS.MAP_NAME.value
    light_col = COLS.CONDITION_LIGHT.value
    if maps is not None and map_col in maps.columns:
        key = WIN_COLS.SCENE_TOKEN.value
        if key in windows.columns and key in maps.columns:
            extra_cols = [key, map_col] + ([light_col] if light_col in maps.columns else [])
            joined = windows.join(maps.select(extra_cols).unique(), on=key, how="left")
    if map_col not in joined.columns:
        joined = joined.with_columns(pl.lit("unknown").alias(map_col))
    if light_col not in joined.columns:
        joined = joined.with_columns(pl.lit("unknown").alias(light_col))
    if "class" not in pseudo.columns:
        raise ValueError("pseudo-GT needs a class column")
    table = PrevalenceTable()
    spec_classes = [c for c in CLASS_IDS if c != "none"]
    present = sorted(set(pseudo["class"].to_list()) | set(spec_classes))
    maps_present = sorted({str(x) for x in joined[map_col].to_list()})
    lights_present = sorted({str(x) for x in joined[light_col].to_list()})
    totals: dict[str, int] = {}
    for cls in present:
        pos = pseudo.filter(pl.col("class") == cls)
        n_pos_total = (
            pos.select(WIN_COLS.WINDOW_ID.value).unique().height
            if WIN_COLS.WINDOW_ID.value in pos.columns
            else pos.height
        )
        totals[cls] = int(n_pos_total)
        headline = n_pos_total >= _OUT
        if not headline:
            table.out_of_headline.append(str(cls))
            if n_pos_total == 0:
                table.notes[cls] = "support not present in corpus — VN-deployment taxonomy"
        for m in maps_present:
            for light in lights_present:
                slice_w = joined.filter((pl.col(map_col) == m) & (pl.col(light_col) == light))
                n_win = slice_w.height
                n_pos = 0
                if WIN_COLS.WINDOW_ID.value in pos.columns and n_win:
                    wids = set(slice_w[WIN_COLS.WINDOW_ID.value].to_list())
                    n_pos = (
                        pos.filter(pl.col(WIN_COLS.WINDOW_ID.value).is_in(list(wids)))
                        .select(WIN_COLS.WINDOW_ID.value)
                        .unique()
                        .height
                    )
                table.rows.append(
                    {
                        "class": cls,
                        "map": m,
                        "light": light,
                        "n_windows": int(n_win),
                        "n_pos": int(n_pos),
                        "n_pos_corpus": int(n_pos_total),
                        "headline": bool(headline),
                        "map_support_ok": joined.filter(pl.col(map_col) == m).height >= _MIN_PER_MAP,
                    }
                )
    # Headline-3: intersection of pre-registered candidates that survive the <3 rule, ranked by n_pos.
    survivors = [c for c in HEADLINE_CANDIDATES if totals.get(c, 0) >= _OUT]
    if len(survivors) < 3:
        ranked = sorted((c for c in present if totals.get(c, 0) >= _OUT), key=lambda c: totals[c], reverse=True)
        for c in ranked:
            if c not in survivors:
                survivors.append(c)
            if len(survivors) >= 3:
                break
    table.headline = survivors[:3]
    return table


def decide_headline(table: PrevalenceTable) -> list[str]:
    """Return the frozen headline-3 class ids (may be shorter if support is missing)."""

    return list(table.headline)


def gt_list_sha(window_ids: Iterable[str]) -> str:
    """SHA-256 of sorted window_ids joined by newline."""

    blob = "\n".join(sorted(str(x) for x in window_ids)).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def build_gt_list(
    frames: pl.DataFrame,
    cfg: Any,
    windows: pl.DataFrame | None = None,
    pseudo: pl.DataFrame | None = None,
    exclude: Iterable[str] = (),
) -> list[str]:
    """Stratified draw: proportional by map, oversample presence-strata, design weights. seed=7."""

    del frames
    if windows is None:
        raise ValueError("build_gt_list needs windows")
    ev = cfg.model_dump() if hasattr(cfg, "model_dump") else dict(cfg)
    n_gt = int(ev.get("n_gt") or _EVAL["n_gt"])
    seed = int((ev.get("seeds") or _EVAL["seeds"])["gt_list"])
    maps_expected = list((ev.get("allocation") or _EVAL["allocation"])["maps_expected"])
    w_presence = float((ev.get("allocation") or {}).get("weight_presence") or 2.0)
    w_night = float((ev.get("allocation") or {}).get("weight_night") or 1.2)
    banned = set(exclude)
    presence: set[str] = set()
    if pseudo is not None and WIN_COLS.WINDOW_ID.value in pseudo.columns:
        presence = set(pseudo[WIN_COLS.WINDOW_ID.value].to_list())
    rng = random.Random(seed)
    by_map: dict[str, list[tuple[float, str]]] = {m: [] for m in maps_expected}
    for w in windows.iter_rows(named=True):
        wid = str(w[WIN_COLS.WINDOW_ID.value])
        if wid in banned:
            continue
        m = str(w.get(COLS.MAP_NAME.value) or "unknown")
        if m not in by_map:
            by_map.setdefault(m, [])
        weight = 1.0
        if wid in presence:
            weight *= w_presence
        if str(w.get(COLS.CONDITION_LIGHT.value) or "") == "night":
            weight *= w_night
        by_map[m].append((weight, wid))
    # Proportional quota with floor min_windows_per_map, then fill remainder by weight.
    available = {m: v for m, v in by_map.items() if v}
    n_maps = max(len(available), 1)
    floor = min(_MIN_PER_MAP, min((len(v) for v in available.values()), default=_MIN_PER_MAP))
    quota = {m: min(floor, len(available[m])) for m in available}
    leftover = n_gt - sum(quota.values())
    total_w = sum(sum(w for w, _ in available[m]) for m in available)
    extras = []
    for m in available:
        share = leftover * (sum(w for w, _ in available[m]) / total_w) if total_w else leftover / n_maps
        extras.append((share, m))
    extras.sort(reverse=True)
    for i, m in enumerate([m for _s, m in extras]):
        if leftover <= 0:
            break
        room = len(available[m]) - quota[m]
        take = min(room, 1 + leftover // max(len(available) - i, 1))
        quota[m] += take
        leftover -= take
    chosen: list[str] = []
    for m, items in available.items():
        k = min(quota.get(m, 0), len(items))
        # weighted sample without replacement
        pool = list(items)
        for _ in range(k):
            tw = sum(w for w, _ in pool)
            r = rng.random() * tw
            acc = 0.0
            idx = 0
            for i, (w, _wid) in enumerate(pool):
                acc += w
                if acc >= r:
                    idx = i
                    break
            chosen.append(pool.pop(idx)[1])
    if len(chosen) < n_gt:
        rest = [wid for items in available.values() for _w, wid in items if wid not in chosen]
        rng.shuffle(rest)
        chosen.extend(rest[: n_gt - len(chosen)])
    return sorted(chosen[:n_gt])


def pick_handcheck_windows(windows: pl.DataFrame, n: int = 10, seed: int = 13) -> list[str]:
    """Stray windows for ped_cross join hand-checks. Must stay out of eval-60."""

    rng = random.Random(seed)
    ids = list(windows[WIN_COLS.WINDOW_ID.value].to_list())
    rng.shuffle(ids)
    # stride through the shuffled list so they are not clustered
    step = max(len(ids) // n, 1)
    return [ids[i] for i in range(0, min(len(ids), step * n), step)][:n]


def pick_quick_scan_windows(windows: pl.DataFrame, n: int = 20, seed: int = 13) -> list[str]:
    """20 stray windows for quick-mode presence flags."""

    return pick_handcheck_windows(windows, n=n, seed=seed)


def load_gt_labels(path: Path | None = None) -> pl.DataFrame:
    """Last-write-wins per window_id so session-2 anchor flips replace session-1 rows."""

    src = Path(path or REPO_ROOT / "results" / "gt_labels.jsonl")
    if not src.exists():
        return pl.DataFrame(schema={"window_id": pl.Utf8, "class_set": pl.List(pl.Utf8)})
    rows = [json.loads(line) for line in src.read_text(encoding="utf-8").splitlines() if line.strip()]
    last: dict[str, dict] = {}
    for rec in rows:
        last[str(rec["window_id"])] = rec
    return pl.DataFrame(list(last.values())) if last else pl.DataFrame()


def labeling_session(windows: pl.DataFrame, frames: pl.DataFrame, labeler: str, blinded: bool = False) -> Path:
    """Launch the 6-frame tool. Session cap lives in eval.yaml."""

    from vsm.labeling.tool import launch

    return launch(windows, frames, labeler=labeler, blinded=blinded)
