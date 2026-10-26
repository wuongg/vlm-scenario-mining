"""Map expansion layers. Version mismatch and out-of-bounds ego raise LOUD."""

from __future__ import annotations

import json
from pathlib import Path

from shapely.geometry import Point, Polygon

from vsm.config import cfg_dict

_INGEST = cfg_dict("ingest")
_PINNED_VERSION = str(_INGEST["map"]["version"])
_LAYER_NAMES = tuple(_INGEST["map"]["layers"])


class MapVersionError(RuntimeError):
    """Map JSON version does not match the pinned expansion version."""


class MapBoundsError(RuntimeError):
    """Ego pose sits outside the map polygon — quarantine the scene."""


def load_map_layers(map_name: str, map_root: Path | None = None) -> dict[str, list[Polygon]]:
    """Return driveable/sidewalk/ped_crossing/lane polygons. Assert version pin."""

    if map_root is None:
        from vsm.config import REPO_ROOT

        map_root = REPO_ROOT / str(_INGEST["paths"]["map_root"])
    map_root = Path(map_root)
    json_path = None
    for cand in (
        map_root / "expansion" / f"{map_name}.json",
        map_root / f"{map_name}.json",
        map_root / "maps" / "expansion" / f"{map_name}.json",
    ):
        if cand.exists():
            json_path = cand
            break
    if json_path is None:
        raise FileNotFoundError(f"map json for {map_name} not under {map_root}")
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    version = str(payload.get("version") or payload.get("map_version") or "")
    if version and _PINNED_VERSION not in version and version not in (_PINNED_VERSION, f"expv{_PINNED_VERSION}"):
        # nuScenes expansion files often omit a top-level version; if present it must match.
        raise MapVersionError(f"map {map_name} version {version!r} != pinned {_PINNED_VERSION!r}")
    node_ix = {n["token"]: n for n in payload.get("node", [])}
    poly_ix = {p["token"]: p for p in payload.get("polygon", [])}

    def _poly_from_nodes(ext: list) -> Polygon | None:
        coords = []
        for tok in ext:
            node = node_ix.get(tok)
            if node is None:
                continue
            coords.append((float(node["x"]), float(node["y"])))
        if len(coords) < 3:
            return None
        if coords[0] != coords[-1]:
            coords.append(coords[0])
        try:
            return Polygon(coords)
        except ValueError:
            return None

    def _polys_from_record(rec: dict) -> list[Polygon]:
        ext = rec.get("exterior_node_tokens") or rec.get("exterior_nodes") or []
        if ext:
            p = _poly_from_nodes(ext)
            return [p] if p is not None and not p.is_empty else []
        tokens: list = []
        if rec.get("polygon_tokens"):
            tokens = list(rec["polygon_tokens"])
        elif rec.get("polygon_token"):
            tokens = [rec["polygon_token"]]
        out_p: list[Polygon] = []
        for tok in tokens:
            linked = poly_ix.get(str(tok)) or {}
            p = _poly_from_nodes(linked.get("exterior_node_tokens") or linked.get("exterior_nodes") or [])
            if p is not None and not p.is_empty:
                out_p.append(p)
        return out_p

    out: dict[str, list[Polygon]] = {name: [] for name in _LAYER_NAMES}
    alias = {"sidewalk": "walkway", "walkway": "walkway", "driveable_area": "drivable_area"}
    for layer in _LAYER_NAMES:
        key = layer
        records = payload.get(key) or payload.get(alias.get(key, key)) or []
        for rec in records:
            if not isinstance(rec, dict):
                continue
            out[layer].extend(_polys_from_record(rec))
    return out


def _union_bounds(polys: list[Polygon]) -> tuple[float, float, float, float]:
    xs: list[float] = []
    ys: list[float] = []
    for p in polys:
        minx, miny, maxx, maxy = p.bounds
        xs.extend([minx, maxx])
        ys.extend([miny, maxy])
    return min(xs), min(ys), max(xs), max(ys)


def assert_ego_in_map(x: float, y: float, layers: dict[str, list[Polygon]], scene: str) -> None:
    """Raise MapBoundsError if (x, y) is outside every loaded layer's bounds."""

    polys = [p for group in layers.values() for p in group]
    if not polys:
        raise MapBoundsError(f"scene {scene}: map layers empty")
    minx, miny, maxx, maxy = _union_bounds(polys)
    if not (minx <= x <= maxx and miny <= y <= maxy):
        raise MapBoundsError(f"scene {scene}: ego ({x:.2f},{y:.2f}) outside map bounds")


def bev_lane_position(x: float, y: float, lane_name: str, layers: dict[str, list[Polygon]] | None = None) -> str:
    """Project pose onto a named lane and return in_lane | adjacent | off_lane."""

    if layers is None:
        return "off_lane"
    lanes = layers.get(lane_name) or layers.get("lane") or []
    pt = Point(float(x), float(y))
    if not lanes:
        raise RuntimeError("arcline/lane layer missing — map version mismatch must raise LOUD")
    distances = []
    inside = False
    for poly in lanes:
        if poly.contains(pt) or poly.covers(pt):
            inside = True
            distances.append(0.0)
        else:
            distances.append(float(poly.exterior.distance(pt)))
    d = min(distances) if distances else float("inf")
    if inside or d <= 0.75:
        return "in_lane"
    if d <= 2.0:
        return "adjacent"
    return "off_lane"


def project_pose_to_lane(x: float, y: float, layers: dict[str, list[Polygon]]) -> str:
    """Smoke wrapper used by the 5-frames-per-scene check."""

    return bev_lane_position(x, y, "lane", layers)


def synthetic_map_layers() -> dict[str, list[Polygon]]:
    """Axis-aligned map used by CI. Origin (0,0), driveable x∈[-20,80], y∈[-6,6]."""

    drive = Polygon([(-20, -6), (80, -6), (80, 6), (-20, 6), (-20, -6)])
    lane = Polygon([(-20, -1.85), (80, -1.85), (80, 1.85), (-20, 1.85), (-20, -1.85)])
    walk = Polygon([(-20, 6), (80, 6), (80, 8), (-20, 8), (-20, 6)])
    cross = Polygon([(20, -6), (24, -6), (24, 6), (20, 6), (20, -6)])
    return {
        "driveable_area": [drive],
        "sidewalk": [walk],
        "ped_crossing": [cross],
        "lane": [lane],
    }
