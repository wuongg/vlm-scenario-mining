"""Light / blur / density / visibility columns. Night-in-hollandv is a recorded confound."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import polars as pl
from PIL import Image

from vsm.config import cfg_dict
from vsm.constants import COLS

_INGEST = cfg_dict("ingest")
_NIGHT_TOKENS = tuple(str(t).lower() for t in _INGEST["conditions"]["night_tokens"])
_NIGHT_MAP = str(_INGEST["conditions"]["night_map_confound"])
_DENSITY_EDGES = [int(x) for x in _INGEST["conditions"]["density_edges"]]
_DENSITY_LABELS = [str(x) for x in _INGEST["conditions"]["density_labels"]]


def _laplacian_variance(path: str) -> float:
    p = Path(path)
    if not p.exists():
        return float("nan")
    with Image.open(p) as im:
        gray = np.asarray(im.convert("L"), dtype=np.float64)
    kernel = np.array([[0.0, 1.0, 0.0], [1.0, -4.0, 1.0], [0.0, 1.0, 0.0]])
    # pad then correlate
    padded = np.pad(gray, 1, mode="edge")
    acc = np.zeros_like(gray)
    for i in range(3):
        for j in range(3):
            acc += kernel[i, j] * padded[i : i + gray.shape[0], j : j + gray.shape[1]]
    return float(acc.var())


def _density_bin(n_obj: int) -> str:
    for i, edge in enumerate(_DENSITY_EDGES[1:]):
        if n_obj < edge:
            return _DENSITY_LABELS[min(i, len(_DENSITY_LABELS) - 1)]
    return _DENSITY_LABELS[-1]


def _n_objects(objects_json: str) -> int:
    try:
        payload = json.loads(objects_json) if objects_json else []
    except json.JSONDecodeError:
        return 0
    return len(payload) if isinstance(payload, list) else 0


def add_condition_columns(frames: pl.DataFrame) -> pl.DataFrame:
    """Return frames with light (description+map), blur quartile, density bin, visibility bin."""

    desc_col = "scene_description" if "scene_description" in frames.columns else None
    lights: list[str] = []
    blurs: list[float] = []
    densities: list[str] = []
    vis: list[int] = []
    for row in frames.iter_rows(named=True):
        desc = str(row.get(desc_col) or "").lower() if desc_col else ""
        map_name = str(row.get(COLS.MAP_NAME.value) or "")
        is_night = any(tok in desc for tok in _NIGHT_TOKENS)
        light = "night" if is_night else "day"
        if "dusk" in desc:
            light = "dusk"
        if is_night and map_name == _NIGHT_MAP:
            # confound recorded, not "fixed"
            light = "night"
        lights.append(light)
        blurs.append(_laplacian_variance(str(row.get(COLS.CLIP_REF.value) or "")))
        densities.append(_density_bin(_n_objects(str(row.get(COLS.OBJECTS_JSON.value) or "[]"))))
        vis.append(int(row.get(COLS.VISIBILITY_BIN.value) or 4))

    out = frames.with_columns(
        pl.Series(COLS.CONDITION_LIGHT.value, lights),
        pl.Series("_blur_var", blurs),
        pl.Series(COLS.DENSITY_BIN.value, densities),
        pl.Series(COLS.VISIBILITY_BIN.value, vis),
    )
    finite = [b for b in blurs if np.isfinite(b)]
    if finite:
        qs = np.quantile(finite, [0.25, 0.5, 0.75])

        def _q(v: float) -> int:
            if not np.isfinite(v):
                return 0
            if v <= qs[0]:
                return 1
            if v <= qs[1]:
                return 2
            if v <= qs[2]:
                return 3
            return 4

        out = out.with_columns(pl.Series(COLS.BLUR_QUARTILE.value, [_q(v) for v in blurs]))
    else:
        out = out.with_columns(pl.lit(0).alias(COLS.BLUR_QUARTILE.value))
    return out.drop("_blur_var")
