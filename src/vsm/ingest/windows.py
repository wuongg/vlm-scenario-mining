"""Grid windows: N = floor((K − w) / s) + 1 per (scene, camera)."""

from __future__ import annotations

import polars as pl

from vsm.config import Settings
from vsm.constants import CAMERAS, COLS, GEOMETRY_A, PROVENANCE_GRID, WIN_COLS, WINDOW_STRIDE, WINDOW_W


def build_grid_windows(frames: pl.DataFrame, cfg: Settings) -> pl.DataFrame:
    """Return one row per sliding window with provenance=grid and a stable window_id."""

    ingest = cfg.model_dump()
    w = int(ingest.get("window", {}).get("w", WINDOW_W))
    stride = int(ingest.get("window", {}).get("stride", WINDOW_STRIDE))
    cameras = tuple(ingest.get("cameras") or CAMERAS)
    geometry_mode = str(ingest.get("geometry_mode", {}).get("mapful", GEOMETRY_A))
    rows: list[dict] = []
    scoped = frames.filter(pl.col(COLS.CAMERA.value).is_in(list(cameras)))
    scoped = scoped.sort([COLS.SCENE_TOKEN.value, COLS.CAMERA.value, COLS.TS_US.value])
    for (scene, cam, scene_name), grp in scoped.group_by(
        [COLS.SCENE_TOKEN.value, COLS.CAMERA.value, COLS.SCENE_NAME.value],
        maintain_order=True,
    ):
        k = grp.height
        if k < w:
            continue
        n = (k - w) // stride + 1
        frame_ids = grp[COLS.FRAME_ID.value].to_list()
        ts = grp[COLS.TS_US.value].to_list()
        for i in range(n):
            start = i * stride
            end = start + w
            ts0 = int(ts[start])
            window_id = f"{scene_name}_{cam}_{ts0}"
            rows.append(
                {
                    WIN_COLS.WINDOW_ID.value: window_id,
                    WIN_COLS.SCENE_TOKEN.value: str(scene),
                    WIN_COLS.CAMERA.value: str(cam),
                    WIN_COLS.TS_START_US.value: int(ts[start]),
                    WIN_COLS.TS_END_US.value: int(ts[end - 1]),
                    WIN_COLS.FRAME_IDS.value: [int(x) for x in frame_ids[start:end]],
                    WIN_COLS.PROVENANCE.value: PROVENANCE_GRID,
                    WIN_COLS.GEOMETRY_MODE.value: geometry_mode,
                    "scene_name": str(scene_name),
                }
            )
    return pl.DataFrame(rows)
