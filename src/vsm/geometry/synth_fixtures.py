"""Synthetic 3-window corpus for CI. Never reads data/ or nuScenes."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import polars as pl
from PIL import Image

from vsm.config import REPO_ROOT
from vsm.constants import COLS, IMAGE_H, IMAGE_W, PROVENANCE_GRID
from vsm.geometry.projection import (
    quaternion_rotation_matrix,
    view_points,
)

SYNTH_SCENE = "scene-synth"
SYNTH_CAM = "CAM_FRONT"
TS0_US = 1_600_000_000_000_000
GAP_US = 500_000  # 0.5 s keyframe gap
N_FRAMES = 18  # w=6, stride=3 → 5 windows (CI still slices to 3)


def identity_wxyz() -> list[float]:
    return [1.0, 0.0, 0.0, 0.0]


def pinhole_k(w: int = IMAGE_W, h: int = IMAGE_H, fx: float = 800.0, fy: float = 800.0) -> list[list[float]]:
    return [[fx, 0.0, w / 2.0], [0.0, fy, h / 2.0], [0.0, 0.0, 1.0]]


def make_synth_nusc() -> dict:
    """Return a mini nusc-like dict: camera at origin looking +x after ego/cs transforms.

    Ego at (0,0,0) identity. Calibrated sensor: camera looks along +x of the
    vehicle with z_cam forward, so cs rotation maps vehicle x → camera z.
    """

    # Vehicle: x forward, y left, z up.
    # Camera: z forward, x right, y down. R_cs such that p_cam = R_cs.T @ (p_ego - t_cs)
    # We store R_cs whose columns are camera axes in vehicle frame:
    # cam_x = -y_veh, cam_y = -z_veh, cam_z = +x_veh
    r_cs = np.array(
        [
            [0.0, 0.0, 1.0],
            [-1.0, 0.0, 0.0],
            [0.0, -1.0, 0.0],
        ],
        dtype=np.float64,
    )
    # quaternion from rotation matrix (wxyz)
    q_cs = _rotmat_to_wxyz(r_cs)
    t_cs = [0.0, 0.0, 1.5]
    ego_token = "ego0"
    cs_token = "cs0"
    sd_token = "sd0"
    sample_token = "sample0"
    return {
        "ego_pose": {
            ego_token: {"token": ego_token, "translation": [0.0, 0.0, 0.0], "rotation": identity_wxyz()},
        },
        "calibrated_sensor": {
            cs_token: {
                "token": cs_token,
                "translation": t_cs,
                "rotation": q_cs,
                "camera_intrinsic": pinhole_k(),
            }
        },
        "sample_data": {
            sd_token: {
                "token": sd_token,
                "sample_token": sample_token,
                "ego_pose_token": ego_token,
                "calibrated_sensor_token": cs_token,
                "width": IMAGE_W,
                "height": IMAGE_H,
                "channel": SYNTH_CAM,
            }
        },
        "sd_token": sd_token,
        "r_cs": r_cs,
        "t_cs": t_cs,
    }


def _rotmat_to_wxyz(r: np.ndarray) -> list[float]:
    m = np.asarray(r, dtype=np.float64)
    t = float(np.trace(m))
    if t > 0:
        s = 0.5 / np.sqrt(t + 1.0)
        w = 0.25 / s
        x = (m[2, 1] - m[1, 2]) * s
        y = (m[0, 2] - m[2, 0]) * s
        z = (m[1, 0] - m[0, 1]) * s
    else:
        i = int(np.argmax([m[0, 0], m[1, 1], m[2, 2]]))
        if i == 0:
            s = 2.0 * np.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2])
            w = (m[2, 1] - m[1, 2]) / s
            x = 0.25 * s
            y = (m[0, 1] + m[1, 0]) / s
            z = (m[0, 2] + m[2, 0]) / s
        elif i == 1:
            s = 2.0 * np.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2])
            w = (m[0, 2] - m[2, 0]) / s
            x = (m[0, 1] + m[1, 0]) / s
            y = 0.25 * s
            z = (m[1, 2] + m[2, 1]) / s
        else:
            s = 2.0 * np.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1])
            w = (m[1, 0] - m[0, 1]) / s
            x = (m[0, 2] + m[2, 0]) / s
            y = (m[1, 2] + m[2, 1]) / s
            z = 0.25 * s
    q = np.array([w, x, y, z], dtype=np.float64)
    q = q / np.linalg.norm(q)
    return [float(v) for v in q]


def expected_pixel_of_point(nusc: dict, xyz_global: np.ndarray) -> np.ndarray:
    """Independent pinhole oracle used by the geometry fixture (not the code under test)."""

    ego = next(iter(nusc["ego_pose"].values()))
    cs = next(iter(nusc["calibrated_sensor"].values()))
    p = np.asarray(xyz_global, dtype=np.float64).reshape(3, 1)
    p = p - np.asarray(ego["translation"]).reshape(3, 1)
    p = quaternion_rotation_matrix(np.asarray(ego["rotation"])).T @ p
    p = p - np.asarray(cs["translation"]).reshape(3, 1)
    p = quaternion_rotation_matrix(np.asarray(cs["rotation"])).T @ p
    img = view_points(p, np.asarray(cs["camera_intrinsic"]), normalize=True)
    return np.array([img[0, 0], img[1, 0]], dtype=np.float64)


def _write_gray_png(path: Path, seed: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    arr = rng.integers(40, 220, size=(IMAGE_H // 4, IMAGE_W // 4), dtype=np.uint8)
    Image.fromarray(arr, mode="L").resize((IMAGE_W, IMAGE_H)).save(path)


def build_synth_frames(n_frames: int = N_FRAMES, out_dir: Path | None = None) -> pl.DataFrame:
    """Return a 12-keyframe CAM_FRONT table with motorcycle / pedestrian / empty thirds."""

    out_dir = out_dir or (REPO_ROOT / "data" / "cache" / "synth")
    rows = []
    for i in range(n_frames):
        ts = TS0_US + i * GAP_US
        if i < 4:
            objects = [{"class": "motorcycle", "bbox_2d": [420, 400, 580, 720], "conf": 0.9, "source": "synth"}]
            light = "day"
        else:
            # Last 8 frames empty so window 2 (stride-3) is true negative space.
            objects = []
            light = "night" if i < 8 else "day"
        png = out_dir / f"f{i:02d}.png"
        _write_gray_png(png, seed=i)
        pose = {
            "translation": [float(i * 2.0), 0.0, 0.0],
            "rotation": identity_wxyz(),
            "heading": 0.0,
            "xyz": [float(i * 2.0), 0.0, 0.0],
        }
        rows.append(
            {
                COLS.FRAME_ID.value: i,
                COLS.SAMPLE_TOKEN.value: f"s{i:02d}",
                COLS.SCENE_TOKEN.value: SYNTH_SCENE,
                COLS.SCENE_NAME.value: SYNTH_SCENE,
                COLS.CAMERA.value: SYNTH_CAM,
                COLS.TS_US.value: ts,
                COLS.IS_KEYFRAME.value: True,
                COLS.OBJECTS_JSON.value: json.dumps(objects),
                COLS.EGO_POSE_JSON.value: json.dumps(pose),
                COLS.CLIP_REF.value: str(png),
                COLS.PROVENANCE.value: PROVENANCE_GRID,
                COLS.CONDITION_LIGHT.value: light,
                COLS.BLUR_QUARTILE.value: 2,
                COLS.DENSITY_BIN.value: "sparse" if objects else "empty",
                COLS.VISIBILITY_BIN.value: 4,
                COLS.GT_DROP_RATE.value: 0.0,
                COLS.MAP_NAME.value: "boston-seaport" if light == "day" else "singapore-hollandvillage",
                "scene_description": "Night rain" if light == "night" else "Daytime traffic",
            }
        )
    return pl.DataFrame(rows)


def three_synth_windows(frames: pl.DataFrame | None = None) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Return (frames, windows) with exactly 3 grid windows for the CI mini-pipeline."""

    from vsm.config import load_named
    from vsm.ingest.windows import build_grid_windows

    frames = frames if frames is not None else build_synth_frames()
    windows = build_grid_windows(frames, load_named("ingest"))
    if windows.height < 3:
        raise RuntimeError(f"expected ≥3 synth windows, got {windows.height}")
    return frames, windows.head(3)
