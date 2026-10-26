"""Devkit-native projection: ego pose BEFORE calibrated_sensor; view_points axis-2."""

from __future__ import annotations

from typing import Any

import numpy as np

from vsm.constants import QWEN_BBOX_MAX


def quaternion_rotation_matrix(q: np.ndarray) -> np.ndarray:
    """Return 3x3 R such that p' = R @ p for a wxyz unit quaternion."""

    w, x, y, z = [float(v) for v in np.asarray(q, dtype=np.float64).reshape(-1)[:4]]
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def box_corners_global(center: np.ndarray, wlh: np.ndarray, rotation_q: np.ndarray) -> np.ndarray:
    """Return (3, 8) corners. x forward=length, y left=width, z up=height (nuScenes Box)."""

    width, length, height = [float(v) for v in np.asarray(wlh, dtype=np.float64).reshape(-1)[:3]]
    x_corners = length / 2 * np.array([1, 1, 1, 1, -1, -1, -1, -1], dtype=np.float64)
    y_corners = width / 2 * np.array([1, -1, -1, 1, 1, -1, -1, 1], dtype=np.float64)
    z_corners = height / 2 * np.array([1, 1, -1, -1, 1, 1, -1, -1], dtype=np.float64)
    corners = np.vstack((x_corners, y_corners, z_corners))
    rot = quaternion_rotation_matrix(rotation_q)
    corners = rot @ corners
    c = np.asarray(center, dtype=np.float64).reshape(3)
    corners = corners + c.reshape(3, 1)
    return corners


def view_points(points: np.ndarray, view: np.ndarray, normalize: bool) -> np.ndarray:
    """Project (3, N) points with a 3x3/3x4/4x4 view. Normalize divides by axis-2 (z)."""

    points = np.asarray(points, dtype=np.float64)
    view = np.asarray(view, dtype=np.float64)
    viewpad = np.eye(4, dtype=np.float64)
    viewpad[: view.shape[0], : view.shape[1]] = view
    nbr = points.shape[1]
    homo = np.concatenate((points, np.ones((1, nbr), dtype=np.float64)), axis=0)
    out = viewpad @ homo
    out = out[:3, :]
    if normalize:
        z = out[2:3, :]
        z = np.where(np.abs(z) < 1e-12, np.sign(z) * 1e-12 + (z == 0) * 1e-12, z)
        out = out / np.repeat(z, 3, axis=0)
    return out


def _transform_global_to_sensor(
    points: np.ndarray,
    ego_t: np.ndarray,
    ego_q: np.ndarray,
    cs_t: np.ndarray,
    cs_q: np.ndarray,
    *,
    ego_first: bool = True,
    use_r_transpose: bool = True,
) -> np.ndarray:
    """Transform (3, N) global points into the camera frame.

    Default matches the devkit: ego pose first, then calibrated_sensor,
    each as translate(-t) then rotate(R.T).
    """

    pts = np.asarray(points, dtype=np.float64).copy()
    ego_r = quaternion_rotation_matrix(ego_q)
    cs_r = quaternion_rotation_matrix(cs_q)
    ego_rot = ego_r.T if use_r_transpose else ego_r
    cs_rot = cs_r.T if use_r_transpose else cs_r

    def apply(t: np.ndarray, rot: np.ndarray) -> None:
        nonlocal pts
        pts = pts - np.asarray(t, dtype=np.float64).reshape(3, 1)
        pts = rot @ pts

    if ego_first:
        apply(ego_t, ego_rot)
        apply(cs_t, cs_rot)
    else:
        apply(cs_t, cs_rot)
        apply(ego_t, ego_rot)
    return pts


def box_corners_in_camera(box: dict, nusc: dict[str, Any], sd_token: str) -> tuple[np.ndarray, np.ndarray]:
    """Return ((8, 2) pixels, (8,) camera-frame z). Ego transform runs before calibrated_sensor."""

    sd = nusc["sample_data"][sd_token] if not isinstance(nusc["sample_data"], dict) else nusc["sample_data"][sd_token]
    if isinstance(nusc["sample_data"], list):
        sd = next(x for x in nusc["sample_data"] if x["token"] == sd_token)
    ego = nusc["ego_pose"][sd["ego_pose_token"]]
    cs = nusc["calibrated_sensor"][sd["calibrated_sensor_token"]]
    corners = box_corners_global(box["translation"], box["size"], box["rotation"])
    cam = _transform_global_to_sensor(
        corners,
        np.asarray(ego["translation"], dtype=np.float64),
        np.asarray(ego["rotation"], dtype=np.float64),
        np.asarray(cs["translation"], dtype=np.float64),
        np.asarray(cs["rotation"], dtype=np.float64),
        ego_first=True,
        use_r_transpose=True,
    )
    k = np.asarray(cs["camera_intrinsic"], dtype=np.float64)
    img = view_points(cam, k, normalize=True)
    # sign-flip trap: z<0 corners invert after normalize — caller must filter via z
    pixels = np.stack((img[0, :], img[1, :]), axis=1)
    return pixels, np.asarray(cam[2, :], dtype=np.float64).reshape(-1)


def project_box_global_to_pixel(box: dict, nusc: dict[str, Any], sd_token: str) -> np.ndarray:
    """Return (8, 2) pixel corners. Ego transform runs before calibrated_sensor."""

    pixels, _z = box_corners_in_camera(box, nusc, sd_token)
    return pixels


def filter_boxes_in_image(
    boxes: list[np.ndarray],
    W: int,
    H: int,
    near: float = 0.1,
    z_values: list[np.ndarray] | None = None,
) -> tuple[list[np.ndarray], float]:
    """Keep boxes with at least one in-front corner inside the image. Return (kept, drop_rate)."""

    if not boxes:
        return [], 0.0
    kept: list[np.ndarray] = []
    for i, corners in enumerate(boxes):
        pix = np.asarray(corners, dtype=np.float64)
        if pix.ndim != 2 or pix.shape[1] != 2:
            continue
        visible = (pix[:, 0] >= 0) & (pix[:, 0] < W) & (pix[:, 1] >= 0) & (pix[:, 1] < H)
        if z_values is not None and i < len(z_values):
            in_front = np.asarray(z_values[i], dtype=np.float64).reshape(-1) > near
            if in_front.size == visible.size:
                visible = visible & in_front
            elif not bool(np.any(in_front)):
                continue
        if bool(np.any(visible)):
            kept.append(pix)
    drop_rate = 1.0 - (len(kept) / len(boxes))
    return kept, float(drop_rate)


def rescale_qwen_bbox(bbox_1000: tuple | list, W: int, H: int) -> tuple[float, float, float, float]:
    """Map a Qwen [0, 1000) box onto pixel W×H using independent axes (not min(W,H))."""

    x1, y1, x2, y2 = [float(v) for v in bbox_1000]
    s = float(QWEN_BBOX_MAX)
    return x1 / s * W, y1 / s * H, x2 / s * W, y2 / s * H


def pixel_to_qwen_bbox(bbox_px: tuple | list, W: int, H: int) -> list[int]:
    """Map a pixel box to integer [0, 1000) with independent W/H axes."""

    x1, y1, x2, y2 = [float(v) for v in bbox_px]
    s = float(QWEN_BBOX_MAX)
    return [
        int(np.clip(x1 / W * s, 0, s - 1)),
        int(np.clip(y1 / H * s, 0, s - 1)),
        int(np.clip(x2 / W * s, 0, s - 1)),
        int(np.clip(y2 / H * s, 0, s - 1)),
    ]


def qwen_bottom_center_px(bbox_1000: list | tuple, W: int, H: int) -> tuple[float, float]:
    """Return pixel (u, v) of the bbox bottom-center (feet)."""

    x1, y1, x2, y2 = rescale_qwen_bbox(bbox_1000, W, H)
    return (x1 + x2) / 2.0, y2


def pixel_to_ground_xy(
    u: float,
    v: float,
    ego: dict,
    cs: dict,
    plane_z: float | None = None,
) -> tuple[float, float] | None:
    """Unproject a pixel through K / cs / ego onto the z=plane_z ground (global)."""

    k = np.asarray(cs["camera_intrinsic"], dtype=np.float64)
    try:
        kinv = np.linalg.inv(k)
    except np.linalg.LinAlgError:
        return None
    d_cam = kinv @ np.array([float(u), float(v), 1.0], dtype=np.float64)
    r_cs = quaternion_rotation_matrix(np.asarray(cs["rotation"], dtype=np.float64))
    r_ego = quaternion_rotation_matrix(np.asarray(ego["rotation"], dtype=np.float64))
    d_glob = r_ego @ (r_cs @ d_cam)
    origin = r_ego @ np.asarray(cs["translation"], dtype=np.float64) + np.asarray(ego["translation"], dtype=np.float64)
    z_plane = float(ego["translation"][2] if plane_z is None else plane_z)
    if abs(float(d_glob[2])) < 1e-8:
        return None
    t = (z_plane - float(origin[2])) / float(d_glob[2])
    if t <= 0:
        return None
    p = origin + t * d_glob
    return float(p[0]), float(p[1])
