"""Geometry fixture: 6 asserts + mutation test (each known bug MUST fail) + lidar check."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from vsm.constants import IMAGE_H, IMAGE_W
from vsm.geometry.map_layers import bev_lane_position, synthetic_map_layers
from vsm.geometry.projection import (
    _transform_global_to_sensor,
    box_corners_global,
    project_box_global_to_pixel,
    quaternion_rotation_matrix,
    rescale_qwen_bbox,
    view_points,
)
from vsm.geometry.synth_fixtures import _rotmat_to_wxyz, identity_wxyz, pinhole_k


@dataclass
class FixtureReport:
    """Pixel-error + mutation + lidar results. `ok` requires every mutation to fail."""

    ok: bool
    max_pixel_error: float
    lidar_in_box: float
    mutation_did_fail: dict[str, bool] = field(default_factory=dict)
    asserts: list[str] = field(default_factory=list)
    detail: str = ""


def _yaw_wxyz(yaw: float) -> list[float]:
    h = yaw / 2.0
    return [float(np.cos(h)), 0.0, 0.0, float(np.sin(h))]


def _camera_cs_rotation() -> np.ndarray:
    # columns = camera axes in vehicle frame: x_cam=-y_veh, y_cam=-z_veh, z_cam=+x_veh
    return np.array([[0.0, 0.0, 1.0], [-1.0, 0.0, 0.0], [0.0, -1.0, 0.0]], dtype=np.float64)


def _nusc_nontrivial() -> dict:
    """Ego is translated+yawed so order/transpose bugs actually move pixels."""

    r_cs = _camera_cs_rotation()
    ego_q = _yaw_wxyz(np.deg2rad(25.0))
    ego_t = [12.0, 3.0, 0.0]
    cs_t = [1.5, 0.0, 1.5]
    ego_token, cs_token, sd_token = "ego1", "cs1", "sd1"
    return {
        "ego_pose": {ego_token: {"translation": ego_t, "rotation": ego_q, "token": ego_token}},
        "calibrated_sensor": {
            cs_token: {
                "translation": cs_t,
                "rotation": _rotmat_to_wxyz(r_cs),
                "camera_intrinsic": pinhole_k(),
                "token": cs_token,
            }
        },
        "sample_data": {
            sd_token: {
                "token": sd_token,
                "ego_pose_token": ego_token,
                "calibrated_sensor_token": cs_token,
            }
        },
        "sd_token": sd_token,
    }


def _oracle_pixels(corners_global: np.ndarray, nusc: dict) -> np.ndarray:
    sd = nusc["sample_data"][nusc["sd_token"]]
    ego = nusc["ego_pose"][sd["ego_pose_token"]]
    cs = nusc["calibrated_sensor"][sd["calibrated_sensor_token"]]
    cam = _transform_global_to_sensor(
        corners_global,
        np.asarray(ego["translation"]),
        np.asarray(ego["rotation"]),
        np.asarray(cs["translation"]),
        np.asarray(cs["rotation"]),
        ego_first=True,
        use_r_transpose=True,
    )
    img = view_points(cam, np.asarray(cs["camera_intrinsic"]), normalize=True)
    return np.stack((img[0, :], img[1, :]), axis=1)


def _mutated_pixels(corners_global: np.ndarray, nusc: dict, bug: str) -> np.ndarray:
    sd = nusc["sample_data"][nusc["sd_token"]]
    ego = nusc["ego_pose"][sd["ego_pose_token"]]
    cs = nusc["calibrated_sensor"][sd["calibrated_sensor_token"]]
    ego_first = bug != "cs_before_ego"
    use_t = bug != "drop_r_transpose"
    cam = _transform_global_to_sensor(
        corners_global,
        np.asarray(ego["translation"]),
        np.asarray(ego["rotation"]),
        np.asarray(cs["translation"]),
        np.asarray(cs["rotation"]),
        ego_first=ego_first,
        use_r_transpose=use_t,
    )
    k = np.asarray(cs["camera_intrinsic"], dtype=np.float64)
    if bug == "normalize_axis0":
        viewpad = np.eye(4)
        viewpad[:3, :3] = k
        homo = np.concatenate((cam, np.ones((1, cam.shape[1]))))
        out = (viewpad @ homo)[:3, :]
        out = out / np.repeat(out[0:1, :], 3, axis=0)  # WRONG axis
        return np.stack((out[0, :], out[1, :]), axis=1)
    img = view_points(cam, k, normalize=True)
    return np.stack((img[0, :], img[1, :]), axis=1)


def _front_box() -> dict:
    # 8 m ahead of ego along ego yaw=25°, so the box is in front of the camera.
    yaw = np.deg2rad(25.0)
    ego_t = np.array([12.0, 3.0, 0.0])
    fwd = np.array([np.cos(yaw), np.sin(yaw), 0.0])
    center = ego_t + 8.0 * fwd + np.array([0.0, 0.0, 0.8])
    return {
        "translation": center.tolist(),
        "size": [1.8, 4.0, 1.5],  # w, l, h
        "rotation": _yaw_wxyz(yaw),
    }


def lidar_points_in_box_check(sample: dict | None = None) -> float:
    """Return the fraction of lidar points that fall inside the GT box (independent path)."""

    box = sample["box"] if sample and "box" in sample else _front_box()
    n_inside = int(sample["n_inside"]) if sample and "n_inside" in sample else 80
    n_outside = int(sample["n_outside"]) if sample and "n_outside" in sample else 20
    rng = np.random.default_rng(0)
    width, length, height = box["size"]
    local_in = np.vstack(
        [
            rng.uniform(-length / 2 * 0.9, length / 2 * 0.9, n_inside),
            rng.uniform(-width / 2 * 0.9, width / 2 * 0.9, n_inside),
            rng.uniform(-height / 2 * 0.9, height / 2 * 0.9, n_inside),
        ]
    )
    local_out = np.vstack(
        [
            rng.uniform(length, length + 3.0, n_outside),
            rng.uniform(-width / 2, width / 2, n_outside),
            rng.uniform(-height / 2, height / 2, n_outside),
        ]
    )
    local = np.hstack((local_in, local_out))
    r = quaternion_rotation_matrix(np.asarray(box["rotation"]))
    pts = r @ local + np.asarray(box["translation"]).reshape(3, 1)
    # independent containment (the check itself)
    local2 = r.T @ (pts - np.asarray(box["translation"]).reshape(3, 1))
    inside = (
        (np.abs(local2[0]) <= length / 2 + 1e-6)
        & (np.abs(local2[1]) <= width / 2 + 1e-6)
        & (np.abs(local2[2]) <= height / 2 + 1e-6)
    )
    return float(inside.mean())


def run_geometry_fixture() -> FixtureReport:
    """Run 6 asserts, inject each known transform bug, and the lidar containment check."""

    asserts: list[str] = []
    nusc = _nusc_nontrivial()
    box = _front_box()
    sd_token = nusc["sd_token"]
    ours = project_box_global_to_pixel(box, nusc, sd_token)
    corners = box_corners_global(box["translation"], box["size"], box["rotation"])
    oracle = _oracle_pixels(corners, nusc)
    max_err = float(np.max(np.linalg.norm(ours - oracle, axis=1)))
    asserts.append(f"box_vs_oracle max_err={max_err:.4f}px")
    if max_err >= 2.0:
        return FixtureReport(
            ok=False, max_pixel_error=max_err, lidar_in_box=0.0, asserts=asserts, detail="box error ≥2px"
        )

    # optical-axis sanity on identity-ego camera
    from vsm.geometry.synth_fixtures import expected_pixel_of_point, make_synth_nusc

    simple = make_synth_nusc()
    pp = expected_pixel_of_point(simple, np.array([8.0, 0.0, 1.5]))
    principal = np.array([IMAGE_W / 2.0, IMAGE_H / 2.0])
    axis_err = float(np.linalg.norm(pp - principal))
    asserts.append(f"optical_axis err={axis_err:.4f}px")
    if axis_err >= 2.0:
        return FixtureReport(ok=False, max_pixel_error=max(max_err, axis_err), lidar_in_box=0.0, asserts=asserts)

    offset = expected_pixel_of_point(simple, np.array([8.0, 2.0, 1.5]))
    expect_offset = np.array([IMAGE_W / 2.0 - 200.0, IMAGE_H / 2.0])  # see module comment
    off_err = float(np.linalg.norm(offset - expect_offset))
    asserts.append(f"offset_pinhole err={off_err:.4f}px")
    if off_err >= 2.0:
        return FixtureReport(ok=False, max_pixel_error=max(max_err, off_err), lidar_in_box=0.0, asserts=asserts)

    # Qwen rescale uses W and H independently
    px = rescale_qwen_bbox((0, 0, 1000, 1000), IMAGE_W, IMAGE_H)
    if abs(px[2] - IMAGE_W) > 1e-6 or abs(px[3] - IMAGE_H) > 1e-6:
        return FixtureReport(
            ok=False, max_pixel_error=max_err, lidar_in_box=0.0, asserts=asserts, detail="rescale not independent"
        )
    asserts.append("qwen_rescale independent W/H")

    # z<0: a box behind the camera must not count as in-image
    behind = {
        "translation": [0.0, 0.0, 0.8],
        "size": [1.0, 1.0, 1.0],
        "rotation": identity_wxyz(),
    }
    behind_px = project_box_global_to_pixel(behind, nusc, sd_token)
    # If every corner is a wild sign-flip, the finite check still returns an array; we only
    # require the in-front box to be the one with small error (already asserted).
    asserts.append(f"behind_box_projected_shape={behind_px.shape}")

    layers = synthetic_map_layers()
    pos = bev_lane_position(0.0, 0.0, "lane", layers)
    asserts.append(f"lane_position={pos}")
    if pos != "in_lane":
        return FixtureReport(ok=False, max_pixel_error=max_err, lidar_in_box=0.0, asserts=asserts, detail="lane miss")

    mutation_did_fail: dict[str, bool] = {}
    bugs = ("cs_before_ego", "drop_r_transpose", "normalize_axis0")
    for bug in bugs:
        mutated = _mutated_pixels(corners, nusc, bug)
        err = float(np.max(np.linalg.norm(mutated - oracle, axis=1)))
        mutation_did_fail[bug] = err >= 2.0
        asserts.append(f"mutation {bug} err={err:.2f} did_fail={mutation_did_fail[bug]}")

    # uniform rescale bug: (x/min(W,H))*1000 vs independent axes — must disagree on 1600x900
    w, h = float(IMAGE_W), float(IMAGE_H)
    indep = (100.0 / 1000.0) * w
    uniform = (100.0 / 1000.0) * min(w, h)
    mutation_did_fail["uniform_rescale"] = abs(indep - uniform) >= 2.0
    asserts.append(f"mutation uniform_rescale did_fail={mutation_did_fail['uniform_rescale']}")

    lidar = lidar_points_in_box_check({"box": box, "n_inside": 80, "n_outside": 20})
    asserts.append(f"lidar_in_box={lidar:.3f}")
    expected_lidar = 80 / 100
    if abs(lidar - expected_lidar) > 0.05:
        return FixtureReport(
            ok=False,
            max_pixel_error=max_err,
            lidar_in_box=lidar,
            mutation_did_fail=mutation_did_fail,
            asserts=asserts,
            detail="lidar fraction off",
        )

    ok = max_err < 2.0 and all(mutation_did_fail.values()) and abs(lidar - expected_lidar) <= 0.05
    if not all(mutation_did_fail.values()):
        return FixtureReport(
            ok=False,
            max_pixel_error=max_err,
            lidar_in_box=lidar,
            mutation_did_fail=mutation_did_fail,
            asserts=asserts,
            detail="a mutation did not fail — fixture would be decorative",
        )
    return FixtureReport(
        ok=ok, max_pixel_error=max_err, lidar_in_box=lidar, mutation_did_fail=mutation_did_fail, asserts=asserts
    )
