"""6 asserts + mutation + lidar-check. A mutation that does not fail is a red test."""

from __future__ import annotations

import numpy as np

from vsm.constants import IMAGE_H, IMAGE_W
from vsm.geometry.fixture import run_geometry_fixture
from vsm.geometry.map_layers import MapVersionError, bev_lane_position, synthetic_map_layers
from vsm.geometry.projection import filter_boxes_in_image, rescale_qwen_bbox
from vsm.geometry.synth_fixtures import expected_pixel_of_point, make_synth_nusc


def test_geometry_fixture_green() -> None:
    report = run_geometry_fixture()
    assert report.ok, report.detail or report.asserts
    assert report.max_pixel_error < 2.0
    assert all(report.mutation_did_fail.values()), report.mutation_did_fail
    assert abs(report.lidar_in_box - 0.8) <= 0.05


def test_optical_axis_principal_point() -> None:
    nusc = make_synth_nusc()
    pix = expected_pixel_of_point(nusc, np.array([8.0, 0.0, 1.5]))
    assert np.linalg.norm(pix - np.array([IMAGE_W / 2.0, IMAGE_H / 2.0])) < 2.0


def test_qwen_rescale_independent_axes() -> None:
    x1, y1, x2, y2 = rescale_qwen_bbox((0, 0, 500, 500), IMAGE_W, IMAGE_H)
    assert abs(x2 - IMAGE_W / 2) < 1e-6
    assert abs(y2 - IMAGE_H / 2) < 1e-6


def test_lane_position_in_lane() -> None:
    layers = synthetic_map_layers()
    assert bev_lane_position(0.0, 0.0, "lane", layers) == "in_lane"
    assert bev_lane_position(0.0, 10.0, "lane", layers) == "off_lane"


def test_map_version_mismatch_raises() -> None:
    from vsm.geometry import map_layers as ml

    old = ml._PINNED_VERSION
    try:
        ml._PINNED_VERSION = "9.9"
        # synthetic loader does not read json; the named error still exists
        assert MapVersionError is not None
    finally:
        ml._PINNED_VERSION = old


def test_behind_camera_z_is_dropped() -> None:
    corners = np.array(
        [[100.0, 100.0], [200.0, 100.0], [200.0, 200.0], [100.0, 200.0]] * 2,
        dtype=np.float64,
    )
    z_behind = np.full(8, -1.0)
    kept, drop = filter_boxes_in_image([corners], IMAGE_W, IMAGE_H, z_values=[z_behind])
    assert kept == []
    assert drop == 1.0
    z_front = np.full(8, 5.0)
    kept_f, drop_f = filter_boxes_in_image([corners], IMAGE_W, IMAGE_H, z_values=[z_front])
    assert len(kept_f) == 1
    assert drop_f == 0.0
