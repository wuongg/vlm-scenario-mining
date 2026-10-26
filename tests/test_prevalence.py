"""Prevalence audit + GT-list SHA on synthetic windows."""

from __future__ import annotations

from vsm.config import load_named
from vsm.eval.gt import build_gt_list, gt_list_sha, prevalence_audit
from vsm.geometry.synth_fixtures import three_synth_windows
from vsm.ingest.conditions import add_condition_columns


def test_prevalence_out_of_headline_rule() -> None:
    import polars as pl

    frames, windows = three_synth_windows()
    frames = add_condition_columns(frames)
    windows = windows.with_columns(
        pl.lit("boston-seaport").alias("map_name"),
        pl.lit("day").alias("condition_light"),
    )
    pseudo = pl.DataFrame(
        {
            "window_id": [windows["window_id"][0], windows["window_id"][0]],
            "class": ["pedestrian_cross", "cut_in"],
        }
    )
    table = prevalence_audit(windows, pseudo, frames)
    assert "lead_braking" in table.out_of_headline
    assert table.to_dataframe().height > 0


def test_gt_list_sha_stable() -> None:
    ids = ["b", "a", "c"]
    assert gt_list_sha(ids) == gt_list_sha(["a", "c", "b"])
    assert len(gt_list_sha(ids)) == 64


def test_build_gt_list_seed_deterministic() -> None:
    frames, _w = three_synth_windows()
    frames = add_condition_columns(frames)
    from vsm.config import load_named as ln
    from vsm.ingest.windows import build_grid_windows as bgw

    windows = bgw(frames, ln("ingest"))
    import polars as pl

    windows = windows.with_columns(
        pl.lit("boston-seaport").alias("map_name"),
        pl.lit("day").alias("condition_light"),
    )
    a = build_gt_list(frames, load_named("eval"), windows=windows)
    b = build_gt_list(frames, load_named("eval"), windows=windows)
    assert a == b
    assert a == sorted(a)
