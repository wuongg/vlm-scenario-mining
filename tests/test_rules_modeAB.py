"""Mode A vs Mode B on synthetic windows — class-set can match when boxes allow."""

from __future__ import annotations

from vsm.geometry.synth_fixtures import three_synth_windows
from vsm.ingest.conditions import add_condition_columns
from vsm.mining.rules import RULES, choose_mode, run_t0_window


def test_choose_mode_degrades_without_map() -> None:
    rule = RULES["lane_split_moto"]
    assert choose_mode({"map", "calibrated_ego"}, rule) == "A"
    assert choose_mode(set(), rule) == "B"


def test_t0_synth_emits_known_classes() -> None:
    frames, windows = three_synth_windows()
    frames = add_condition_columns(frames)
    import polars as pl

    w = windows.head(1).to_dicts()[0]
    subset = frames.filter(pl.col("frame_id").is_in(list(w["frame_ids"]))).to_dicts()
    flags = run_t0_window(w, subset, layers={}, cs=None, available=set())
    classes = {f.cls for f in flags}
    assert classes.issubset(set(RULES))


def test_oncoming_never_auto_fires() -> None:
    from vsm.mining.rules import rule_oncoming

    assert rule_oncoming([], "A", window_id="w") == []
