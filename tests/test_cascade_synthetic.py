"""E2E mini-pipeline on 3 synthetic windows (CI main job)."""

from __future__ import annotations

from vsm.selftest import cascade_synthetic


def test_cascade_synthetic_selftest() -> None:
    assert cascade_synthetic() == 0
