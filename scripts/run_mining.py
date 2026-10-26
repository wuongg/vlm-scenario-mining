"""Day 3+ cascade runner. Day 1: smoke fallback on synth windows."""

from __future__ import annotations

from vsm.config import load_named
from vsm.geometry.synth_fixtures import three_synth_windows
from vsm.mining.cascade import run_cascade
from vsm.vlm.backends.fallback import FallbackBackend


def main() -> int:
    frames, windows = three_synth_windows()
    findings = run_cascade(windows, [FallbackBackend()], load_named("ingest"), frames=frames)
    print(findings)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
