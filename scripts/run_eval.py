"""Day 9–10. Day 1 runs the prevalence skeleton on synth pseudo-GT."""

from __future__ import annotations

from vsm.config import load_named
from vsm.eval.gt import prevalence_audit
from vsm.geometry.synth_fixtures import three_synth_windows
from vsm.ingest.conditions import add_condition_columns
from vsm.mining.cascade import run_cascade
from vsm.vlm.backends.fallback import FallbackBackend


def main() -> int:
    frames, windows = three_synth_windows()
    frames = add_condition_columns(frames)
    findings = run_cascade(windows, [FallbackBackend()], load_named("eval"), frames=frames)
    table = prevalence_audit(windows, findings.select(["window_id", "class"]), frames)
    print(table.to_dataframe())
    print("out_of_headline", table.out_of_headline)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
