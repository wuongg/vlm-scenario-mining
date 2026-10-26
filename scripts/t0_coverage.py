"""Write results/t0_coverage.json with T0-enabled vs T0-disabled split."""

from __future__ import annotations

import json

import polars as pl

from vsm.config import REPO_ROOT
from vsm.eval.coverage import t0_coverage_split
from vsm.mining.cache import atomic_write_json


def main() -> int:
    src = REPO_ROOT / "results" / "t0_flags.parquet"
    if not src.exists():
        raise SystemExit(f"missing {src}")
    flags = pl.read_parquet(src)
    report = t0_coverage_split(flags)
    dest = REPO_ROOT / "results" / "t0_coverage.json"
    atomic_write_json(report, dest)
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
