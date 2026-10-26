"""Re-run the frozen 10 ped_cross hand-checks under join v2."""

from __future__ import annotations

import json

import polars as pl

from vsm.config import REPO_ROOT
from vsm.eval.gt import JOIN_RULE_PED, pseudo_gt_pedestrian_cross


def main() -> int:
    prev = REPO_ROOT / "results" / "handchecks_pedestrian_cross.jsonl"
    old = [json.loads(line) for line in prev.read_text(encoding="utf-8").splitlines() if line.strip()]
    ids = [r["window_id"] for r in old]
    frames = pl.read_parquet(REPO_ROOT / "data" / "frames" / "frames.parquet")
    windows = pl.read_parquet(REPO_ROOT / "data" / "frames" / "windows.parquet")
    subset = windows.filter(pl.col("window_id").is_in(ids))
    pos = set(pseudo_gt_pedestrian_cross(subset, frames)["window_id"].to_list())
    print(json.dumps({"join_rule_version": JOIN_RULE_PED, "n_pos": len(pos), "pos": sorted(pos)}, indent=2))
    for rec in old:
        rec["join_rule_version"] = JOIN_RULE_PED
        rec["pseudo_positive_v2"] = rec["window_id"] in pos
        print(
            rec["window_id"],
            "v1_pos",
            rec.get("pseudo_positive"),
            "v2_pos",
            rec["window_id"] in pos,
            rec.get("note", "")[:80],
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
