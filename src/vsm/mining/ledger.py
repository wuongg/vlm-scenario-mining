"""Cost ledger. Checkpoint N=50, heartbeat file, error_kind taxonomy."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import polars as pl

from vsm.config import REPO_ROOT, cfg_dict

_LEDGER = cfg_dict("model")["ledger"]
LEDGER_PATH = REPO_ROOT / str(_LEDGER["path"])
HEARTBEAT_PATH = REPO_ROOT / str(_LEDGER["heartbeat"])
CHECKPOINT_N = int(_LEDGER["checkpoint_n"])


def append_ledger(row: dict[str, Any]) -> None:
    """Append one ledger row. error_kind ∈ {rate, quota, network, schema, oom} or empty."""

    LEDGER_PATH.parent.mkdir(parents=True, exist_ok=True)
    payload = dict(row)
    payload.setdefault("t_end", time.time())
    df = pl.DataFrame([payload])
    if LEDGER_PATH.exists():
        prev = pl.read_csv(LEDGER_PATH)
        df = pl.concat([prev, df], how="diagonal_relaxed")
    df.write_csv(LEDGER_PATH)


def write_heartbeat(folder: Path, extra: dict | None = None) -> None:
    HEARTBEAT_PATH.parent.mkdir(parents=True, exist_ok=True)
    body = {"utc": time.time(), "folder": str(folder), **(extra or {})}
    HEARTBEAT_PATH.write_text(json.dumps(body), encoding="utf-8")


def cache_hit_miss() -> dict[str, int]:
    if not LEDGER_PATH.exists():
        return {"hit": 0, "miss": 0, "n": 0}
    df = pl.read_csv(LEDGER_PATH)
    if "status" in df.columns:
        df = df.filter(pl.col("status") == "ok")
    if df.height == 0:
        return {"hit": 0, "miss": 0, "n": 0}
    if "cache_hit" not in df.columns:
        return {"hit": 0, "miss": 0, "n": df.height}
    hit = int(df.filter(pl.col("cache_hit") == True).height)  # noqa: E712
    return {"hit": hit, "miss": df.height - hit, "n": df.height}
