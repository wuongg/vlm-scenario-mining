"""Cache keys + atomic parquet/json write. Ledger row pending BEFORE the call."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Callable

import polars as pl


def cache_key(window: dict, prompt_hash: str, model_id: str, decoding_hash: str) -> str:
    """SHA-256 of window_id + hashes. Stable across reruns of the same pass."""

    wid = str(window.get("window_id") or "")
    n = len(list(window.get("frame_ids") or []))
    payload = f"{wid}|{n}|{prompt_hash}|{model_id}|{decoding_hash}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def atomic_write_parquet(df: pl.DataFrame, dest: Path) -> None:
    """Write tmp then rename. No partial JSON/parquet at dest."""

    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".tmp")
    df.write_parquet(tmp)
    tmp.replace(dest)


def atomic_write_json(payload: Any, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    tmp.replace(dest)


def load_or_run(
    dest: Path,
    runner: Callable[[], pl.DataFrame],
    ledger_pending: dict | None = None,
) -> tuple[pl.DataFrame, dict]:
    """Return cached parquet if present; else run, atomic-write, mark cache_hit."""

    from vsm.mining.ledger import append_ledger, write_heartbeat

    row = dict(ledger_pending or {})
    if dest.exists():
        row["cache_hit"] = True
        append_ledger(row)
        write_heartbeat(dest.parent, extra={"cache_hit": dest.name})
        return pl.read_parquet(dest), row
    row["cache_hit"] = False
    append_ledger({**row, "status": "pending"})
    df = runner()
    atomic_write_parquet(df, dest)
    row["status"] = "ok"
    append_ledger(row)
    write_heartbeat(dest.parent, extra={"wrote": dest.name, "n": df.height})
    return df, row
