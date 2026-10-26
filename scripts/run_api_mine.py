"""Free-Quota-Only IteraCompute mining. Canary-lite first. GT-60 then corpus if budget remains.

Concurrency 5. Monthly 429 stops. $0. Never names the backend qwen3-vl-plus.
"""

from __future__ import annotations

import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

import polars as pl

from vsm.config import REPO_ROOT, cfg_dict
from vsm.constants import COLS, WIN_COLS
from vsm.eval.canary import canary_signals
from vsm.eval.gt import _attach_meta
from vsm.mining.cache import atomic_write_json
from vsm.vlm.backends.api_vlm import ApiVLM, TerminalQuotaError, _load_key
from vsm.vlm.prompt import build_system_prompt

_EVAL = cfg_dict("eval")
_API = cfg_dict("model")["backends"]["api_vlm"]
_CANARY_IDS = list((_EVAL.get("canary") or {}).get("window_ids") or [])
_CONCURRENCY = min(int(_API.get("concurrency") or 5), 5)
_BUDGET = int(_API.get("budget_tokens_gt60") or 800_000)
_MONTHLY = int(_API.get("monthly_token_allowance") or 30_000_000)


def _write_status(payload: dict[str, Any]) -> None:
    atomic_write_json(payload, REPO_ROOT / "results" / "api_status.json")


def _window_frames(windows: pl.DataFrame, frames: pl.DataFrame, wid: str) -> pl.DataFrame:
    w = windows.filter(pl.col(WIN_COLS.WINDOW_ID.value) == wid)
    if w.height == 0:
        return frames.head(0)
    rec = w.row(0, named=True)
    fids = list(rec.get(WIN_COLS.FRAME_IDS.value) or [])
    return frames.filter(pl.col(COLS.FRAME_ID.value).is_in(fids)).sort(COLS.TS_US.value)


def _record(wid: str, out: Any, role: str, err: str | None = None) -> dict[str, Any]:
    class_set = sorted({f.claim for f in (out.findings if out else [])}) or (["none"] if out and out.none_ else [])
    sub = None
    if out is not None and out.parse_subfail is not None:
        sub = out.parse_subfail.value
    usage_hash = (out.decoding_params_hash if out else "") or ""
    tokens_in = 0
    tokens_out = int(out.completion_tokens) if out else 0
    if "in" in usage_hash:
        try:
            tokens_in = int(usage_hash.split("|")[0].replace("in", ""))
        except ValueError:
            tokens_in = 0
    return {
        "window_id": wid,
        "role": role,
        "parse_ok": bool(out.parse_ok) if out else False,
        "parse_subfail": sub,
        "finish_reason": out.finish_reason if out else "",
        "truncated": bool(out.truncated) if out else False,
        "backend": "iteracompute",
        "model_id": str(_API["model_id"]),
        "prompt_version_hash": out.prompt_version_hash if out else "",
        "class_set": class_set,
        "none": bool(out.none_) if out else False,
        "n_findings": len(out.findings) if out else 0,
        "has_bbox": any(o.bbox_2d for f in (out.findings if out else []) for o in f.objects),
        "completion_tokens": tokens_out,
        "prompt_tokens": tokens_in,
        "error": err,
    }


def _mine_one(backend: ApiVLM, subset: pl.DataFrame, wid: str, role: str) -> dict[str, Any]:
    try:
        out = backend.mine_window(subset, fps=2.0, prompts={"user": build_system_prompt()})
        return _record(wid, out, role)
    except TerminalQuotaError as exc:
        rec = _record(wid, None, role, err=str(exc))
        rec["stop"] = "monthly_or_terminal"
        return rec
    except Exception as exc:  # noqa: BLE001 — miner must keep going unless monthly stop
        return _record(wid, None, role, err=f"{type(exc).__name__}: {exc}")


def _pool(backend: ApiVLM, jobs: list[tuple[str, pl.DataFrame, str]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    if not jobs:
        return out
    with ThreadPoolExecutor(max_workers=_CONCURRENCY) as pool:
        futs = {pool.submit(_mine_one, backend, subset, wid, role): wid for wid, subset, role in jobs}
        for fut in as_completed(futs):
            rec = fut.result()
            out.append(rec)
            keys = (
                "window_id",
                "parse_ok",
                "parse_subfail",
                "none",
                "n_findings",
                "prompt_tokens",
                "completion_tokens",
                "error",
            )
            print(json.dumps({k: rec[k] for k in keys if k in rec}))
            if rec.get("stop") == "monthly_or_terminal":
                pool.shutdown(wait=False, cancel_futures=True)
                break
    return out


def main() -> int:
    recorded: dict[str, Any] = {
        "recorded_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "backend": "iteracompute",
        "model_id": str(_API["model_id"]),
        "free_quota_only": True,
        "budget_usd": 0,
        "monthly_allowance_tokens": _MONTHLY,
        "budget_tokens_gt60": _BUDGET,
        "concurrency": _CONCURRENCY,
        "key_present": bool(_load_key()),
        "n_mined_gt60": 0,
        "n_mined_corpus": 0,
        "subfail": {},
        "token_usage": {"prompt": 0, "completion": 0, "total": 0},
        "canary": {"ran": False, "alarms": []},
        "status": "key_missing",
    }
    if not recorded["key_present"]:
        recorded["note"] = "ITERACOMPUTE_API_KEY missing — no API mining, no mock."
        _write_status(recorded)
        print(json.dumps(recorded, indent=2))
        return 0
    frames = pl.read_parquet(REPO_ROOT / "data" / "frames" / "frames.parquet")
    windows = _attach_meta(pl.read_parquet(REPO_ROOT / "data" / "frames" / "windows.parquet"), frames)
    backend = ApiVLM()
    canary_path = REPO_ROOT / "results" / "itera_canary.json"
    baseline_path = REPO_ROOT / "results" / "canary_baseline.json"
    if canary_path.exists() and not baseline_path.exists():
        payload = json.loads(canary_path.read_text(encoding="utf-8"))
        first = [
            {
                "window_id": r["window_id"],
                "class_set": ["none"] if r.get("none") else ["claim"],
                "completion_tokens": (r.get("usage") or {}).get("completion_tokens") or 0,
                "parse_ok": r.get("parse_ok"),
            }
            for r in payload.get("records") or []
        ]
        atomic_write_json(first, baseline_path)
        recorded["canary"] = {
            "ran": True,
            "source": "itera_canary.json",
            "baseline_written": True,
            "alarms": [],
            "first_run_n": len(first),
        }
    elif baseline_path.exists() and canary_path.exists():
        prev = {r["window_id"]: r for r in json.loads(baseline_path.read_text(encoding="utf-8"))}
        now_rows = json.loads(canary_path.read_text(encoding="utf-8")).get("records") or []
        alarms = []
        for rec in now_rows:
            old = prev.get(rec["window_id"])
            if not old:
                continue
            now = {
                "class_set": ["none"] if rec.get("none") else ["claim"],
                "completion_tokens": (rec.get("usage") or {}).get("completion_tokens") or 0,
                "parse_ok": rec.get("parse_ok"),
            }
            sig = canary_signals(old, now)
            if any(sig.values()):
                alarms.append({"window_id": rec["window_id"], **sig})
        recorded["canary"] = {"ran": True, "source": "itera_canary.json", "alarms": alarms}
    records: list[dict[str, Any]] = []
    gt_ids = list(_EVAL["gt_list"]["window_ids"])
    jobs = [(wid, _window_frames(windows, frames, wid), "gt60") for wid in gt_ids]
    try:
        mined = _pool(backend, jobs)
    except TerminalQuotaError as exc:
        recorded["status"] = "terminal_quota"
        recorded["error"] = str(exc)
        mined = []
    records.extend(mined)
    recorded["n_mined_gt60"] = len(mined)
    recorded["n_parse_ok"] = sum(1 for r in mined if r.get("parse_ok"))
    prompt_tok = sum(int(r.get("prompt_tokens") or 0) for r in records)
    comp_tok = sum(int(r.get("completion_tokens") or 0) for r in records)
    recorded["token_usage"] = {"prompt": prompt_tok, "completion": comp_tok, "total": prompt_tok + comp_tok}
    subfail: dict[str, int] = {}
    for r in records:
        kind = r.get("parse_subfail") or ("error" if r.get("error") else "ok")
        subfail[str(kind)] = subfail.get(str(kind), 0) + 1
    recorded["subfail"] = subfail
    stopped = any(r.get("stop") == "monthly_or_terminal" for r in records)
    if stopped:
        recorded["status"] = "stopped_monthly_429"
    elif recorded["n_mined_gt60"] >= len(gt_ids):
        recorded["status"] = "gt60_done"
    else:
        recorded["status"] = "gt60_partial"
    # corpus only if GT-60 complete and residual budget large
    if recorded["status"] == "gt60_done" and (prompt_tok + comp_tok) < (_MONTHLY * 0.05):
        corpus_ids = [
            str(w[WIN_COLS.WINDOW_ID.value])
            for w in windows.iter_rows(named=True)
            if str(w[WIN_COLS.WINDOW_ID.value]) not in gt_ids
        ]
        recorded["note"] = "quota residual large; corpus not auto-started this session (GT-60 first, 800K estimate)"
        del corpus_ids
    elif recorded["status"] == "gt60_done":
        recorded["note"] = "GT-60 complete; corpus skipped to keep free-tier headroom (30M/mo, ~800K used-scale)"
    atomic_write_json(records, REPO_ROOT / "results" / "api_mine_records.json")
    _write_status(recorded)
    print(json.dumps({k: v for k, v in recorded.items() if k != "canary"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
