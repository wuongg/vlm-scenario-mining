"""One-call probe: does IteraCompute return logprobs on qwen/qwen3.8-27b?

Canary window, T=0, enable_thinking=false, 1 image. Do not name qwen3-vl-plus.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from vsm.config import REPO_ROOT, cfg_dict
from vsm.constants import COLS, WIN_COLS
from vsm.eval.gt import _attach_meta
from vsm.mining.cache import atomic_write_json
from vsm.vlm.backends.api_vlm import ApiVLM, _jpeg_b64, _load_key
from vsm.vlm.prompt import build_system_prompt

_EVAL = cfg_dict("eval")
_API = cfg_dict("model")["backends"]["api_vlm"]
_CANARY = list((_EVAL.get("canary") or {}).get("window_ids") or [])


def _middle_path(frames) -> Path | None:
    import polars as pl

    rows = frames.to_dicts() if isinstance(frames, pl.DataFrame) else list(frames)
    if not rows:
        return None
    mid = rows[len(rows) // 2]
    path = Path(str(mid.get(COLS.CLIP_REF.value) or ""))
    return path if path.exists() else None


def main() -> int:
    import polars as pl
    from openai import OpenAI

    if not _load_key():
        payload = {"ok": False, "supported": False, "error": "missing ITERACOMPUTE_API_KEY"}
        atomic_write_json(payload, REPO_ROOT / "results" / "logprob_probe.json")
        print(json.dumps(payload, indent=2))
        return 1
    backend = ApiVLM()
    frames = pl.read_parquet(REPO_ROOT / "data" / "frames" / "frames.parquet")
    windows = _attach_meta(pl.read_parquet(REPO_ROOT / "data" / "frames" / "windows.parquet"), frames)
    wid = _CANARY[0]
    w = windows.filter(pl.col(WIN_COLS.WINDOW_ID.value) == wid)
    rec = w.row(0, named=True)
    fids = list(rec.get(WIN_COLS.FRAME_IDS.value) or [])
    subset = frames.filter(pl.col(COLS.FRAME_ID.value).is_in(fids)).sort(COLS.TS_US.value)
    image = _middle_path(subset)
    prompt = build_system_prompt()
    content: list[dict[str, Any]] = [{"type": "text", "text": prompt}]
    if image is not None:
        content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{_jpeg_b64(image)}"}})
    client = OpenAI(api_key=backend.api_key, base_url=backend.base_url, timeout=180.0)
    t0 = time.time()
    error = None
    logprobs_obj = None
    raw = ""
    usage: dict[str, Any] = {}
    try:
        resp = client.chat.completions.create(
            model=backend.model_id,
            temperature=float(_API["temperature"]),
            top_p=float(_API["top_p"]),
            max_tokens=int(_API["max_tokens"]),
            logprobs=True,
            top_logprobs=5,
            extra_body={
                "chat_template_kwargs": {"enable_thinking": False},
                "reasoning": False,
            },
            messages=[{"role": "user", "content": content}],
        )
        choice = resp.choices[0]
        raw = str(choice.message.content or "")[:500]
        logprobs_obj = getattr(choice, "logprobs", None)
        u = getattr(resp, "usage", None)
        usage = {
            "prompt_tokens": int(getattr(u, "prompt_tokens", 0) or 0),
            "completion_tokens": int(getattr(u, "completion_tokens", 0) or 0),
        }
    except Exception as exc:  # noqa: BLE001 — probe must record the raw rejection
        error = f"{type(exc).__name__}: {exc}"
    content_logprobs = None
    if logprobs_obj is not None:
        content_logprobs = getattr(logprobs_obj, "content", None)
        if content_logprobs is None and isinstance(logprobs_obj, dict):
            content_logprobs = logprobs_obj.get("content")
    n_tokens = len(content_logprobs) if content_logprobs else 0
    supported = error is None and n_tokens > 0
    payload = {
        "window_id": wid,
        "model_id": backend.model_id,
        "ok": error is None,
        "supported": supported,
        "error": error,
        "n_logprob_tokens": n_tokens,
        "usage": usage,
        "latency_s": time.time() - t0,
        "raw_head": raw,
        "decision": (
            "logprobs accepted but stub — keep dummy confs"
            if (error is None and n_tokens <= 1)
            else ("logprob-of-verdict-token" if supported else "dummy confs stay; self-consistency not run")
        ),
        "usable_for_tau_sweep": bool(supported and n_tokens > 8),
    }
    atomic_write_json(payload, REPO_ROOT / "results" / "logprob_probe.json")
    print(json.dumps(payload, indent=2, default=str)[:3000])
    return 0 if error is None else 2


if __name__ == "__main__":
    raise SystemExit(main())
