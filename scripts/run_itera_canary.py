"""BƯỚC 0 — IteraCompute canary on the 5 frozen windows. Do not bump pre-reg from this script.

Reads ITERACOMPUTE_API_KEY from the environment or results/env. Never prints the key.
"""

from __future__ import annotations

import base64
import json
import os
import re
import time
from io import BytesIO
from pathlib import Path
from typing import Any

import polars as pl
from PIL import Image

from vsm.config import REPO_ROOT, cfg_dict
from vsm.constants import COLS, WIN_COLS
from vsm.eval.gt import _attach_meta, load_gt_labels
from vsm.mining.cache import atomic_write_json
from vsm.vlm.parse import parse_output
from vsm.vlm.prompt import build_system_prompt, prompt_version_hash

_EVAL = cfg_dict("eval")
_CANARY = list((_EVAL.get("canary") or {}).get("window_ids") or [])
BASE_URL = "https://api.iteracompute.com/v1"
MODEL_ID = "qwen/qwen3.8-27b"
MAX_TOKENS = 1024
TEMPERATURE = 0.0
TOP_P = 1.0
FPS = 2.0
INPUT_WH = (960, 540)


def _load_key() -> str:
    key = os.environ.get("ITERACOMPUTE_API_KEY", "").strip().strip('"')
    if key:
        return key
    env_path = REPO_ROOT / "results" / "env"
    if env_path.exists():
        for line in env_path.read_text(encoding="utf-8").splitlines():
            if line.startswith("ITERACOMPUTE_API_KEY="):
                return line.split("=", 1)[1].strip().strip('"')
    raise SystemExit("ITERACOMPUTE_API_KEY missing (env or results/env)")


def _client(api_key: str):
    from openai import OpenAI

    return OpenAI(base_url=BASE_URL, api_key=api_key, timeout=180.0)


def _window_frames(windows: pl.DataFrame, frames: pl.DataFrame, wid: str) -> pl.DataFrame:
    w = windows.filter(pl.col(WIN_COLS.WINDOW_ID.value) == wid)
    if w.height == 0:
        raise SystemExit(f"missing window {wid}")
    rec = w.row(0, named=True)
    fids = list(rec.get(WIN_COLS.FRAME_IDS.value) or [])
    return frames.filter(pl.col(COLS.FRAME_ID.value).is_in(fids)).sort(COLS.TS_US.value)


def _jpeg_b64(path: Path) -> str:
    image = Image.open(path).convert("RGB")
    image = image.resize(INPUT_WH)
    buf = BytesIO()
    image.save(buf, format="JPEG", quality=75)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _select_frames(subset: pl.DataFrame, n_keep: int) -> pl.DataFrame:
    if n_keep <= 0 or subset.height <= n_keep:
        return subset
    if n_keep == 1:
        return subset.slice(subset.height // 2, 1)
    return subset.head(n_keep)


def _image_parts(subset: pl.DataFrame) -> list[dict[str, Any]]:
    parts: list[dict[str, Any]] = []
    for row in subset.to_dicts():
        path = Path(str(row.get(COLS.CLIP_REF.value) or ""))
        if not path.exists():
            continue
        parts.append(
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{_jpeg_b64(path)}"},
            }
        )
    return parts


def _has_bbox(payload: Any) -> bool:
    if isinstance(payload, dict):
        if "bbox_2d" in payload or "bbox" in payload:
            box = payload.get("bbox_2d") or payload.get("bbox")
            return isinstance(box, (list, tuple)) and len(box) == 4
        return any(_has_bbox(v) for v in payload.values())
    if isinstance(payload, list):
        return any(_has_bbox(v) for v in payload)
    return False


def _parse_json(raw: str) -> Any:
    text = (raw or "").strip()
    m = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL | re.IGNORECASE)
    if m:
        text = m.group(1).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        brace = re.search(r"\{.*\}", text, re.DOTALL)
        if brace:
            try:
                return json.loads(brace.group(0))
            except json.JSONDecodeError:
                return None
        return None


def _call(
    client: Any,
    content: list[dict[str, Any]],
    *,
    use_video: bool,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "model": MODEL_ID,
        "messages": [{"role": "user", "content": content}],
        "temperature": TEMPERATURE,
        "top_p": TOP_P,
        "max_tokens": MAX_TOKENS,
        "extra_body": {
            "chat_template_kwargs": {"enable_thinking": False},
            "reasoning": False,
            **({"fps": FPS} if use_video else {}),
        },
    }
    if extra:
        kwargs.update(extra)
    t0 = time.time()
    try:
        resp = client.chat.completions.create(**kwargs)
    except Exception as exc:  # noqa: BLE001 — canary must record the raw failure
        return {
            "ok": False,
            "error": f"{type(exc).__name__}: {exc}",
            "latency_s": time.time() - t0,
            "raw": "",
            "finish_reason": "",
            "usage": {},
        }
    choice = resp.choices[0]
    raw = str(choice.message.content or "")
    usage = getattr(resp, "usage", None)
    return {
        "ok": True,
        "error": None,
        "latency_s": time.time() - t0,
        "raw": raw,
        "finish_reason": str(getattr(choice, "finish_reason", "") or ""),
        "usage": {
            "prompt_tokens": int(getattr(usage, "prompt_tokens", 0) or 0),
            "completion_tokens": int(getattr(usage, "completion_tokens", 0) or 0),
            "total_tokens": int(getattr(usage, "total_tokens", 0) or 0),
        },
    }


def main() -> int:
    api_key = _load_key()
    client = _client(api_key)
    prompt = build_system_prompt()
    prompt_hash = prompt_version_hash(prompt)
    frames = pl.read_parquet(REPO_ROOT / "data" / "frames" / "frames.parquet")
    windows = _attach_meta(pl.read_parquet(REPO_ROOT / "data" / "frames" / "windows.parquet"), frames)
    labels = {r["window_id"]: r for r in load_gt_labels().to_dicts()}
    first = _CANARY[0]
    subset = _window_frames(windows, frames, first)
    clip = Path(str(subset[COLS.CLIP_REF.value][0]))
    video_probe = _call(
        client,
        [
            {"type": "text", "text": prompt},
            {"type": "video_url", "video_url": {"url": str(clip)}},
        ],
        use_video=True,
    )
    input_mode = "video_file" if video_probe.get("ok") and video_probe.get("raw") else "image_list"
    six_probe: dict[str, Any] = {}
    if input_mode != "video_file":
        video_probe["fallback_reason"] = video_probe.get("error") or "empty content"
        six_probe = _call(
            client,
            [{"type": "text", "text": prompt}, *_image_parts(subset)],
            use_video=False,
        )
        if six_probe.get("ok") and six_probe.get("raw"):
            input_mode = "image_list_6"
        else:
            input_mode = "image_middle_1"
            six_probe["fallback_reason"] = six_probe.get("error") or "empty content"
    records: list[dict[str, Any]] = []
    checks = {"schema_json": [], "grounding_bbox": [], "none_slot": [], "latency_s": [], "tokens": []}
    for wid in _CANARY:
        subset = _window_frames(windows, frames, wid)
        if input_mode == "video_file":
            content = [
                {"type": "text", "text": prompt},
                {"type": "video_url", "video_url": {"url": str(Path(str(subset[COLS.CLIP_REF.value][0])))}},
            ]
        else:
            n_keep = 6 if input_mode == "image_list_6" else 1
            content = [{"type": "text", "text": prompt}, *_image_parts(_select_frames(subset, n_keep))]
        call = _call(client, content, use_video=input_mode == "video_file")
        parsed = parse_output(call.get("raw") or "", call.get("finish_reason") or "stop", MAX_TOKENS, wid)
        payload = _parse_json(call.get("raw") or "")
        schema_ok = payload is not None
        grounding = _has_bbox(payload) if payload is not None else False
        gt = labels.get(wid) or {}
        gt_none = "none" in (gt.get("class_set") or ["none"])
        invented = bool(parsed.findings) and gt_none
        rec = {
            "window_id": wid,
            "gt_class_set": gt.get("class_set"),
            "input_mode": input_mode,
            "n_frames": subset.height,
            "latency_s": round(float(call.get("latency_s") or 0), 3),
            "usage": call.get("usage"),
            "finish_reason": call.get("finish_reason"),
            "error": call.get("error"),
            "schema_json": bool(schema_ok and payload is not None),
            "fence_stripped": "```" in (call.get("raw") or ""),
            "grounding_bbox": grounding,
            "parse_ok": parsed.parse_ok,
            "parse_subfail": parsed.parse_subfail.value if parsed.parse_subfail else None,
            "none": parsed.none_,
            "uncertain": parsed.uncertain,
            "n_findings": len(parsed.findings),
            "invented_findings_on_none_gt": invented,
            "raw_head": (call.get("raw") or "")[:800],
            "prompt_version_hash": prompt_hash,
            "backend": "iteracompute",
            "model_id": MODEL_ID,
        }
        records.append(rec)
        checks["schema_json"].append(rec["schema_json"])
        checks["grounding_bbox"].append(rec["grounding_bbox"])
        checks["none_slot"].append(not invented)
        checks["latency_s"].append(rec["latency_s"])
        checks["tokens"].append(call.get("usage") or {})
        print(
            json.dumps(
                {
                    k: rec[k]
                    for k in (
                        "window_id",
                        "schema_json",
                        "grounding_bbox",
                        "none",
                        "n_findings",
                        "latency_s",
                        "usage",
                        "parse_subfail",
                    )
                },
                ensure_ascii=True,
            )
        )
    n = len(records) or 1
    schema_pass = sum(checks["schema_json"]) >= max(1, int(0.8 * n)) and any(checks["schema_json"])
    grounding_pass = any(checks["grounding_bbox"])
    none_pass = all(checks["none_slot"]) if checks["none_slot"] else False
    verdict = "detector_only"
    if schema_pass and grounding_pass:
        verdict = "t1_t2_full"
    elif schema_pass and not grounding_pass:
        verdict = "screener_t1_no_grounding"
    summary = {
        "backend": "iteracompute",
        "model_id": MODEL_ID,
        "base_url": BASE_URL,
        "input_mode": input_mode,
        "video_probe": {
            k: video_probe.get(k) for k in ("ok", "error", "latency_s", "finish_reason") if k in video_probe
        }
        | {"fallback_reason": video_probe.get("fallback_reason")},
        "image6_probe": {k: six_probe.get(k) for k in ("ok", "error", "latency_s", "finish_reason") if k in six_probe}
        | {"fallback_reason": six_probe.get("fallback_reason")},
        "input_tradeoff": (
            "reseller allows at most 1 image/request; video-file 503/unavailable; "
            "GT-60 mining uses 1 middle keyframe (token saving vs 6 serial calls)"
            if input_mode == "image_middle_1"
            else None
        ),
        "decoding": {"temperature": TEMPERATURE, "top_p": TOP_P, "max_tokens": MAX_TOKENS, "enable_thinking": False},
        "prompt_version_hash": prompt_hash,
        "checks": {
            "1_schema_json": {"pass": schema_pass, "n_ok": sum(checks["schema_json"]), "n": n},
            "2_grounding_bbox": {"pass": grounding_pass, "n_ok": sum(checks["grounding_bbox"]), "n": n},
            "3_none_uncertain": {"pass": none_pass, "n_ok": sum(checks["none_slot"]), "n": n},
            "4_latency_tokens": {
                "latency_s": checks["latency_s"],
                "mean_latency_s": sum(checks["latency_s"]) / n,
                "usage": checks["tokens"],
            },
        },
        "verdict": verdict,
        "grounding_note": (
            None
            if grounding_pass
            else (
                "grounding not supported on this deployment — "
                "GT-bounded hallucination rate limited to class-level claims "
                "(no bbox audit on API tier)"
            )
        ),
        "records": records,
    }
    dest = REPO_ROOT / "results" / "itera_canary.json"
    atomic_write_json(summary, dest)
    print(
        json.dumps(
            {k: summary[k] for k in ("verdict", "input_mode", "checks") if k != "records"}, indent=2, default=str
        )
    )
    print("wrote", dest)
    return 0 if schema_pass else 2


if __name__ == "__main__":
    raise SystemExit(main())
