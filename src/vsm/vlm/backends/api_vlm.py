"""IteraCompute OpenAI-compatible API. 1 image/request. Tokens from the response.

Do not name this backend qwen3-vl-plus. DashScope remains a commented fallback in model.yaml.
"""

from __future__ import annotations

import base64
import os
import random
import time
from io import BytesIO
from pathlib import Path
from typing import Any

import polars as pl
from PIL import Image

from vsm.config import REPO_ROOT, cfg_dict
from vsm.constants import API_MODEL_ID, COLS, GEOMETRY_B, PROVENANCE_GRID
from vsm.vlm.parse import parse_output, repair_once
from vsm.vlm.prompt import prompt_version_hash
from vsm.vlm.schema import WindowOutput

_API = cfg_dict("model")["backends"]["api_vlm"]
_LEDGER = cfg_dict("model")["ledger"]
_INPUT_WH = (960, 540)


class TerminalQuotaError(RuntimeError):
    """Monthly allowance or 401/403 — stop. No auto-overage on the free tier."""


class RetryableRateError(RuntimeError):
    """429 TPM / capacity — retry after Retry-After + full-jitter. Never in parallel copies."""


def _load_key() -> str:
    env_name = str(_API["api_key_env"])
    key = os.environ.get(env_name, "").strip().strip('"')
    if key:
        return key
    env_path = REPO_ROOT / "results" / "env"
    if env_path.exists():
        for line in env_path.read_text(encoding="utf-8").splitlines():
            if line.startswith(f"{env_name}="):
                return line.split("=", 1)[1].strip().strip('"')
    return ""


def _sleep_full_jitter(attempt: int, retry_after: float | None = None) -> None:
    if retry_after is not None and retry_after > 0:
        time.sleep(retry_after)
        return
    base = float(_API["backoff_base_s"])
    cap = float(_API["backoff_max_s"])
    exp = min(cap, base * (2**attempt))
    time.sleep(random.uniform(0.0, exp))


def _retry_after(exc: Exception) -> float | None:
    resp = getattr(exc, "response", None)
    if resp is None:
        return None
    headers = getattr(resp, "headers", {}) or {}
    raw = headers.get("Retry-After") or headers.get("retry-after")
    if not raw:
        return None
    try:
        return float(raw)
    except (TypeError, ValueError):
        return None


def _error_code(exc: Exception) -> str:
    body = str(exc)
    for code in list(_API.get("monthly_stop_codes") or []) + list(_API.get("retryable_error_codes") or []):
        if str(code) in body:
            return str(code)
    return ""


def _is_monthly_stop(status: int, body: str, code: str) -> bool:
    if code in {str(x) for x in (_API.get("monthly_stop_codes") or [])}:
        return True
    if status in {int(x) for x in _API["terminal_status"]}:
        return True
    lower = body.lower()
    return any(s.lower() in lower for s in _API["terminal_error_substrings"])


def append_ledger_row(row: dict[str, Any]) -> None:
    """Append one ledger row (pending before the call, filled after)."""

    path = REPO_ROOT / str(_LEDGER["path"])
    path.parent.mkdir(parents=True, exist_ok=True)
    df = pl.DataFrame([row])
    if path.exists():
        prev = pl.read_csv(path)
        df = pl.concat([prev, df], how="diagonal_relaxed")
    df.write_csv(path)


def _jpeg_b64(path: Path) -> str:
    image = Image.open(path).convert("RGB")
    image = image.resize(_INPUT_WH)
    buf = BytesIO()
    image.save(buf, format="JPEG", quality=75)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _middle_image(frames: Any) -> Path | None:
    rows = frames.to_dicts() if isinstance(frames, pl.DataFrame) else list(frames)
    if not rows:
        return None
    mid = rows[len(rows) // 2]
    path = Path(str(mid.get(COLS.CLIP_REF.value) or ""))
    return path if path.exists() else None


class ApiVLM:
    """IteraCompute qwen/qwen3.8-27b. Free tier. 1 middle image per window."""

    name = "iteracompute"

    def __init__(self) -> None:
        self.model_id = str(_API["model_id"]) or API_MODEL_ID
        self.free_quota_only = bool(_API["free_quota_only"])
        self.base_url = os.environ.get(str(_API["base_url_env"]), str(_API["base_url_default"]))
        self.api_key = _load_key()

    def _client(self):
        if not self.api_key:
            raise TerminalQuotaError(f"missing ${_API['api_key_env']} — Free Quota Only, no silent mock")
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise RuntimeError("install extra [api]") from exc
        return OpenAI(api_key=self.api_key, base_url=self.base_url, timeout=180.0)

    def mine_window(self, frames: Any, fps: float, prompts: dict[str, str]) -> WindowOutput:
        """Send 1 middle image (reseller cap). fps kept in the signature; not sent."""

        del fps
        rows = frames.to_dicts() if isinstance(frames, pl.DataFrame) else list(frames)
        window_id = "api"
        if rows:
            window_id = (
                f"{rows[0].get(COLS.SCENE_NAME.value)}_{rows[0].get(COLS.CAMERA.value)}_{rows[0].get(COLS.TS_US.value)}"
            )
        image = _middle_image(frames)
        prompt = prompts.get("user") or prompts.get("system") or ""
        content: list[dict[str, Any]] = [{"type": "text", "text": prompt}]
        if image is not None:
            content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{_jpeg_b64(image)}"}})
        pending = {
            "window_id": window_id,
            "backend": self.name,
            "model_id": self.model_id,
            "cache_hit": False,
            "tokens_in": 0,
            "tokens_out": 0,
            "t_start": time.time(),
            "t_end": 0.0,
            "attempts": 0,
            "error_kind": None,
            "precision_tag": "api",
            "status": "pending",
        }
        append_ledger_row(pending)
        client = self._client()
        max_retries = int(_API["max_retries_429"])
        last_err: Exception | None = None
        response = None
        raw = ""
        finish = "stop"
        for attempt in range(max_retries + 1):
            pending["attempts"] = attempt + 1
            try:
                response = client.chat.completions.create(
                    model=self.model_id,
                    temperature=float(_API["temperature"]),
                    top_p=float(_API["top_p"]),
                    max_tokens=int(_API["max_tokens"]),
                    extra_body={
                        "chat_template_kwargs": {"enable_thinking": bool(_API.get("enable_thinking", False))},
                        "reasoning": bool(_API.get("enable_thinking", False)),
                    },
                    messages=[{"role": "user", "content": content}],
                )
                last_err = None
                break
            except Exception as exc:  # noqa: BLE001 — classify then re-raise
                msg = str(exc)
                status = int(
                    getattr(exc, "status_code", 0) or getattr(getattr(exc, "response", None), "status_code", 0) or 0
                )
                code = _error_code(exc)
                if _is_monthly_stop(status, msg, code):
                    pending["error_kind"] = "quota"
                    pending["t_end"] = time.time()
                    pending["status"] = "stop"
                    append_ledger_row(pending)
                    raise TerminalQuotaError(msg) from exc
                retryable = status in {int(x) for x in _API["retryable_status"]} or code in {
                    str(x) for x in (_API.get("retryable_error_codes") or [])
                }
                if retryable:
                    last_err = RetryableRateError(msg)
                    pending["error_kind"] = "rate"
                    _sleep_full_jitter(attempt, _retry_after(exc))
                    continue
                pending["error_kind"] = "network"
                pending["t_end"] = time.time()
                pending["status"] = "error"
                append_ledger_row(pending)
                raise
        if last_err is not None or response is None:
            pending["t_end"] = time.time()
            pending["status"] = "error"
            append_ledger_row(pending)
            raise last_err or RuntimeError("api_vlm returned no response")
        usage = getattr(response, "usage", None)
        prompt_tokens = int(getattr(usage, "prompt_tokens", 0) or 0)
        completion_tokens = int(getattr(usage, "completion_tokens", 0) or 0)
        pending["tokens_in"] = prompt_tokens
        pending["tokens_out"] = completion_tokens
        pending["t_end"] = time.time()
        pending["error_kind"] = None
        pending["status"] = "ok"
        append_ledger_row(pending)
        choice = response.choices[0]
        finish = str(getattr(choice, "finish_reason", "stop") or "stop")
        raw = str(choice.message.content or "") if choice.message is not None else ""
        parsed = parse_output(raw, finish, int(_API["max_tokens"]), window_id=window_id)
        repair_kind = "none"
        if not parsed.parse_ok and "```" in raw:
            parsed, repair_kind = repair_once(parsed, "fence", raw=raw)
        if parsed.truncated:
            parsed, repair_kind = repair_once(parsed, "truncation", raw=raw)
        parsed.prompt_version_hash = prompt_version_hash()
        parsed.backend = self.name
        parsed.model_id = self.model_id
        parsed.completion_tokens = completion_tokens
        parsed.geometry_mode = GEOMETRY_B
        parsed.provenance = PROVENANCE_GRID
        # stash usage for the miner without changing the frozen schema fields
        parsed.decoding_params_hash = f"in{prompt_tokens}|out{completion_tokens}|repair={repair_kind}"
        return parsed

    def detect_objects(self, frames: Any) -> pl.DataFrame:
        del frames
        raise NotImplementedError("api_vlm is a VLM, not a detector")
