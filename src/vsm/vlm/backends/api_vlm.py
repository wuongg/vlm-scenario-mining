"""qwen3-vl-plus via OpenAI-compatible API. Video-file + fps required. Tokens from the response."""

from __future__ import annotations

import os
import random
import time
from pathlib import Path
from typing import Any

import polars as pl

from vsm.config import cfg_dict
from vsm.constants import API_MODEL_ID, COLS, GEOMETRY_B, PROVENANCE_GRID
from vsm.vlm.schema import Subfail, WindowOutput

_API = cfg_dict("model")["backends"]["api_vlm"]
_LEDGER = cfg_dict("model")["ledger"]


class TerminalQuotaError(RuntimeError):
    """403 / quota — kill-switch, do not retry."""


class RetryableRateError(RuntimeError):
    """429 — retryable with full-jitter backoff."""


def _sleep_full_jitter(attempt: int) -> None:
    base = float(_API["backoff_base_s"])
    cap = float(_API["backoff_max_s"])
    exp = min(cap, base * (2**attempt))
    time.sleep(random.uniform(0.0, exp))


def _is_terminal(status: int, body: str) -> bool:
    if status in {int(x) for x in _API["terminal_status"]}:
        return True
    lower = body.lower()
    return any(s.lower() in lower for s in _API["terminal_error_substrings"])


def append_ledger_row(row: dict[str, Any]) -> None:
    """Append one ledger row (pending before the call, filled after)."""

    from vsm.config import REPO_ROOT

    path = REPO_ROOT / str(_LEDGER["path"])
    path.parent.mkdir(parents=True, exist_ok=True)
    df = pl.DataFrame([row])
    if path.exists():
        prev = pl.read_csv(path)
        df = pl.concat([prev, df], how="diagonal_relaxed")
    df.write_csv(path)


class ApiVLM:
    """OpenAI-compatible Qwen3-VL-Plus. Free Quota Only is on by default."""

    name = "api_vlm"

    def __init__(self) -> None:
        self.model_id = str(_API["model_id"]) or API_MODEL_ID
        self.free_quota_only = bool(_API["free_quota_only"])
        self.base_url = os.environ.get(str(_API["base_url_env"]), str(_API["base_url_default"]))
        self.api_key = os.environ.get(str(_API["api_key_env"]), "")

    def _client(self):
        if not self.api_key:
            raise TerminalQuotaError(f"missing ${_API['api_key_env']} — Free Quota Only, no silent mock")
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise RuntimeError("install extra [api]") from exc
        return OpenAI(api_key=self.api_key, base_url=self.base_url)

    def mine_window(self, frames: Any, fps: float, prompts: dict[str, str]) -> WindowOutput:
        """Send a video-file + fps (required) and log usage.prompt_tokens from the response."""

        if fps <= 0:
            raise ValueError("fps param is required and must be > 0")
        rows = frames.to_dicts() if isinstance(frames, pl.DataFrame) else list(frames)
        window_id = "api"
        clip = None
        if rows:
            window_id = (
                f"{rows[0].get(COLS.SCENE_NAME.value)}_{rows[0].get(COLS.CAMERA.value)}_{rows[0].get(COLS.TS_US.value)}"
            )
            clip = Path(str(rows[0].get(COLS.CLIP_REF.value) or ""))
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
        }
        append_ledger_row(pending)
        client = self._client()
        video_ref = str(clip) if clip and clip.exists() else ""
        max_retries = int(_API["max_retries_429"])
        last_err: Exception | None = None
        response = None
        for attempt in range(max_retries + 1):
            pending["attempts"] = attempt + 1
            try:
                response = client.chat.completions.create(
                    model=self.model_id,
                    temperature=float(_API["temperature"]),
                    top_p=float(_API["top_p"]),
                    max_tokens=int(_API["max_tokens"]),
                    extra_body={"fps": fps} if True else {},
                    messages=[
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "video_url" if video_ref.startswith("http") else "video",
                                    "video": video_ref,
                                    "fps": fps,
                                },
                                {"type": "text", "text": prompts.get("user") or prompts.get("system") or ""},
                            ],
                        }
                    ],
                )
                last_err = None
                break
            except Exception as exc:  # noqa: BLE001 — classify then re-raise
                msg = str(exc)
                status = int(
                    getattr(exc, "status_code", 0) or getattr(getattr(exc, "response", None), "status_code", 0) or 0
                )
                if _is_terminal(status, msg):
                    pending["error_kind"] = "quota"
                    pending["t_end"] = time.time()
                    append_ledger_row(pending)
                    raise TerminalQuotaError(msg) from exc
                if status in {int(x) for x in _API["retryable_status"]} or "429" in msg:
                    last_err = RetryableRateError(msg)
                    pending["error_kind"] = "rate"
                    _sleep_full_jitter(attempt)
                    continue
                pending["error_kind"] = "network"
                pending["t_end"] = time.time()
                append_ledger_row(pending)
                raise
        if last_err is not None or response is None:
            pending["t_end"] = time.time()
            append_ledger_row(pending)
            raise last_err or RuntimeError("api_vlm returned no response")
        usage = getattr(response, "usage", None)
        prompt_tokens = int(getattr(usage, "prompt_tokens", 0) or 0)
        completion_tokens = int(getattr(usage, "completion_tokens", 0) or 0)
        pending["tokens_in"] = prompt_tokens
        pending["tokens_out"] = completion_tokens
        pending["t_end"] = time.time()
        pending["error_kind"] = None
        append_ledger_row(pending)
        choice = response.choices[0]
        finish = str(getattr(choice, "finish_reason", "stop") or "stop")
        truncated = finish == "length"
        text = ""
        if choice.message is not None:
            text = str(choice.message.content or "")
        none_ = "none" in text.lower() and "cut_in" not in text.lower()
        uncertain = "uncertain" in text.lower()
        return WindowOutput(
            window_id=window_id,
            findings=[],
            none_=none_ or (not text.strip()),
            uncertain=uncertain,
            parse_ok=not truncated,
            parse_subfail=Subfail.truncated if truncated else None,
            finish_reason=finish,
            completion_tokens=completion_tokens,
            truncated=truncated,
            prompt_version_hash="day1-api-skeleton",
            backend=self.name,
            geometry_mode=GEOMETRY_B,
            provenance=PROVENANCE_GRID,
            model_id=self.model_id,
        )

    def detect_objects(self, frames: Any) -> pl.DataFrame:
        del frames
        raise NotImplementedError("api_vlm is a VLM, not a detector")
