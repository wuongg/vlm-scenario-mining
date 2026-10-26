"""Backend protocols. Implementations live in sibling modules."""

from __future__ import annotations

from typing import Any, Protocol

import polars as pl

from vsm.vlm.schema import WindowOutput


class VLMBackend(Protocol):
    def mine_window(self, frames: Any, fps: float, prompts: dict[str, str]) -> WindowOutput:
        """Return a schema-valid WindowOutput for one window of frames."""
        ...


class DetectorBackend(Protocol):
    def detect_objects(self, frames: Any) -> pl.DataFrame:
        """Return boxes+logits; caller caches to Parquet."""
        ...
