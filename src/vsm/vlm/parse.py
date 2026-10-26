"""Parse/repair live on Day 3. Day 1 exports the names so imports do not 404."""

from __future__ import annotations

from vsm.vlm.schema import Subfail, WindowOutput


def parse_output(raw: str, finish_reason: str, max_tokens: int) -> WindowOutput:
    """Day 3. Strip fence with 0 extra model calls; truncated if finish_reason=length."""

    del raw, max_tokens
    raise NotImplementedError("parse_output is Day 3")


def classify_subfail(raw: str) -> Subfail:
    """Day 3. Loop = n-gram repetition threshold."""

    del raw
    raise NotImplementedError("classify_subfail is Day 3")


def repair_once(output: WindowOutput, error_kind: str) -> WindowOutput:
    """Day 3. Fence→strip (0 call); truncation→max_tokens×1.5 once."""

    del output, error_kind
    raise NotImplementedError("repair_once is Day 3")
