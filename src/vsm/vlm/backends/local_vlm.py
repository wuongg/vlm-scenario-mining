"""Qwen3-VL-2B skeleton. Day 1 does not run this unless RAM test passes (local-only / P2)."""

from __future__ import annotations

import os
from typing import Any

import polars as pl

from vsm.config import cfg_dict
from vsm.vlm.schema import WindowOutput

_LOCAL = cfg_dict("model")["backends"]["local_vlm"]


class LocalVLM:
    """Local 2B VLM. Precision tag is fp16|4bit. Not executed on Day 1 by default."""

    name = "local_vlm"

    def __init__(self) -> None:
        self.model_id = str(_LOCAL["model_id"])
        self.min_ram_gb = float(_LOCAL["min_ram_gb"])
        self.run_today = bool(_LOCAL["run_today"])

    def ram_ok(self) -> bool:
        """Return True if the host looks to have enough RAM for a 2B fp16 load."""

        total = 0.0
        try:
            import ctypes

            class MEMORYSTATUSEX(ctypes.Structure):
                _fields_ = [
                    ("dwLength", ctypes.c_ulong),
                    ("dwMemoryLoad", ctypes.c_ulong),
                    ("ullTotalPhys", ctypes.c_ulonglong),
                    ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong),
                    ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong),
                    ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("sullAvailExtendedVirtual", ctypes.c_ulonglong),
                ]

            stat = MEMORYSTATUSEX()
            stat.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
            ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat))
            total = stat.ullTotalPhys / (1024**3)
        except Exception:
            try:
                pages = int(os.sysconf("SC_PAGE_SIZE")) * int(os.sysconf("SC_PHYS_PAGES"))
                total = pages / (1024**3)
            except Exception:
                total = 0.0
        return total >= self.min_ram_gb

    def mine_window(self, frames: Any, fps: float, prompts: dict[str, str]) -> WindowOutput:
        """Refuse to run on Day 1 unless run_today and RAM gate pass."""

        del frames, prompts
        if fps <= 0:
            raise ValueError("fps param is required")
        if not self.run_today or not self.ram_ok():
            raise RuntimeError("local_vlm skeleton only — RAM test failed or run_today=false")
        raise RuntimeError("local_vlm weights not loaded on Day 1")

    def detect_objects(self, frames: Any) -> pl.DataFrame:
        del frames
        raise NotImplementedError("local_vlm is a VLM, not a detector")
