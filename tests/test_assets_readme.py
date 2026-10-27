"""make_assets is deterministic; README T2 numbers match the frozen CSV."""

from __future__ import annotations

import hashlib
import runpy
from pathlib import Path

from vsm.config import REPO_ROOT
from vsm.viz.plots import write_all


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_make_assets_reproducible(tmp_path: Path) -> None:
    first = write_all(tmp_path)
    hashes = {k: _sha(Path(p)) for k, p in first.items()}
    write_all(tmp_path)
    for key, path in first.items():
        p = Path(path)
        assert p.exists() and p.stat().st_size > 800
        assert _sha(p) == hashes[key]


def test_readme_selfcheck_script() -> None:
    ns = runpy.run_path(str(REPO_ROOT / "scripts" / "readme_selfcheck.py"))
    assert ns["main"]() == 0


def test_readme_t1_wordcount_and_primary() -> None:
    import re

    from vsm.eval.metrics import interpretation_gate, readme_primary_number

    text = (REPO_ROOT / "README.md").read_text(encoding="utf-8")
    t1 = text.split("## T1")[1].split("## T2")[0]
    words = re.findall(r"[A-Za-z0-9$]+", re.sub(r"!\[.*?\]\(.*?\)", " ", t1))
    assert len(words) <= 150
    assert "1.0" in t1
    assert "results/demo.gif" in t1
    assert interpretation_gate(1.0, readme_primary_number(text))["ok"] is True
