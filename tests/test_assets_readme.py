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
