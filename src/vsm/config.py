"""Load YAML configs and stamp provenance on every artifact."""

from __future__ import annotations

import hashlib
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = REPO_ROOT / "configs"
MANIFEST_PATH = REPO_ROOT / "data" / "manifest.yaml"


class Settings(BaseModel):
    """Loose YAML document. Nested keys stay dicts."""

    model_config = ConfigDict(extra="allow")


def load_config(path: Path) -> Settings:
    """Return a pydantic Settings model for one YAML file."""

    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if raw is None:
        raw = {}
    if not isinstance(raw, dict):
        raise ValueError(f"config root must be a mapping: {path}")
    return Settings.model_validate(raw)


def load_named(name: str) -> Settings:
    """Return Settings for configs/<name>.yaml (name without suffix is ok)."""

    path = CONFIG_DIR / name if name.endswith(".yaml") else CONFIG_DIR / f"{name}.yaml"
    return load_config(path)


def cfg_dict(name: str) -> dict[str, Any]:
    """Return a plain dict for configs/<name>.yaml."""

    return load_named(name).model_dump()


def config_hash(cfg: Settings) -> str:
    """Return SHA-256 of the config rendered as sorted-key YAML."""

    payload = cfg.model_dump(mode="json")
    canonical = yaml.safe_dump(payload, sort_keys=True, allow_unicode=True)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _git_sha() -> str:
    try:
        sha = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=REPO_ROOT,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError, OSError):
        return "unknown"
    try:
        status = subprocess.check_output(
            ["git", "status", "--porcelain"],
            cwd=REPO_ROOT,
            text=True,
            stderr=subprocess.DEVNULL,
        )
    except (subprocess.CalledProcessError, FileNotFoundError, OSError):
        status = ""
    if status.strip():
        return f"{sha}-dirty"
    return sha


def provenance(config_path: Path | None = None) -> dict[str, str]:
    """Return git/config/data identity for a generated artifact."""

    cfg_hash = ""
    if config_path is not None:
        cfg_hash = config_hash(load_config(config_path))
    data_hash = ""
    if MANIFEST_PATH.exists():
        data_hash = hashlib.sha256(MANIFEST_PATH.read_bytes()).hexdigest()
    return {
        "git_sha": _git_sha(),
        "config_hash": cfg_hash,
        "data_hash": data_hash,
        "utc_now": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
