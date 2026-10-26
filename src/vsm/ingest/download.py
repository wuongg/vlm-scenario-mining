"""Download mini / map / CODA with curl resume. Mini is not self-SHA'd."""

from __future__ import annotations

import gzip
import hashlib
import json
import shutil
import subprocess
import tarfile
import zipfile
from pathlib import Path

from vsm.config import REPO_ROOT, cfg_dict

_INGEST = cfg_dict("ingest")


def _curl_bin() -> str:
    """Return curl.exe on Windows so PowerShell's curl alias is not used."""

    return "curl.exe" if shutil.which("curl.exe") else "curl"


def download_minifest(url: str, dest: Path) -> Path:
    """GET url into dest with curl -C - resume. Return dest. Does not SHA the mini tgz."""

    dest.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        _curl_bin(),
        "-C",
        "-",
        "-L",
        "--fail",
        "--retry",
        "5",
        "--retry-delay",
        "2",
        "-o",
        str(dest),
        url,
    ]
    subprocess.run(cmd, check=True)
    return dest


def assert_zip_magic(path: Path) -> None:
    """Raise if path is not a ZIP (official nuScenes map URL often returns an HTML login page)."""

    with path.open("rb") as fh:
        magic = fh.read(4)
    if magic[:2] != b"PK":
        raise RuntimeError(f"{path} is not a zip (magic={magic!r}); likely an HTML interstitial")


def _gzip_test(tgz: Path) -> None:
    """Scan the gzip stream; raise if the compressor reports corruption."""

    with gzip.open(tgz, "rb") as fh:
        while fh.read(1024 * 1024):
            pass


def _parse_json_member(tf: tarfile.TarFile, member: tarfile.TarInfo) -> object:
    extracted = tf.extractfile(member)
    if extracted is None:
        raise RuntimeError(f"cannot extract {member.name}")
    return json.loads(extracted.read().decode("utf-8"))


def integrity_gate(tgz: Path) -> None:
    """gzip-test + tar-list + parse required tables; scene==10; sample count in band."""

    if not tgz.exists():
        raise FileNotFoundError(tgz)
    _gzip_test(tgz)
    required = list(_INGEST["mini"]["tables_required"])
    expected_scenes = int(_INGEST["mini"]["expected_scenes"])
    lo, hi = [int(x) for x in _INGEST["mini"]["expected_samples_band"]]
    found: dict[str, object] = {}
    with tarfile.open(tgz, "r:gz") as tf:
        names = tf.getnames()
        if not names:
            raise RuntimeError("tgz lists zero members")
        for member in tf.getmembers():
            if not member.isfile() or not member.name.endswith(".json"):
                continue
            stem = Path(member.name).stem
            if stem in required and stem not in found:
                found[stem] = _parse_json_member(tf, member)
    missing = [t for t in required if t not in found]
    if missing:
        raise RuntimeError(f"required tables missing or unreadable: {missing}")
    scenes = found["scene"]
    samples = found["sample"]
    if not isinstance(scenes, list) or not isinstance(samples, list):
        raise RuntimeError("scene/sample JSON must be lists")
    if len(scenes) != expected_scenes:
        raise RuntimeError(f"scene count {len(scenes)} != {expected_scenes}")
    if not (lo <= len(samples) <= hi):
        raise RuntimeError(f"sample count {len(samples)} outside band [{lo}, {hi}]")


def sha256_file(path: Path) -> str:
    """Return hex sha256 of a file (used for map + CODA pins, never as the mini gate)."""

    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def extract_tgz(tgz: Path, dest: Path) -> Path:
    """Extract tgz into dest and return the directory that contains scene.json."""

    dest.mkdir(parents=True, exist_ok=True)
    with tarfile.open(tgz, "r:gz") as tf:
        tf.extractall(dest)
    matches = list(dest.rglob("scene.json"))
    if not matches:
        raise RuntimeError(f"scene.json not found under {dest}")
    return matches[0].parent.parent if matches[0].parent.name.startswith("v1.0") else matches[0].parent


def extract_zip(zip_path: Path, dest: Path) -> Path:
    """Extract a zip and return dest."""

    dest.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(dest)
    return dest


def sha256_dir(root: Path) -> str:
    """Return sha256 of sorted relative-path + file-digest pairs (CODA pin)."""

    digest = hashlib.sha256()
    files = sorted(p for p in root.rglob("*") if p.is_file() and ".cache" not in p.parts)
    for path in files:
        rel = path.relative_to(root).as_posix().encode("utf-8")
        digest.update(rel)
        digest.update(b"\0")
        digest.update(sha256_file(path).encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def pre_download_coda(dest: Path) -> None:
    """Download CODA-LM Test 500 and record sha256. HF Hub preferred; curl archive is often HTML."""

    dest.mkdir(parents=True, exist_ok=True)
    unpacked = dest / "unpacked"
    url = str(_INGEST["coda"]["url"])
    n_expected = int(_INGEST["coda"]["expected_n_test"])
    try:
        from huggingface_hub import snapshot_download

        snapshot_download(
            repo_id="KaiChen1998/coda-lm",
            repo_type="dataset",
            allow_patterns=["CODA-LM/Test/*"],
            local_dir=str(unpacked),
        )
    except Exception:
        filename = str(_INGEST["coda"]["filename"])
        archive = dest / filename
        download_minifest(url, archive)
        with archive.open("rb") as fh:
            magic = fh.read(4)
        if magic[:2] != b"PK":
            raise RuntimeError(
                "CODA archive was not a zip (HTML interstitial). Install huggingface-hub or pass a real zip."
            )
        extract_zip(archive, unpacked)
    tests = list(unpacked.rglob("test_*.json"))
    if len(tests) < n_expected:
        raise RuntimeError(f"CODA Test json count {len(tests)} < {n_expected}")
    digest = sha256_dir(unpacked)
    pin = str(_INGEST["coda"].get("sha256") or "")
    if pin and pin != digest:
        raise RuntimeError(f"CODA sha256 mismatch: got {digest} expected {pin}")
    manifest = {
        "name": _INGEST["coda"]["name"],
        "url": url,
        "sha256": digest,
        "n_test_json": len(tests),
        "path": str(unpacked.relative_to(REPO_ROOT)).replace("\\", "/"),
    }
    dest.joinpath("SHA256.txt").write_text(digest + "\n", encoding="utf-8")
    dest.joinpath("manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
