"""Download mini + map expansion v1.3 + CODA-LM Test. Local only — never called from CI."""

from __future__ import annotations

from vsm.config import REPO_ROOT, cfg_dict
from vsm.ingest.download import (
    assert_zip_magic,
    download_minifest,
    extract_tgz,
    extract_zip,
    integrity_gate,
    pre_download_coda,
    sha256_file,
)

_INGEST = cfg_dict("ingest")


def _write_manifest(entries: list[dict]) -> None:
    path = REPO_ROOT / "data" / "manifest.yaml"
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["# Hashes only. Mini tgz is structure-gated, not self-SHA'd.\n"]
    for e in entries:
        lines.append(f"- name: {e['name']}\n  url: {e['url']}\n  path: {e['path']}\n  sha256: {e['sha256']}\n")
    path.write_text("".join(lines), encoding="utf-8")


def main() -> int:
    raw = REPO_ROOT / str(_INGEST["paths"]["raw"])
    raw.mkdir(parents=True, exist_ok=True)
    mini_tgz = raw / str(_INGEST["mini"]["filename"])
    print(f"download mini → {mini_tgz}")
    download_minifest(str(_INGEST["mini"]["url"]), mini_tgz)
    print("integrity_gate mini")
    integrity_gate(mini_tgz)
    mini_root = extract_tgz(mini_tgz, raw / "v1.0-mini")
    print(f"extracted mini_root={mini_root}")

    map_zip = raw / str(_INGEST["map"]["filename"])
    map_urls = [str(_INGEST["map"]["url"]), str(_INGEST["map"]["url_fallback"])]
    map_ok = False
    last = None
    used_url = map_urls[0]
    for url in map_urls:
        try:
            print(f"download map → {url}")
            if map_zip.exists():
                map_zip.unlink()
            download_minifest(url, map_zip)
            assert_zip_magic(map_zip)
            map_ok = True
            used_url = url
            break
        except Exception as exc:  # noqa: BLE001
            last = exc
            print(f"map url failed: {exc}")
    if not map_ok:
        raise SystemExit(f"map expansion download failed: {last}")
    map_sha = sha256_file(map_zip)
    map_root = REPO_ROOT / str(_INGEST["paths"]["map_root"])
    extract_zip(map_zip, map_root)
    print(f"map sha256={map_sha}")

    coda_dest = REPO_ROOT / str(_INGEST["paths"]["coda"])
    print("pre_download_coda")
    pre_download_coda(coda_dest)
    coda_sha = (coda_dest / "SHA256.txt").read_text(encoding="utf-8").strip()

    _write_manifest(
        [
            {
                "name": "nuscenes-mini",
                "url": _INGEST["mini"]["url"],
                "path": str(mini_tgz.relative_to(REPO_ROOT)).replace("\\", "/"),
                "sha256": "structure-gate-not-self-sha",
            },
            {
                "name": "nuscenes-map-expansion-v1.3",
                "url": used_url,
                "path": str(map_zip.relative_to(REPO_ROOT)).replace("\\", "/"),
                "sha256": map_sha,
            },
            {
                "name": "coda-lm-test-500",
                "url": _INGEST["coda"]["url"],
                "path": str(_INGEST["paths"]["coda"]),
                "sha256": coda_sha,
            },
        ]
    )
    print("done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
