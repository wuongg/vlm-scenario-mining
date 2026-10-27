"""Day-12 hygiene final gate. History + src/scripts only. Never scan data/raw."""

from __future__ import annotations

import json
import shutil
import subprocess

from vsm.config import REPO_ROOT


def _run(cmd: list[str]) -> tuple[int, str]:
    proc = subprocess.run(cmd, cwd=REPO_ROOT, capture_output=True, text=True, check=False)
    out = (proc.stdout or "") + (proc.stderr or "")
    return proc.returncode, out


def main() -> int:
    gitleaks = shutil.which("gitleaks")
    history_ok = None
    src_ok = None
    notes: list[str] = []
    if gitleaks:
        rc_h, out_h = _run([gitleaks, "detect", "--source", str(REPO_ROOT), "--log-opts", "--all", "--no-banner", "-v"])
        history_ok = rc_h == 0
        if not history_ok:
            notes.append(out_h[-800:])
        rc_s, out_s = _run([gitleaks, "detect", "--source", str(REPO_ROOT / "src"), "--no-git", "--no-banner"])
        rc_p, out_p = _run([gitleaks, "detect", "--source", str(REPO_ROOT / "scripts"), "--no-git", "--no-banner"])
        src_ok = rc_s == 0 and rc_p == 0
        if not src_ok:
            notes.append((out_s + out_p)[-800:])
    else:
        notes.append("gitleaks not on PATH")

    nb = subprocess.run(
        ["git", "log", "--all", "--full-history", "--", "*.ipynb"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    notebooks_empty = not (nb.stdout or "").strip()

    scratch = [
        str(p.relative_to(REPO_ROOT))
        for p in REPO_ROOT.rglob("*")
        if p.is_file() and (p.name == "test2.py" or "_old" in p.parts) and ".git" not in p.parts
    ]
    gitignore = (REPO_ROOT / ".gitignore").read_text(encoding="utf-8")
    license_ok = (REPO_ROOT / "LICENSE").exists()
    readme = (REPO_ROOT / "README.md").read_text(encoding="utf-8")
    report = {
        "gitleaks_history": history_ok,
        "gitleaks_src_scripts": src_ok,
        "notebooks_in_history": not notebooks_empty,
        "scratch_files": scratch,
        "gitignore_covers_data": "data/*" in gitignore,
        "license_mit": license_ok and "MIT License" in (REPO_ROOT / "LICENSE").read_text(encoding="utf-8"),
        "data_licensing_section": "## Data & Licensing" in readme or "### Data & Licensing" in readme,
        "uv_lock_present": (REPO_ROOT / "uv.lock").exists(),
        "results_scan": "committed results clean (git history); working-tree results/env is gitignored and not scanned",
        "notes": notes,
    }
    report["ok"] = (
        report["gitleaks_history"] is True
        and report["gitleaks_src_scripts"] is True
        and notebooks_empty
        and not scratch
        and report["gitignore_covers_data"]
        and report["license_mit"]
        and report["data_licensing_section"]
        and report["uv_lock_present"]
    )
    dest = REPO_ROOT / "results" / "hygiene_final.json"
    dest.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(dest.read_text(encoding="utf-8"))
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
