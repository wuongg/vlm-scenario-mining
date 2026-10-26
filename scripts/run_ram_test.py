"""Day-2 RAM gate for local Qwen3-VL-2B. Writes results/ram_test.json."""

from __future__ import annotations

import json
import time

from vsm.config import REPO_ROOT
from vsm.vlm.backends.local_vlm import LocalVLM


def main() -> int:
    backend = LocalVLM()
    ok = backend.ram_ok()
    payload = {
        "backend": backend.name,
        "model_id": backend.model_id,
        "min_ram_gb": backend.min_ram_gb,
        "ram_ok": ok,
        "run_today": backend.run_today,
        "import_torch": None,
        "load_attempted": False,
        "oom": False,
        "decision": "",
    }
    try:
        import torch

        payload["import_torch"] = True
        payload["cuda"] = bool(torch.cuda.is_available())
        payload["torch_version"] = str(torch.__version__)
    except Exception as exc:  # noqa: BLE001
        payload["import_torch"] = False
        payload["import_error"] = f"{type(exc).__name__}: {exc}"
    if not ok:
        payload["decision"] = "local-only branch — RAM gate failed; do not load 2B"
    else:
        payload["decision"] = "RAM gate passed; 2B load is opt-in (run_today=false). Not loaded today."
    payload["t_end"] = time.time()
    out = REPO_ROOT / "results" / "ram_test.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))
    return 0 if ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
