"""Regenerate presentation plots from frozen CSVs. Run twice → same bytes."""

from __future__ import annotations

import json

from vsm.viz.plots import write_all


def main() -> int:
    written = write_all()
    print(json.dumps(written, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
