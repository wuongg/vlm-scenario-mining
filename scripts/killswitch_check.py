"""Day 7 18:00 checklist. Prints configs/eval.yaml kill_switch items."""

from __future__ import annotations

from vsm.config import cfg_dict


def main() -> int:
    ks = cfg_dict("eval")["kill_switch"]
    print("deadline", ks["deadline_local"])
    for item in ks["checklist"]:
        print("- [ ]", item)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
