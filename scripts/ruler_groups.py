"""Group RULER per-task scores into single-needle, multi-needle and other tasks (the README group table)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

GROUPS = {"single needle": lambda t: t.startswith("niah_single"),
          "multi-needle": lambda t: t.startswith("niah_multi"),
          "other": lambda t: not t.startswith("niah")}


def main() -> None:
    for path in sys.argv[1:]:
        pt = json.loads(Path(path).read_text())["per_task"]
        dense, sparse = list(pt)
        print(f"{path} ({sparse} vs {dense})")
        for g, sel in GROUPS.items():
            ts = [t for t in pt[dense] if sel(t)]
            d = sum(pt[dense][t] for t in ts) / len(ts)
            s = sum(pt[sparse][t] for t in ts) / len(ts)
            print(f"  {g:14s} {len(ts)} tasks  {d:5.1f} -> {s:5.1f}  ({s - d:+.1f})")


if __name__ == "__main__":
    main()
