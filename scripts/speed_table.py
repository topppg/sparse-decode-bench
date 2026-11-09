"""Decode tokens/s and speedup from pairs of llama-bench JSONL files (dense, sparse).

Args: label=dense.jsonl,sparse.jsonl ... For prompt+generation runs (wall-clock) it prints seconds instead.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path


def row(path: Path) -> dict[str, float]:
    return json.loads(next(x for x in path.read_text().splitlines() if x.startswith("{")))


def main() -> None:
    for arg in sys.argv[1:]:
        label, files = arg.split("=", 1)
        d, s = (row(Path(f)) for f in files.split(","))
        if d["n_prompt"] > 0:
            td, ts = d["avg_ns"] / 1e9, s["avg_ns"] / 1e9
            print(f"{label:30s} dense {td:6.0f} s  sparse {ts:6.0f} s  end-to-end {td / ts:.2f}x")
        else:
            print(f"{label:30s} dense {d['avg_ts']:6.1f}  sparse {s['avg_ts']:6.1f} tok/s  speedup {s['avg_ts'] / d['avg_ts']:.2f}x")


if __name__ == "__main__":
    main()
