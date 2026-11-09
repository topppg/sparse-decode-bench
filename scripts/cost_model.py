"""Predict sparse decode speed: t_sparse(T) = t_dense(T) − L·[a_dense(T) − a_sparse(T, B)] + o0 + o1·T.

t_dense: llama-bench dense decode at T. a: FLASH_ATTN_EXT microbenchmark times (test-backend-ops perf).
o0, o1: least-squares fit of the remaining sparse overhead at the calibration depths.

--models is ';'-separated "tag|gqa|kv_type|n_layer|T:B,T:B,..." (a T may repeat with several B). Sparse runs are read
from <tag>_d<T>_sparse.jsonl, or <tag>_d<T>_b<B>.jsonl when several budgets share a T. For the RTX 3090 run:
  "3b|3|f16|28|8192:2048,16384:3072,32768:4096,65536:14336,131072:28672;8b|4|q8_0|32|...;qwen3|4|q8_0|36|..."
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np

PERF = re.compile(r"FLASH_ATTN_EXT\((.*?)\):\s+\d+ runs -\s+([\d.]+) us/run")
Key = tuple[int, str, int, int]  # (gqa, K type, kv length, sparse budget or 0)


def perf_file(path: Path) -> dict[Key, float]:
    out: dict[Key, float] = {}
    for m in PERF.finditer(path.read_text()):
        p = dict(kv.split("=", 1) for kv in re.split(r",(?![^\[]*\])", m.group(1)))
        out[(int(p["nr23"].strip("[]").split(",")[0]), p["type_K"], int(p["kv"]), int(p["n_kv_max"]))] = float(m.group(2))
    return out


def median_runs(files: list[Path]) -> dict[Key, float]:
    runs = [perf_file(f) for f in files]
    keys = set().union(*runs) if runs else set()
    return {k: float(np.median([r[k] for r in runs if k in r])) for k in keys}


def tg_us(path: Path) -> float:
    ls = [x for x in path.read_text().splitlines() if x.startswith("{")] if path.exists() else []
    return 1e6 / float(json.loads(ls[0])["avg_ts"]) if ls else float("nan")


def sparse_file(bench: Path, tag: str, T: int, B: int) -> Path:
    one = bench / f"{tag}_d{T}_sparse.jsonl"
    return one if one.exists() else bench / f"{tag}_d{T}_b{B}.jsonl"


def predict(bench: Path, dense: dict[Key, float], sparse: dict[Key, float], tag: str, g: int, kv: str, n_layer: int,
            points: list[tuple[int, int]], calib: tuple[int, ...]) -> dict[str, object]:
    td = {T: tg_us(bench / f"{tag}_d{T}_dense.jsonl") for T, _ in points}
    ts = {(T, B): tg_us(sparse_file(bench, tag, T, B)) for T, B in points}
    base = {(T, B): td[T] - n_layer * (dense[(g, kv, T, 0)] - sparse[(g, kv, T, B)]) for T, B in points}
    cal = [(T, B) for T, B in points if T in calib]
    o1, o0 = (float(x) for x in np.polyfit([T for T, _ in cal], [ts[c] - base[c] for c in cal], 1))
    out: dict[str, object] = {"o0_us": o0, "o1_us_per_1k": 1e3 * o1}
    n_at = {T: sum(t == T for t, _ in points) for T, _ in points}
    for T, B in points:
        pred = base[(T, B)] + o0 + o1 * T
        rec: dict[str, object] = {"B": B, "dense_tps": 1e6 / td[T], "pred_sparse_tps": 1e6 / pred, "pred_S": td[T] / pred}
        if not np.isnan(ts[(T, B)]):
            m = ts[(T, B)]
            rec.update({"sparse_tps": 1e6 / m, "S": td[T] / m, "pred_err": (td[T] / pred) / (td[T] / m) - 1})
        out[str(T) if n_at[T] == 1 else f"{T}_{B}"] = rec
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--micro", required=True, help="dir with dense_*.txt and grp_g<gqa>_*.txt")
    ap.add_argument("--bench", required=True, help="dir with <tag>_d<T>_{dense,sparse}.jsonl")
    ap.add_argument("--models", required=True)
    ap.add_argument("--calib", default="8192,16384,32768")
    ap.add_argument("--out")
    a = ap.parse_args()
    micro = Path(a.micro)
    dense = median_runs(sorted(micro.glob("dense_*.txt")))
    sparse: dict[Key, float] = {}
    for g in (3, 4):
        sparse.update({k: v for k, v in median_runs(sorted(micro.glob(f"grp_g{g}_*.txt"))).items() if k[0] == g})
    calib = tuple(int(x) for x in a.calib.split(","))
    res: dict[str, object] = {}
    for spec in a.models.split(";"):
        tag, g, kv, n_layer, pts = spec.split("|")
        points = [(int(t), int(b)) for t, b in (p.split(":") for p in pts.split(","))]
        res[tag] = predict(Path(a.bench), dense, sparse, tag, int(g), kv, int(n_layer), points, calib)
    text = json.dumps(res, indent=1)
    if a.out:
        Path(a.out).write_text(text)
    print(text)


if __name__ == "__main__":
    main()
