"""Effects of single changes on the six-task RULER mean, paired over the same prompts (README "Why mv failed").

Usage: ruler_effects.py --meta M --model TOK da=GEN [db=GEN] [r1=GEN]
Each GEN is a llama-sparse-tail --gen file; configs are keyed as <tag>/<config>, scored on the prompts of da/dense.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from ruler_score import hit, post
from ruler_tokens import tokenizer

PAIRS = {"S": ("da/quest:27648", "da/mv:27648"), "E": ("da/mv:27648:d2", "da/mv:27648"),
         "S_d2": ("da/quest:27648:d2", "da/mv:27648:d2"), "E_quest": ("da/quest:27648:d2", "da/quest:27648"),
         "P": ("db/mv:27648", "r1/mv:27648")}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--meta", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("runs", nargs="+", help="tag=path")
    ap.add_argument("--out")
    a = ap.parse_args()
    tok = tokenizer(a.model)
    meta = json.loads(Path(a.meta).read_text())
    sc: dict[str, dict[int, tuple[str, float]]] = defaultdict(dict)
    for tag, path in (r.split("=", 1) for r in a.runs):
        for line in Path(path).read_text().splitlines():
            r = json.loads(line)
            m = meta[str(r["id"])]
            sc[f"{tag}/{r['config']}"][r["id"]] = (m["task"], hit(post(tok.decode(r["out"], skip_special_tokens=True)), m["outputs"], m["task"]))
    ids = sorted(sc["da/dense"])
    for k in sc:
        sc[k] = {i: v for i, v in sc[k].items() if i in set(ids)}
    tasks = sorted({sc["da/dense"][i][0] for i in ids})
    rng = np.random.default_rng(0)

    def tmean(k: str) -> float:
        return 100 * float(np.mean([np.mean([sc[k][i][1] for i in ids if sc[k][i][0] == t]) for t in tasks]))

    def eff(x: str, y: str) -> tuple[float, list[float]]:
        d = {t: np.array([sc[x][i][1] - sc[y][i][1] for i in ids if sc[x][i][0] == t]) for t in tasks}
        e = 100 * float(np.mean([v.mean() for v in d.values()]))
        bs = [100 * np.mean([v[rng.integers(0, len(v), len(v))].mean() for v in d.values()]) for _ in range(10000)]
        return round(e, 2), [round(float(np.percentile(bs, 2.5)), 2), round(float(np.percentile(bs, 97.5)), 2)]

    out: dict[str, object] = {"n_prompts": len(ids), "tasks": tasks, "mean": {k: round(tmean(k), 2) for k in sc}}
    out["delta_vs_dense"] = {k: eff(k, k.split("/")[0] + "/dense") for k in sc if not k.endswith("/dense")}
    out["effects"] = {n: eff(x, y) for n, (x, y) in PAIRS.items() if x in sc and y in sc}
    text = json.dumps(out, indent=1)
    if a.out:
        Path(a.out).write_text(text)
    print(text)


if __name__ == "__main__":
    main()
