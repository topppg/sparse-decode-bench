"""Score llama-sparse-tail --gen output on RULER; Δ vs dense with a paired bootstrap over prompts."""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
from transformers import AutoTokenizer

from ruler_tokens import TASKS

CTRL = re.compile(r"[\x00-\x1f]")


def post(s: str) -> str:
    return CTRL.sub("\n", s.strip()).strip()


def hit(pred: str, refs: list[str], task: str) -> float:
    # RULER metric: qa counts any reference, the other tasks the fraction of references found
    m = [1.0 if r.lower() in pred.lower() else 0.0 for r in refs]
    return max(m) if task.startswith("qa") else sum(m) / len(m)


def per_prompt(gen: Path, meta: dict[str, dict[str, object]], tok: AutoTokenizer) -> dict[str, dict[str, dict[int, float]]]:
    sc: dict[str, dict[str, dict[int, float]]] = {}
    for line in gen.read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        m = meta[str(r["id"])]
        s = hit(post(tok.decode(r["out"], skip_special_tokens=True)), m["outputs"], m["task"])
        sc.setdefault(r["config"], {}).setdefault(m["task"], {})[r["id"]] = s
    return sc


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gen", required=True)
    ap.add_argument("--meta", required=True)
    ap.add_argument("--model", required=True, help="tokenizer used to decode the outputs")
    ap.add_argument("--out")
    ap.add_argument("--boot", type=int, default=10000)
    a = ap.parse_args()
    tok = AutoTokenizer.from_pretrained(a.model)
    sc = per_prompt(Path(a.gen), json.loads(Path(a.meta).read_text()), tok)
    cfgs = sorted(sc, key=lambda c: c != "dense")
    tasks = [t for t in TASKS if t in sc["dense"]]
    out: dict[str, object] = {"per_task": {c: {t: 100 * float(np.mean(list(sc[c][t].values()))) for t in tasks if t in sc[c]} for c in cfgs}}
    out["mean"] = {c: float(np.mean(list(out["per_task"][c].values()))) for c in cfgs}
    rng = np.random.default_rng(0)
    for c in cfgs[1:]:
        diff = {t: np.array([sc[c][t][i] - sc["dense"][t][i] for i in sorted(set(sc["dense"][t]) & set(sc[c][t]))]) for t in tasks}
        diff = {t: d for t, d in diff.items() if len(d)}
        delta = 100 * float(np.mean([d.mean() for d in diff.values()]))
        boots = [100 * np.mean([d[rng.integers(0, len(d), len(d))].mean() for d in diff.values()]) for _ in range(a.boot)]
        out[f"delta_{c}"] = {"delta": delta, "ci95": [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))],
                             "n_pairs": int(sum(len(d) for d in diff.values()))}
    text = json.dumps(out, indent=1)
    if a.out:
        Path(a.out).write_text(text)
    print(text)


if __name__ == "__main__":
    main()
