"""256k test: Δ(256k) − Δ(128k) on the same six RULER tasks at a fixed budget, paired bootstrap."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from transformers import PreTrainedTokenizerBase

from ruler_score import hit, post
from ruler_tokens import tokenizer


def paired(gen: Path, meta: Path, tok: PreTrainedTokenizerBase, cfg: str) -> dict[str, np.ndarray]:
    m = json.loads(meta.read_text())
    sc: dict[str, dict[str, dict[int, float]]] = {}
    for line in gen.read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        if r["config"] not in ("dense", cfg):
            continue
        t = m[str(r["id"])]["task"]
        sc.setdefault(r["config"], {}).setdefault(t, {})[r["id"]] = hit(post(tok.decode(r["out"], skip_special_tokens=True)), m[str(r["id"])]["outputs"], t)
    return {t: np.array([sc[cfg][t][i] - sc["dense"][t][i] for i in sorted(sc["dense"][t])]) for t in sorted(sc["dense"])}


def delta(d: dict[str, np.ndarray], rng: np.random.Generator | None = None) -> float:
    if rng is None:
        return 100 * float(np.mean([v.mean() for v in d.values()]))
    return 100 * float(np.mean([v[rng.integers(0, len(v), len(v))].mean() for v in d.values()]))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True, help="results/cuda_4090_48gb_portl/ruler")
    ap.add_argument("--model", required=True)
    ap.add_argument("--config", default="quest:49152")
    ap.add_argument("--boot", type=int, default=10000)
    ap.add_argument("--out")
    a = ap.parse_args()
    d = Path(a.dir)
    tok = tokenizer(a.model)
    d128 = paired(d / "gen_131072.jsonl", d / "ruler131072_ul.i32.meta.json", tok, a.config)
    d256 = paired(d / "gen_262144.jsonl", d / "ruler262144_ul.i32.meta.json", tok, a.config)
    rng = np.random.default_rng(0)
    boots = [delta(d256, rng) - delta(d128, rng) for _ in range(a.boot)]
    diff = delta(d256) - delta(d128)
    lo, hi = float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))
    verdict = "supported" if diff >= -3 else ("refuted" if hi < 0 else "inconclusive")
    text = json.dumps({"delta_128k": delta(d128), "delta_256k": delta(d256), "difference": diff, "ci95": [lo, hi], "verdict": verdict}, indent=1)
    if a.out:
        Path(a.out).write_text(text)
    print(text)


if __name__ == "__main__":
    main()
