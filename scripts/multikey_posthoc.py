"""Post-hoc reading of RULER failures (not pre-registered): answer token spans relative to 64-cell pages, and
dense vs sparse answers on the prompts where the scores differ."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from ruler_score import hit, post
from ruler_tokens import read_prompts, tokenizer

PAGE = 64


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompts", required=True, help="the .i32 prompt file given to llama-sparse-tail --gen")
    ap.add_argument("--meta", required=True)
    ap.add_argument("--gen", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--config", required=True, help="sparse config to compare with dense, e.g. quest:28672")
    ap.add_argument("--tasks", default="niah_multikey_2,niah_multivalue,niah_multiquery")
    a = ap.parse_args()
    tok = tokenizer(a.model)
    meta = json.loads(Path(a.meta).read_text())
    prompts = read_prompts(Path(a.prompts))
    outs: dict[int, dict[str, str]] = {}
    for line in Path(a.gen).read_text().splitlines():
        r = json.loads(line)
        outs.setdefault(r["id"], {})[r["config"]] = post(tok.decode(r["out"], skip_special_tokens=True))
    for i, c in sorted(outs.items()):
        m = meta[str(i)]
        if m["task"] not in a.tasks.split(","):
            continue
        ids = prompts[i]
        spans = []
        for ref in m["outputs"]:
            rt = tok(" " + ref, add_special_tokens=False)["input_ids"]
            # the leading space can be its own token; locate the full match, report only the answer tokens
            lead = 1 if tok.decode(rt[:1]).strip() == "" else 0
            n = len(rt) - lead
            spans += [f"{p + lead}+{n}{' edge' if (p + lead) % PAGE + n > PAGE else ''}"
                      for p in range(len(ids) - len(rt) + 1) if ids[p:p + len(rt)] == rt]
        d, q = hit(c["dense"], m["outputs"], m["task"]), hit(c[a.config], m["outputs"], m["task"])
        print(f"{m['task']} {i} dense={d:.2f} quest={q:.2f} refs={m['outputs']} spans={spans}")
        if d != q:
            print(f"  dense: {c['dense'][:100]!r}\n  quest: {c[a.config][:100]!r}")


if __name__ == "__main__":
    main()
