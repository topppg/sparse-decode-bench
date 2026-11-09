"""Tokenise prepared RULER data into the binary prompt file read by llama-sparse-tail --gen."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from transformers import AutoTokenizer, PreTrainedTokenizerBase

TASKS = ("niah_single_1", "niah_single_2", "niah_single_3", "niah_multikey_1", "niah_multikey_2", "niah_multikey_3",
         "niah_multivalue", "niah_multiquery", "vt", "cwe", "fwe", "qa_1", "qa_2")
GEN = {"niah": 128, "vt": 30, "cwe": 120, "fwe": 50, "qa": 32}


def tokenizer(spec: str) -> PreTrainedTokenizerBase:
    """Load a Hugging Face tokenizer from 'repo' or 'repo@revision'."""
    repo, _, rev = spec.partition("@")
    return AutoTokenizer.from_pretrained(repo, revision=rev or None)


def read_prompts(path: Path) -> dict[int, list[int]]:
    # layout: int32 n, then per prompt int32 [id, max_new, len, tokens...]
    a = np.fromfile(path, np.int32)
    out: dict[int, list[int]] = {}
    o = 1
    for _ in range(int(a[0])):
        pid, _, n = (int(x) for x in a[o:o + 3])
        out[pid] = a[o + 3:o + 3 + n].tolist()
        o += 3 + n
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="RULER save_dir with <task>/validation.jsonl")
    ap.add_argument("--model", required=True, help="tokenizer: repo or repo@revision")
    ap.add_argument("--out", required=True)
    ap.add_argument("--tasks", default=",".join(TASKS))
    a = ap.parse_args()
    tok = tokenizer(a.model)
    recs: list[np.ndarray] = []
    meta: dict[str, dict[str, object]] = {}
    for ti, task in enumerate(TASKS):
        if task not in a.tasks.split(","):
            continue
        for line in (Path(a.data) / task / "validation.jsonl").read_text().splitlines():
            s = json.loads(line)
            pid = 1000 * ti + int(s["index"])
            ids = np.array(tok(s["input"] + s.get("answer_prefix", ""))["input_ids"], np.int32)
            recs.append(np.concatenate([np.array([pid, GEN[task.split("_")[0]], len(ids)], np.int32), ids]))
            meta[str(pid)] = {"task": task, "outputs": s["outputs"], "len": int(len(ids))}
    with Path(a.out).open("wb") as f:
        np.array([len(recs)], np.int32).tofile(f)
        for r in recs:
            r.tofile(f)
    Path(a.out + ".meta.json").write_text(json.dumps(meta))
    print(a.out, len(recs), max(int(r[2]) + int(r[1]) for r in recs))


if __name__ == "__main__":
    main()
