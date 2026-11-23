"""PF-NT output kinds, gates F and D, and the row decoder/scorer, shared by pfn_decide.py and pfn_live.py (on the Mac).

kind() and EMPTY_BLOCK are src/pfb_output_kinds.py's, copied exactly (that module loads a tokenizer and the PF-BF16 data
at import); tests/pfn_tests.sh checks that both give the same kind on a set of edge cases. Kinds are taken on the output
decoded as the scorer decodes it (skip_special_tokens=True):
  empty       no generated token
  blank       tokens generated, but the decoded text is only whitespace
  answer      no "<think>" in the text
  emptythink  only whitespace-only "<think></think>" blocks, then the rest
  think_done  a "<think>" block with text inside, closed by "</think>"
  think_open  a "<think>" block never closed

Gate F (format), per format, over the 20 dense outputs of the first two units of the box that runs that format first
(b38: A's 1.11 and 1.8; q38: B's 2.11 and 2.8): PASS iff no output contains the string "<think>" (kinds emptythink,
think_done, think_open) and at most 2 are empty (no token) or blank (whitespace only). gate_f() on fewer than 20 rows is
PENDING unless the rows present already fail it (the live helper's view); pfn_decide.py treats a missing or invalid input
unit as F unavailable before calling it.
Gate D: the 10 dense rows of a task-0 unit score >= 80 (mean x 100; inclusive with EPS).

Scorer: port_r_score.py's hit() on post() of the decoded text, tokenizer Qwen/Qwen3.8-27B at the pinned revision
37561816e66b5b9153b1b64c10968c18c1e6f4a0, references and task from the
prompt meta (pf_nt_q38.i32.meta.json). With score_field (synthetic tests only) the score is read from that row field and
the decoded text from text_field.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

EXP = Path(__file__).resolve().parents[2]
SRC = EXP / "src"
META = EXP / "results" / "pf_nt" / "inputs" / "pf_nt_q38.i32.meta.json"
TOKENIZER = "Qwen/Qwen3.8-27B"
TOKENIZER_REV = "37561816e66b5b9153b1b64c10968c18c1e6f4a0"   # pinned (PF-NT, REV-029)
EPS = 1e-9

KINDS = ["answer", "emptythink", "think_done", "think_open", "blank", "empty"]
EMPTY_BLOCK = re.compile(r"<think>\s*</think>")
THINK_KINDS = ("emptythink", "think_done", "think_open")   # the decoded text contains "<think>"
NOOUT_KINDS = ("empty", "blank")
F_N, F_MAX_NOOUT = 20, 2
D_N, D_MIN = 10, 80.0


def kind(text: str, n_tokens: int) -> str:
    if n_tokens == 0:
        return "empty"
    if not text.strip():
        return "blank"
    rest = EMPTY_BLOCK.sub("", text)
    if "<think>" in rest:
        return "think_done" if "</think>" in rest.split("<think>", 1)[1] else "think_open"
    return "emptythink" if rest != text else "answer"


class Scorer:
    """decoded text and score of a row, cached per (model, id, config)"""

    def __init__(self, meta: str | Path = META, score_field: str | None = None, text_field: str | None = None) -> None:
        self.score_field, self.text_field = score_field, text_field
        self.cache: dict[tuple[str, int, str], tuple[str, float]] = {}
        if score_field:
            if not text_field:
                raise SystemExit("--text-field is required with --score-field (synthetic tests)")
            return
        self.meta = json.loads(Path(meta).read_text())
        sys.path.insert(0, str(SRC))
        from transformers import AutoTokenizer
        from port_r_score import hit, post
        self.hit, self.post = hit, post
        self.tok = AutoTokenizer.from_pretrained(TOKENIZER, revision=TOKENIZER_REV)

    def get(self, model: str, r: dict) -> tuple[str, float]:
        key = (model, r["id"], r["config"])
        if key not in self.cache:
            if self.score_field:
                self.cache[key] = (str(r.get(self.text_field, "")), float(r[self.score_field]))
            else:
                m = self.meta[str(r["id"])]
                text = self.tok.decode(r["out"], skip_special_tokens=True)
                self.cache[key] = (text, self.hit(self.post(text), m["outputs"], m["task"]))
        return self.cache[key]

    def text(self, model: str, r: dict) -> str:
        return self.get(model, r)[0]

    def score(self, model: str, r: dict) -> float:
        return self.get(model, r)[1]

    def kind(self, model: str, r: dict) -> str:
        return kind(self.text(model, r), len(r["out"]))


def gate_f(kinds: list[str]) -> dict[str, object]:
    """gate F over the dense kinds of one box's first two units"""
    n_think = sum(k in THINK_KINDS for k in kinds)
    n_noout = sum(k in NOOUT_KINDS for k in kinds)
    counts = {k: kinds.count(k) for k in KINDS}
    if n_think or n_noout > F_MAX_NOOUT:
        st = "FAIL"
    elif len(kinds) < F_N:
        st = "PENDING"
    elif len(kinds) > F_N:
        st = "INVALID"
    else:
        st = "PASS"
    why = (f"{len(kinds)} dense outputs: {n_think} with <think> text (must be 0), {n_noout} empty or blank "
           f"(at most {F_MAX_NOOUT})" + (f"; {F_N} needed" if st in ("PENDING", "INVALID") else ""))
    return {"status": st, "n": len(kinds), "n_think": n_think, "n_empty_or_blank": n_noout, "kinds": counts, "reason": why}


def gate_d(scores: list[float]) -> dict[str, object]:
    """gate D over the dense scores of one task-0 unit"""
    if len(scores) != D_N:
        return {"status": "PENDING" if len(scores) < D_N else "INVALID", "n": len(scores), "score": None,
                "reason": f"{len(scores)} dense rows, {D_N} needed"}
    s = 100 * sum(scores) / len(scores)
    ok = s >= D_MIN - EPS
    return {"status": "PASS" if ok else "FAIL", "n": len(scores), "score": s,
            "reason": f"task-0 dense {s:.1f} {'>=' if ok else '<'} {D_MIN:g}"}
