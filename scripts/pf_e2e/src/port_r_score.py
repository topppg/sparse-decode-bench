"""RULER scoring functions for the PF-NT / PF-E2E decision scripts (pfn_gates.Scorer imports them as port_r_score).

post() and hit() are copied exactly from the project repository's experiments/src/port_r_score.py (commit 44a9e78), whose
command-line part and imports of other project modules are left out. They are the same functions as scripts/ruler_score.py's.
"""
from __future__ import annotations

import re

NP = re.compile(r"[\x00-\x1f]")


def post(s: str) -> str:
    return NP.sub("\n", s.strip()).strip()


def hit(pred: str, refs: list[str], task: str) -> float:
    m = [1.0 if r.lower() in pred.lower() else 0.0 for r in refs]
    return max(m) if task.startswith("qa") else sum(m) / len(m)
