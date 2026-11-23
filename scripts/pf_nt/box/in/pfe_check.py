"""PF-EMUL checks on the box and on the Mac (no tokenizer needed).

  pfe_check.py subset <prompts.i32> --task T --n 10          the subset holds exactly ids T*1000 + 0..n-1
  pfe_check.py gate-a <gen.jsonl> <tf.jsonl> --model M      gate A verdict (A1 / A2 / FAIL), counts of the pf:1 row checked
  pfe_check.py task <rows.jsonl> --model M --task T --configs C --err <stderr file>
        one task's artifact before it is committed: exact (id, config) coverage, unique rows, integer outputs, the sparse
        prefill counts of every row as expected for its prompt length and config, no reused prefill graph.
        exit 0 valid, 1 coverage problem (crash, missing rows), 2 invalid (counts, reuse, duplicates)
  pfe_check.py cohort <ruler dir> --model M --configs C [--tag T] [--tasks 0-12] [--out pairs.jsonl]
        exact coverage of the committed per-task files <M><T>_t<task>.jsonl: every (id, config) exactly once, counts valid;
        exit 0 only if complete

Expected counts (llama_sparse_prefill_counts, summed over the full-attention layers of one prefill of n = len - 1 tokens in
ubatches of 512 from position 0; W from the config): per ubatch with c0 = 512 i and nt rows, in this order (as
build_sparse_prefill): rows restricted and none sparse -> dense_rows += nt; no block with a selectable page -> no_sel += 1;
else sparse += n_kv_h, sparse_rows += r1 - r0, dense_rows += nt - (r1 - r0). fallback must be 0, layers = the model's
full-attention layers, every such layer has the same number of sparse ubatches.
"""
from __future__ import annotations

import argparse
import json
import math
import re
import struct
import sys
from collections import Counter
from pathlib import Path

MODELS = {"q36": {"layers": 10, "kv_heads": 2}, "goss": {"layers": 12, "kv_heads": 8}}
UB = 512
KL_TOL, DLOGIT_TOL, A2_MIN = 1e-4, 0.25, 64   # provisional gate A tolerances (no CUDA measurement yet); values are recorded
REF_KEYS = ("kl_mean", "kl_p99", "kl_max", "flip", "max_abs_logit_diff", "kl_rows")   # A3 (amendment 1): exact equality


def parse_pf(cfg: str) -> tuple[float, int, str] | None:
    """(F, W, rows) of a config's prefill part, None for a dense prefill."""
    for part in cfg.split("+"):
        p = part.split(":")
        if p[0] == "pf":
            f, w, r = float(p[1]), 512, ""
            for x in p[2:]:
                if re.fullmatch(r"\d+", x):
                    w = int(x)
                else:
                    r = x
            return f, w, r
    return None


def rows_range(r: str, n: int) -> tuple[int, int] | None:
    """LLAMA_SPARSE_PREFILL_ROWS as sparse-tail sets it (prefill_rows), None = all rows."""
    if not r:
        return None
    if r == "du":
        return 0, (n - 1) // UB * UB
    m = max(0, n - int(r[1:]))
    return (0, m) if r[0] == "d" else (m, 1 << 62)


def expected_counts(cfg: str, n: int, layers: int, kv_heads: int) -> dict[str, int]:
    pf = parse_pf(cfg)
    z = {"sparse": 0, "sparse_rows": 0, "dense_rows": 0, "no_sel": 0, "fallback": 0, "layers": 0, "ub_min": 0, "ub_max": 0}
    if pf is None:
        return z
    _, w, r = pf
    rr = rows_range(r, n)
    ub_sparse = 0
    for c0 in range(0, n, UB):
        nt = min(UB, n - c0)
        r0, r1 = 0, nt
        if rr is not None:
            r0 = min(max(rr[0] - c0, 0), nt)
            r1 = min(max(rr[1] - c0, 0), nt)
            if r0 >= r1:
                z["dense_rows"] += nt
                continue
        if not any(c0 + 64 * b - w >= 128 for b in range((nt + 63) // 64)):
            z["no_sel"] += 1
            continue
        ub_sparse += 1
        z["sparse"] += kv_heads
        z["sparse_rows"] += r1 - r0
        z["dense_rows"] += nt - (r1 - r0)
    for k in ("sparse", "sparse_rows", "dense_rows", "no_sel"):
        z[k] *= layers
    if ub_sparse:
        z.update(layers=layers, ub_min=ub_sparse, ub_max=ub_sparse)
    return z


def model_of(a: argparse.Namespace) -> dict[str, int]:
    m = dict(MODELS.get(a.model, {}))
    if a.layers is not None:
        m["layers"] = a.layers
    if a.kv_heads is not None:
        m["kv_heads"] = a.kv_heads
    if "layers" not in m:
        raise SystemExit(f"unknown model {a.model}")
    return m


def read_rows(path: Path) -> tuple[list[dict], list[str]]:
    rows, bad = [], []
    for i, line in enumerate(path.read_text().splitlines()):
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            bad.append(f"line {i + 1} is not JSON")
    return rows, bad


def check_rows(rows: list[dict], ids: set[int], cfgs: list[str], m: dict[str, int]) -> tuple[list[str], list[str]]:
    """(coverage problems, invalid problems) of a set of rows that must hold every (id, config) of ids x cfgs once."""
    cov, inv = [], []
    cnt = Counter((r.get("id"), r.get("config")) for r in rows)
    exp = {(i, c) for i in ids for c in cfgs}
    dups = [k for k, v in cnt.items() if v > 1]
    if dups:
        inv.append(f"{len(dups)} duplicate (id, config) rows, e.g. {dups[0]}")
    extra = [k for k in cnt if k not in exp]
    if extra:
        inv.append(f"{len(extra)} unexpected (id, config) rows, e.g. {extra[0]}")
    miss = sorted(exp - set(cnt))
    if miss:
        cov.append(f"{len(miss)} of {len(exp)} (id, config) rows missing, e.g. {miss[0]}")
    for r in rows:
        key = (r.get("id"), r.get("config"))
        if key not in exp:
            continue
        if not isinstance(r.get("out"), list) or not all(isinstance(x, int) for x in r["out"]):
            inv.append(f"{key}: output is not a token list")
            continue
        want = expected_counts(r["config"], int(r["len"]) - 1, m["layers"], m["kv_heads"])
        if r.get("pf") != want:
            inv.append(f"{key}: sparse prefill counts {r.get('pf')} != expected {want}")
    return cov, inv[:20]


def cmd_subset(a: argparse.Namespace) -> int:
    raw = Path(a.path).read_bytes()
    (n,) = struct.unpack_from("<i", raw, 0)
    off, ids = 4, []
    for _ in range(n):
        pid, _mx, ln = struct.unpack_from("<3i", raw, off)
        ids.append(pid)
        off += 12 + 4 * ln
    ok = off == len(raw) and sorted(ids) == [a.task * 1000 + i for i in range(a.n)]
    print(f"subset {a.path}: {n} prompts, ids {ids[:3]}..., {'OK' if ok else 'WRONG'}")
    return 0 if ok else 1


def cmd_gate_a(a: argparse.Namespace) -> int:
    m = model_of(a)
    rd = lambda f: {x["config"]: x for x in read_rows(Path(f))[0]}
    g, t = rd(a.gen), rd(a.tf)
    if set(g) != {"dense", "pf:1:512"} or set(t) != {"dense", "pf:1:512"}:
        print(f"FAIL configs {sorted(g)} / {sorted(t)}"); return 1
    d, p, tf = g["dense"], g["pf:1:512"], t["pf:1:512"]
    probs = []
    for row, n in ((p, int(p["len"]) - 1), (tf, int(tf["T"]) - a.n_tail)):
        want = expected_counts("pf:1:512", n, m["layers"], m["kv_heads"])
        if row.get("pf") != want or want["sparse"] == 0:
            probs.append(f"pf:1 counts {row.get('pf')} != {want}")
    kl, mx = float(tf["kl_mean"]), float(tf["max_abs_logit_diff"])
    num_ok = math.isfinite(kl) and math.isfinite(mx) and kl <= KL_TOL and mx <= DLOGIT_TOL
    if a.ref_tf:
        # amendment 1 (A3): the teacher-forced pf:1 statistics equal the diagnostic's exactly (same box, tree and prompt; the
        # diagnostic showed them equal to a per-KV-head split with the plain causal mask, and dense deterministic)
        # REV-018: every reference field is required (64 per-step KLs), and the greedy outputs of the generation run must
        # equal the first run's (--ref-gen) for both configs, so a failure confined to generation is caught too
        # the greedy reference sits next to the teacher-forced one (ref_tf_<m>.jsonl -> ref_gen_<m>.jsonl) unless given
        a.ref_gen = a.ref_gen or str(Path(a.ref_tf).with_name(Path(a.ref_tf).name.replace("ref_tf_", "ref_gen_")))
        r, rg = rd(a.ref_tf).get("pf:1:512", {}), rd(a.ref_gen) if Path(a.ref_gen).is_file() else {}
        num_ok = (math.isfinite(kl) and math.isfinite(mx) and all(k in r for k in REF_KEYS)
                  and len(r["kl_rows"]) == a.n_tail and all(tf.get(k) == r[k] for k in REF_KEYS)
                  and set(rg) == {"dense", "pf:1:512"} and all(len(g[c]["out"]) > 0 and g[c]["out"] == rg[c]["out"] for c in rg))
    x, y = d["out"], p["out"]
    first = None if x == y else next((i for i, (u, v) in enumerate(zip(x, y)) if u != v), min(len(x), len(y)))
    meas = (f"teacher-forced kl_mean {kl:.3g} (tol {KL_TOL:g}) max|dlogit| {mx:.3g} (tol {DLOGIT_TOL:g}); "
            f"prefill dense {d['prefill_s']:.1f} s, pf:1 {p['prefill_s']:.1f} s, ratio {p['prefill_s'] / d['prefill_s']:.2f}")
    if probs:
        v = "FAIL (" + "; ".join(probs) + ")"
    elif a.ref_tf:
        v = (f"PASS A3 (teacher-forced pf:1 and both greedy outputs identical to the references {Path(a.ref_tf).name}, {Path(a.ref_gen).name}; first differing greedy token {first})"
             if num_ok else f"FAIL (A3: teacher-forced pf:1 or a greedy output differs from the references, or a reference is incomplete)")
    elif not num_ok:
        v = f"FAIL (numerical check not met; first differing token {first})"
    elif first is None:
        v = "PASS A1 (identical tokens, numerical check met)"
    elif first >= min(A2_MIN, len(x)):
        v = f"PASS A2 (first differing token {first} of {len(x)}, numerical check met)"
    else:
        v = f"FAIL (first differing token {first} of {len(x)} < {min(A2_MIN, len(x))})"
    print(f"{v}; {meas}")
    return 0 if v.startswith("PASS") else 1


def cmd_task(a: argparse.Namespace) -> int:
    m = model_of(a)
    cfgs = a.configs.split(",")
    rows, bad = read_rows(Path(a.rows))
    cov, inv = check_rows(rows, {a.task * 1000 + i for i in range(a.n)}, cfgs, m)
    inv = bad + inv
    if a.err and Path(a.err).exists():
        reused = [l for l in Path(a.err).read_text().splitlines() if re.match(r"^prefill .*graphs reused", l)]
        if reused:
            inv.append(f"{len(reused)} prefills reused a graph: {reused[0]}")
    print(json.dumps({"rows": len(rows), "coverage": cov, "invalid": inv}))
    print("TASK " + ("INVALID " + "; ".join(inv) if inv else "INCOMPLETE " + "; ".join(cov) if cov else "OK"))
    return 2 if inv else 1 if cov else 0


def cmd_cohort(a: argparse.Namespace) -> int:
    m = model_of(a)
    cfgs = a.configs.split(",")
    lo, hi = (int(x) for x in a.tasks.split("-"))
    rows, probs = [], []
    for t in range(lo, hi + 1):
        f = Path(a.dir) / f"{a.model}{a.tag}_t{t}.jsonl"
        if not f.exists():
            probs.append(f"task {t}: no committed file")
            continue
        r, bad = read_rows(f)
        probs += [f"task {t}: {b}" for b in bad]
        rows += r
    ids = {t * 1000 + i for t in range(lo, hi + 1) for i in range(a.n)}
    cov, inv = check_rows(rows, ids, cfgs, m)
    probs += cov + inv
    status = "COMPLETE" if not probs else "INCOMPLETE"
    if a.out:
        with open(a.out, "w") as fo:
            for r in rows:
                fo.write(json.dumps(r) + "\n")
    print(json.dumps({"model": a.model, "tag": a.tag, "status": status, "n_rows": len(rows), "n_expected": len(ids) * len(cfgs),
                      "problems": probs[:20]}))
    return 0 if status == "COMPLETE" else 1


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("subset"); s.add_argument("path"); s.add_argument("--task", type=int, required=True); s.add_argument("--n", type=int, default=10)
    for nm in ("gate-a", "task", "cohort"):
        p = sub.add_parser(nm)
        p.add_argument("--model", required=True); p.add_argument("--layers", type=int); p.add_argument("--kv-heads", type=int)
        if nm == "gate-a":
            p.add_argument("gen"); p.add_argument("tf"); p.add_argument("--n-tail", type=int, default=64)
            p.add_argument("--ref-tf"); p.add_argument("--ref-gen")
        if nm == "task":
            p.add_argument("rows"); p.add_argument("--task", type=int, required=True); p.add_argument("--configs", required=True)
            p.add_argument("--err"); p.add_argument("--n", type=int, default=10)
        if nm == "cohort":
            p.add_argument("dir"); p.add_argument("--configs", required=True); p.add_argument("--tag", default="")
            p.add_argument("--tasks", default="0-12"); p.add_argument("--n", type=int, default=10); p.add_argument("--out")
    a = ap.parse_args()
    sys.exit({"subset": cmd_subset, "gate-a": cmd_gate_a, "task": cmd_task, "cohort": cmd_cohort}[a.cmd](a))


if __name__ == "__main__":
    main()
