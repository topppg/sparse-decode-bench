"""PF-NT checks on the box and on the Mac (no tokenizer needed): PF-BF16's pfb_check.py for two models, BF16 (b38) and
UD-Q4_K_M (q38) Qwen3.8-27B, without the speed blocks and the reference identity (PF-NT has neither). Reuses PF-EMUL's
pfe_check (copied unchanged).

  pfn_check.py subset <prompts.i32> --task T --pass P        exactly ids T*1000 + (0..9 | 10..19), no trailing bytes
  pfn_check.py task <rows.jsonl> --model M --task T --pass P --configs C --err <stderr file>
        one unit before it is committed: every row an object with id (int), config (str), len (int), out (list of int),
        pf (object) and prefill_compute_s (number), checked first (a malformed row is invalid, never an exception);
        then pfe_check's exact (id, config) coverage, unique rows, integer outputs (an empty
        list is valid: the model's immediate end-of-sequence, scored 0 later), sparse prefill counts as expected, len >=
        100000, sparse rows engaged on all 16 layers, no reused prefill graph, prefill_compute_s > 0 on the dense and pf
        rows. The per-prompt pf:0.1:512:d512 / dense prefill_compute_s ratio is informative only: its median, max and
        the count above G_REF (0.85) are reported, never invalid.
        Prints one JSON line, then "TASK OK|INCOMPLETE ...|INVALID ..." with the median gather/dense prefill ratio.
        exit 0 valid, 1 coverage problem (crash, missing rows), 2 invalid (counts, reuse, duplicates, timings)
  pfn_check.py cohort <ruler dir> --model M --configs C [--units "1.0 1.1 ..."] [--out pairs.jsonl]
        one model's units on this box (default all 26: both passes, 13 tasks): <M>_p<P>_t<T>.jsonl hold exactly 10 ids
        per unit x configs, counts valid, timings > 0 (ratios informative); exit 0 only if complete
  pfn_check.py nsys <gather kern_sum.csv> <emulation kern_sum.csv>
        gate U: both tables hold at least one kernel; the total time of kernels whose name contains
        flash_attn_mask_to_sparse_indices is > 0 in the gather and 0 in the emulation. Prints
        "gather <s> s, emulation 0 s" (exit 0) or the reason (exit 1)
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import statistics
import struct
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pfe_check import check_rows, read_rows  # noqa: E402

# Qwen3.8-27B: 16 of 64 layers full attention, 4 KV heads; the BF16 and the UD-Q4_K_M file share the architecture
MODELS = {"b38": {"layers": 16, "kv_heads": 4}, "q38": {"layers": 16, "kv_heads": 4}}
# informative reference only (PF-BF16, REV-026): pf / dense prefill_compute_s above it is counted
G_REF = {"b38": 0.85, "q38": 0.85}
PF = "pf:0.1:512:d512"
INDEX_KERNEL = "flash_attn_mask_to_sparse_indices"          # gate U: launched only by the sparse gather


def pass_ids(task: int, p: int) -> set[int]:
    lo = 0 if p == 1 else 10
    return {task * 1000 + i for i in range(lo, lo + 10)}


def prefill_ratios(rows: list[dict]) -> tuple[list[str], list[float]]:
    """(problems, ratios) of pf:0.1:512:d512 / dense prefill_compute_s over the prompts that have both rows; a problem is
    a non-positive or missing timing only (the ratio itself is informative)"""
    t = {(r.get("id"), r.get("config")): r.get("prefill_compute_s") for r in rows}
    probs, ratios = [], []
    for pid in sorted({i for i, c in t if c == "dense" and (i, PF) in t}, key=str):
        d, p = t[(pid, "dense")], t[(pid, PF)]
        if not (isinstance(d, (int, float)) and isinstance(p, (int, float)) and d > 0 and p > 0):
            probs.append(f"id {pid} prefill_compute_s dense {d} pf {p} (both must be > 0)")
            continue
        ratios.append(p / d)
    return probs, ratios


def med(ratios: list[float]) -> str:
    return f"{statistics.median(ratios):.3f}" if ratios else "n/a"


def ratio_info(ratios: list[float], model: str) -> dict[str, object]:
    return {"gather_dense_median": med(ratios), "gather_dense_max": f"{max(ratios):.3f}" if ratios else "n/a",
            "g_ref": G_REF[model], "n_above_g_ref": sum(x > G_REF[model] for x in ratios)}


def is_int(x: object) -> bool:
    return isinstance(x, int) and not isinstance(x, bool)


def is_num(x: object) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool)


def schema_problem(r: object) -> str | None:
    """why a row cannot be checked at all (wrong type or missing field), or None; checked before pfe_check's checker"""
    if not isinstance(r, dict):
        return f"row is a {type(r).__name__}, not an object"
    key = (r.get("id"), r.get("config"))
    for f, ok in (("id", is_int), ("config", lambda x: isinstance(x, str)), ("len", is_int),
                  ("out", lambda x: isinstance(x, list) and all(is_int(t) for t in x)), ("pf", lambda x: isinstance(x, dict)),
                  ("prefill_compute_s", is_num)):
        if f not in r:
            return f"{key}: field {f} missing"
        if not ok(r[f]):
            return f"{key}: field {f} is {r[f]!r:.40}"
    return None


def unit_problems(rows: list, ids: set[int], cfgs: list[str], model: str) -> tuple[list[str], list[str], list[float]]:
    """(coverage problems, invalid problems, informative pf/dense ratios); malformed rows are invalid and left out of the
    other checks, so no row content can raise"""
    bad = [p for p in map(schema_problem, rows) if p]
    rows = [r for r in rows if schema_problem(r) is None]
    cov, inv = check_rows(rows, ids, cfgs, MODELS[model])
    inv = bad[:20] + inv
    # REV-024: counts are derived from the reported length, so require the registered lengths and real sparse engagement
    for r in rows:
        if r["len"] < 100000:
            inv.append(f"({r['id']}, {r['config']}): len {r['len']} < 100000")
        elif r["config"].startswith("pf:") and not (is_num(r["pf"].get("sparse")) and r["pf"]["sparse"] > 0
                                                    and r["pf"].get("layers") == MODELS[model]["layers"]):
            inv.append(f"({r.get('id')}, {r.get('config')}): sparse prefill did not run: {r.get('pf')}")
    g, ratios = prefill_ratios(rows) if PF in cfgs else ([], [])
    return cov, inv + g[:20], ratios


ALL_UNITS = [(p, t) for p in (1, 2) for t in range(13)]


def parse_units(spec: str | None) -> list[tuple[int, int]]:
    """'1.0 1.1 2.12' -> [(1, 0), (1, 1), (2, 12)]; None or empty = all 26 in pass/task order."""
    if not spec or not spec.strip():
        return list(ALL_UNITS)
    out = []
    for u in spec.split():
        m = re.fullmatch(r"([12])\.(\d+)", u)
        if not m or int(m[2]) > 12 or (int(m[1]), int(m[2])) in out:
            raise SystemExit(f"bad or duplicated unit {u!r} in {spec!r}")
        out.append((int(m[1]), int(m[2])))
    return out


def cohort_rows(dirs: list[Path], model: str, units: list[tuple[int, int]]) -> tuple[list[dict], list[str]]:
    """rows of the given units over one or more ruler dirs; a unit must be committed in exactly one dir"""
    rows, probs = [], []
    for p, t in units:
        fs = [d / f"{model}_p{p}_t{t}.jsonl" for d in dirs if (d / f"{model}_p{p}_t{t}.jsonl").exists()]
        if not fs:
            probs.append(f"pass {p} task {t}: no committed file")
            continue
        if len(fs) > 1:
            probs.append(f"pass {p} task {t}: committed in {len(fs)} dirs ({', '.join(str(f.parent) for f in fs)})")
        for f in fs:
            r, bad = read_rows(f)
            probs += [f"pass {p} task {t}: {b}" for b in bad]
            rows += r
    return rows, probs


def cohort(dirs: list[Path], model: str, cfgs: list[str], units: list[tuple[int, int]] | None = None
           ) -> tuple[list[dict], list[str], list[float]]:
    units = units or list(ALL_UNITS)
    rows, probs = cohort_rows(dirs, model, units)
    ids = {i for p, t in units for i in pass_ids(t, p)}
    cov, inv, ratios = unit_problems(rows, ids, cfgs, model)
    return rows, probs + cov + inv, ratios


def cmd_subset(a: argparse.Namespace) -> int:
    raw = Path(a.path).read_bytes()
    (n,) = struct.unpack_from("<i", raw, 0)
    off, ids = 4, []
    try:
        for _ in range(n):
            pid, _mx, ln = struct.unpack_from("<3i", raw, off)
            ids.append(pid)
            off += 12 + 4 * ln
    except struct.error:
        off = -1
    ok = off == len(raw) and sorted(ids) == sorted(pass_ids(a.task, a.pass_))
    print(f"subset {a.path}: {n} prompts, ids {ids[:3]}..., {'OK' if ok else 'WRONG'}")
    return 0 if ok else 1


def cmd_task(a: argparse.Namespace) -> int:
    cfgs = a.configs.split(",")
    rows, bad = read_rows(Path(a.rows))
    cov, inv, ratios = unit_problems(rows, pass_ids(a.task, a.pass_), cfgs, a.model)
    inv = bad + inv
    if a.err and Path(a.err).exists():
        reused = [l for l in Path(a.err).read_text().splitlines() if re.match(r"^prefill .*graphs reused", l)]
        if reused:
            inv.append(f"{len(reused)} prefills reused a graph: {reused[0]}")
    info = ratio_info(ratios, a.model)
    print(json.dumps({"rows": len(rows), "coverage": cov, "invalid": inv} | info))
    print("TASK " + ("INVALID " + "; ".join(inv) if inv else "INCOMPLETE " + "; ".join(cov) if cov else "OK")
          + f"; above {G_REF[a.model]}: {info['n_above_g_ref']} (informative); gather/dense prefill median {med(ratios)}")
    return 2 if inv else 1 if cov else 0


def cmd_cohort(a: argparse.Namespace) -> int:
    cfgs = a.configs.split(",")
    units = parse_units(a.units)
    rows, probs, ratios = cohort([Path(a.dir)], a.model, cfgs, units)
    status = "COMPLETE" if not probs else "INCOMPLETE"
    if a.out:
        with open(a.out, "w") as fo:
            for r in rows:
                fo.write(json.dumps(r) + "\n")
    print(json.dumps({"model": a.model, "status": status, "n_units": len(units), "n_rows": len(rows),
                      "n_expected": 10 * len(units) * len(cfgs)} | ratio_info(ratios, a.model) | {"problems": probs[:20]}))
    return 0 if status == "COMPLETE" else 1


def kernel_s(path: Path) -> tuple[int, float]:
    """(kernel rows, total seconds of the sparse index kernel) of one nsys cuda_gpu_kern_sum CSV"""
    n, idx = 0, 0.0
    with open(path) as f:
        for row in csv.DictReader(f):
            n += 1
            if INDEX_KERNEL in row["Name"]:
                idx += float(row["Total Time (ns)"]) / 1e9
    return n, idx


def cmd_nsys(a: argparse.Namespace) -> int:
    got = {}
    for tag, f in (("gather", Path(a.gather)), ("emulation", Path(a.emul))):
        if not f.is_file():
            print(f"{tag}: no kernel table {f}")
            return 1
        n, idx = kernel_s(f)
        if n == 0:
            print(f"{tag}: the kernel table is empty (nsys recorded no kernels)")
            return 1
        got[tag] = idx
    line = f"gather {got['gather']:.6f} s, emulation {got['emulation']:.6f} s"
    if not (got["gather"] > 0 and got["emulation"] == 0):
        print(f"{INDEX_KERNEL} {line}: must be > 0 in the gather and 0 in the emulation")
        return 1
    print(f"gather {got['gather']:.6f} s, emulation 0 s")
    return 0


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("subset"); s.add_argument("path")
    s.add_argument("--task", type=int, required=True); s.add_argument("--pass", dest="pass_", type=int, choices=(1, 2), required=True)
    s = sub.add_parser("task"); s.add_argument("rows"); s.add_argument("--model", choices=MODELS, required=True)
    s.add_argument("--task", type=int, required=True); s.add_argument("--pass", dest="pass_", type=int, choices=(1, 2), required=True)
    s.add_argument("--configs", required=True); s.add_argument("--err")
    s = sub.add_parser("cohort"); s.add_argument("dir"); s.add_argument("--model", choices=MODELS, required=True)
    s.add_argument("--configs", required=True); s.add_argument("--out"); s.add_argument("--units", help="'1.0 1.1 ...', default all 26")
    s = sub.add_parser("nsys"); s.add_argument("gather"); s.add_argument("emul")
    a = ap.parse_args()
    sys.exit({"subset": cmd_subset, "task": cmd_task, "cohort": cmd_cohort, "nsys": cmd_nsys}[a.cmd](a))


if __name__ == "__main__":
    main()
