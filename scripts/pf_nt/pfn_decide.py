"""PF-NT decision (on the Mac, after the pull): gate F recomputed from the rows, the quality decision table for each model
(b38 BF16, q38 UD-Q4_K_M) over the run dirs named by the manifest, and the outcome.

  cd experiments/src
  ../.venv/bin/python ../run_scripts/pf_nt/pfn_decide.py --manifest ../results/pf_nt/analysis/manifest.json --out decision.json
        [--meta pf_nt_q38.i32.meta.json] [--raw-b38 PF-BF16 decision.json] [--raw-q38 PF-CONF decision_q38.json]
        [--late F] [--late D] [--score-field F --text-field T] [--boot 10000]

Manifest (analysis/manifest.json): exactly one run dir per (model, pass), every key present:
  {"b38": {"1": "<run dir>", "2": "<run dir>"}, "q38": {"1": "<run dir>", "2": "<run dir>"}}
Relative paths are resolved against the manifest's directory. A value may be null when no run dir holds that (model, pass)
(the model is then INCONCLUSIVE: cohort not exact). A (model, pass)'s units are read only from its run dir, never mixed
across dirs; unit files of that (model, pass) in another manifest dir are ignored and listed. A duplicated key, a missing
key, an unknown key, a list value or a run dir without status.txt is an error (exit 2, no decision). The manifest names the
dir that ran each (model, pass); whether its rows are accepted is decided here (row 1).

Rows are scored with port_r_score.py's hit() on post() of the output decoded with skip_special_tokens=True, tokenizer
Qwen/Qwen3.8-27B at revision 37561816e66b5b9153b1b64c10968c18c1e6f4a0; an empty output decodes to "" and scores 0
(pfn_gates.Scorer). --score-field and --text-field read a numeric score and the decoded text from row fields instead
(synthetic tests only). --meta defaults to results/pf_nt/inputs/pf_nt_q38.i32.meta.json.

Gate F, per format, recomputed from the rows whether or not the lead checked it live: b38 from the (b38, 1) dir's units
1.11 and 1.8, q38 from the (q38, 2) dir's units 2.11 and 2.8. A format's F is unavailable if one of its two unit files is
missing or its rows are invalid (pfn_check's unit validation: coverage, counts, length, sparse engagement, timings);
otherwise the 20 dense outputs are classified with src/pfb_output_kinds.py's rules (pfn_gates.kind) and F passes for the
format iff none contains the string "<think>" and at most 2 are empty or blank. Gate F: UNAVAILABLE if either format is
unavailable; else FAIL if either fails; else PASS.
Rows are schema-checked before anything reads them (pfn_check.schema_problem: id, config, len, out, pf,
prefill_compute_s); a malformed row makes its unit invalid (F unavailable, the cohort not exact) and never stops the
decision from being written.
Gate D, per model: the task-0 dense unit (b38 1.0 from the (b38, 1) dir, q38 2.0 from the (q38, 2) dir) >= 80.
STOP: whether the lead's live STOP happened (a STOP file in a manifest dir, or a "FAILED stopped by the lead" line). A STOP
never invalidates accepted rows. --late records the gates the lead read after the fact.

Quality decision table, per model, first matching row wins:
  1 INVALID       in a manifest dir of this model: SETUP DONE, SOURCE OK, a GATE K blocks/sparse/pages/ops PASS line or
                  GATE U PASS missing; or "<MODEL> UNITS COMPLETE" missing; or no terminal line (COMPLETE or FAILED ...).
                  A FAILED line from the other model's units, or a STOP, does not invalidate a model already complete.
  2 INCONCLUSIVE  the model's cohort (pass 1 from its (model, 1) dir, pass 2 from its (model, 2) dir) is not exact: 13 tasks
                  x 20 prompts x 4 configs, each (id, config) once, counts valid, timings > 0 (pfn_check)
  2n INCONCLUSIVE (amendment 1) any degenerate row in the model's accepted rows: >= 4 generated tokens, at least half of
                  them token id 0 (a numerical failure of the prefill, not attention quality)
  3 INCONCLUSIVE  the model's dense below its floor on any task: 90 on each niah task; vt 80, cwe 60, fwe 70, qa_1 40,
                  qa_2 10 (with EPS 1e-9: a score at the floor is not below it)
  3d INCONCLUSIVE gate D fails for the model (task-0 dense < 80; acceptance and recovery)
  4 PASS          pf:0.1:512:d512+quest:27648:f3 passes the gate: delta >= -3 and the 95% bootstrap lower bound >= -6
  5 FAIL          otherwise
Delta = mean over the 13 tasks of each task's mean paired difference (config - dense) x 100, 20 pairs per task. Bootstrap:
--boot replicates, np.random.default_rng(0); per replicate one resample of the 20 prompt indices per task, shared by every
config (index arrays drawn task by task, shape (boot, 20)). Both bounds inclusive with EPS 1e-9. As PF-CONF, PF-BF16.

Outcome, first match: FORMAT_FAIL (gate F FAIL, inputs available); NO_DECISION (gate F UNAVAILABLE, or either model INVALID
or INCONCLUSIVE); BOTH_PASS; Q4_ONLY (q38 PASS, b38 FAIL); BF16_ONLY (b38 PASS, q38 FAIL); NEITHER (both FAIL).

Reported, never changing the outcome: per model the other two configs' stats, each config per pass (the absolute mean and
per-task scores of all four configs, dense included, and the delta, CI and gate of the three others: 10 prompts per task,
a fresh default_rng(0) with shape (boot, 10)), per-task scores and deltas, empty outputs, output kinds per config and per task;
b38 combined delta minus q38 combined delta, per task as well (descriptive); each model against its raw-prompt run (b38:
PF-BF16's decision.json; q38: PF-CONF's decision_q38.json), per task and config, descriptive only; token identity of b38
and q38 dense rows on the same prompts (complete only for all 260); the per-prompt gather / dense prefill_compute_s ratio
per model (median, max, count above 0.85); the terminal line of each manifest dir; ignored unit files.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

H = Path(__file__).resolve().parent
sys.path.insert(0, str(H / "box" / "in"))
sys.path.insert(0, str(H))
from pfn_check import G_REF, PF, cohort_rows, pass_ids, schema_problem, unit_problems  # noqa: E402
from pfn_gates import EPS, KINDS, META, Scorer, gate_d, gate_f  # noqa: E402

MODELS = ("b38", "q38")
PASSES = (1, 2)
Q = "quest:27648:f3"
COMB = f"{PF}+{Q}"
CFGS = ["dense", Q, PF, COMB]
TASKS = ["niah_single_1", "niah_single_2", "niah_single_3", "niah_multikey_1", "niah_multikey_2", "niah_multikey_3",
         "niah_multivalue", "niah_multiquery", "vt", "cwe", "fwe", "qa_1", "qa_2"]
FLOORS = {t: 90.0 for t in TASKS if t.startswith("niah")} | {"vt": 80.0, "cwe": 60.0, "fwe": 70.0, "qa_1": 40.0, "qa_2": 10.0}
F_UNITS = {"b38": (1, [11, 8]), "q38": (2, [11, 8])}   # the first two units of the box that runs that format first
D_UNITS = {"b38": (1, 0), "q38": (2, 0)}
REQUIRED = ["SETUP DONE", "SOURCE OK", "GATE K blocks PASS", "GATE K sparse PASS", "GATE K pages PASS", "GATE K ops PASS",
            "GATE U PASS"]
EXP = H.parents[1]
RAW = {"b38": EXP / "results" / "pf_bf16" / "analysis" / "decision.json",
       "q38": EXP / "results" / "pf_conf" / "analysis" / "decision_q38.json"}
RAW_NOTE = {"b38": "PF-BF16: raw prompts, BF16, f16 KV; differs in prompt format, box and unit placement (descriptive only)",
            "q38": "PF-CONF: raw prompts, UD-Q4_K_M, q8_0 KV; differs in prompt format, KV type, card, driver and toolkit "
                   "(descriptive only)"}
OUTCOME = {("PASS", "PASS"): "BOTH_PASS", ("FAIL", "PASS"): "Q4_ONLY", ("PASS", "FAIL"): "BF16_ONLY", ("FAIL", "FAIL"): "NEITHER"}
Manifest = dict[tuple[str, int], Path | None]
Score = dict[tuple[int, str], float]


def load_manifest(path: Path) -> Manifest:
    def no_dups(pairs: list[tuple[str, object]]) -> dict[str, object]:
        keys = [k for k, _ in pairs]
        dup = sorted({k for k in keys if keys.count(k) > 1})
        if dup:
            raise SystemExit(f"manifest {path}: duplicated key(s) {dup}")
        return dict(pairs)

    raw = json.loads(path.read_text(), object_pairs_hook=no_dups)
    if not isinstance(raw, dict) or set(raw) != set(MODELS):
        raise SystemExit(f"manifest {path}: top-level keys must be exactly {list(MODELS)}")
    out: Manifest = {}
    for m in MODELS:
        if not isinstance(raw[m], dict) or set(raw[m]) != {"1", "2"}:
            raise SystemExit(f"manifest {path}: {m} must map exactly the passes \"1\" and \"2\" to one run dir (or null)")
        for p in PASSES:
            v = raw[m][str(p)]
            if v is None:
                out[(m, p)] = None
                continue
            if not isinstance(v, str):
                raise SystemExit(f"manifest {path}: {m} pass {p} must be one run dir, got {v!r}")
            d = Path(v) if Path(v).is_absolute() else (path.parent / v)
            if not (d / "status.txt").is_file():
                raise SystemExit(f"manifest {path}: {m} pass {p}: no status.txt in {d}")
            out[(m, p)] = d
    return out


def status_lines(run: Path) -> list[str]:
    return (run / "status.txt").read_text().splitlines() if (run / "status.txt").exists() else []


def terminal(run: Path) -> str | None:
    t = [l for l in status_lines(run) if l.split(" ")[0] == "COMPLETE" or l.startswith("FAILED")]
    return t[-1] if t else None


def quality_invalid(run: Path, model: str) -> str | None:
    """the reason a manifest dir makes the model's quality INVALID (decision row 1), or None"""
    status = status_lines(run)
    missing = [x for x in REQUIRED + [f"{model.upper()} UNITS COMPLETE"] if not any(l.startswith(x) for l in status)]
    if missing:
        return f"{run}: missing status lines: {missing}"
    if terminal(run) is None:
        return f"{run}: no terminal line (COMPLETE or FAILED)"
    return None


def pass_rows(man: Manifest, model: str, p: int, tasks: list[int]) -> tuple[list[dict], list[str]]:
    """rows of (model, p, task) units from the (model, p) manifest dir only"""
    d = man[(model, p)]
    if d is None:
        return [], [f"{model} pass {p}: no run dir in the manifest"]
    return cohort_rows([d / "ruler"], model, [(p, t) for t in tasks])


def gate_stats(s: Score, c: str, prompts: list[int], boot: int, base: str = "dense") -> dict[str, object]:
    """delta, ci95, pass of config c against base over the given prompt indices of every task (aligned bootstrap)."""
    rng = np.random.default_rng(0)
    n = len(prompts)
    idx = [rng.integers(0, n, (boot, n)) for _ in range(13)]
    diff = [np.array([s[(t * 1000 + i, c)] - s[(t * 1000 + i, base)] for i in prompts]) for t in range(13)]
    delta = 100 * float(np.mean([d.mean() for d in diff]))
    boots = 100 * np.mean([diff[t][idx[t]].mean(1) for t in range(13)], axis=0)
    lo, hi = float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))
    return {"delta": delta, "ci95": [lo, hi], "pass": bool(delta >= -3 - EPS and lo >= -6 - EPS), "n_pairs": 13 * n}


def degenerate(r: dict) -> bool:
    """amendment 1: >= 4 generated tokens, at least half of them token id 0 (NaN logits decode as id 0)"""
    o = r["out"]
    return len(o) >= 4 and 2 * sum(1 for t in o if t == 0) >= len(o)


def quality(a: argparse.Namespace, man: Manifest, model: str, sc: Scorer, gd: dict[str, object],
            res: dict[str, object]) -> tuple[str, str]:
    """(decision, reason) of the model's quality table; fills res with the stats"""
    for run in sorted({d for (m, _), d in man.items() if m == model and d is not None}, key=str):
        why = quality_invalid(run, model)
        if why:
            return "INVALID", why

    rows, probs = [], []
    for p in PASSES:
        r, pr = pass_rows(man, model, p, list(range(13)))
        rows += r; probs += pr
    ids = {i for p in PASSES for t in range(13) for i in pass_ids(t, p)}
    cov, inv, ratios = unit_problems(rows, ids, CFGS, model)
    probs += cov + inv
    # informative, over the well-formed rows only (unit_problems drops malformed rows before computing ratios)
    pt = {(r["id"], r["config"]): r["prefill_compute_s"] for r in rows if schema_problem(r) is None}
    res["gather_dense_prefill"] = {"median": float(np.median(ratios)) if ratios else None,
                                   "max": float(max(ratios)) if ratios else None, "n": len(ratios),
                                   "n_above_0.85": sum(x > G_REF[model] for x in ratios),
                                   "per_prompt": {str(i): pt[(i, PF)] / pt[(i, "dense")] for i, c in sorted(pt, key=str)
                                                  if c == "dense" and (i, PF) in pt and pt[(i, PF)] > 0 and pt[(i, c)] > 0}}
    if probs:
        return "INCONCLUSIVE", f"{model} cohort not exact: {probs[:3]}"

    deg = sorted((r["id"], r["config"]) for r in rows if degenerate(r))
    res["degenerate_rows"] = {"n": len(deg), "rows": [f"{i}:{c}" for i, c in deg]}
    s = {(r["id"], r["config"]): sc.score(model, r) for r in rows}
    all20, p1, p2 = list(range(20)), list(range(10)), list(range(10, 20))
    per_task = {c: {TASKS[t]: 100 * float(np.mean([s[(t * 1000 + i, c)] for i in all20])) for t in range(13)} for c in CFGS}
    res["per_task"] = per_task
    res["per_pass"] = {str(p): {"per_task": {c: {TASKS[t]: 100 * float(np.mean([s[(t * 1000 + i, c)] for i in ix]))
                                                 for t in range(13)} for c in CFGS}} for p, ix in ((1, p1), (2, p2))}
    for v in res["per_pass"].values():
        v["mean"] = {c: float(np.mean(list(v["per_task"][c].values()))) for c in CFGS}
    res["per_task_delta"] = {c: {t: per_task[c][t] - per_task["dense"][t] for t in TASKS} for c in CFGS if c != "dense"}
    res["mean"] = {c: float(np.mean(list(per_task[c].values()))) for c in CFGS}
    res["empty_outputs"] = {c: sum(1 for r in rows if r["config"] == c and not r["out"]) for c in CFGS}
    kinds = {(r["id"], r["config"]): sc.kind(model, r) for r in rows}
    res["kinds"] = {c: {k: sum(1 for (i, cc), v in kinds.items() if cc == c and v == k) for k in KINDS} for c in CFGS}
    res["kinds_per_task"] = {c: {TASKS[t]: {k: sum(1 for (i, cc), v in kinds.items() if cc == c and i // 1000 == t and v == k)
                                            for k in KINDS} for t in range(13)} for c in CFGS}
    stats = {c: gate_stats(s, c, all20, a.boot) for c in (COMB, Q, PF)}
    res["stats"] = stats
    res["stats_pass1"] = {c: gate_stats(s, c, p1, a.boot) for c in (COMB, Q, PF)}
    res["stats_pass2"] = {c: gate_stats(s, c, p2, a.boot) for c in (COMB, Q, PF)}

    if deg:
        return "INCONCLUSIVE", f"{model} has {len(deg)} degenerate rows (token id 0 runs), e.g. {res['degenerate_rows']['rows'][:4]}"
    res["dense_floors"] = {t: {"dense": per_task["dense"][t], "floor": FLOORS[t]} for t in TASKS}
    low = [t for t in TASKS if per_task["dense"][t] < FLOORS[t] - EPS]
    if low:
        return "INCONCLUSIVE", f"{model} dense below its floor on {low}"
    if gd["status"] != "PASS":
        return "INCONCLUSIVE", f"{model} gate D {gd['status']}: {gd['reason']}"
    st = stats[COMB]
    if st["pass"]:
        return "PASS", f"{COMB} delta {st['delta']:+.2f} [{st['ci95'][0]:+.2f}, {st['ci95'][1]:+.2f}] passes the gate"
    return "FAIL", (f"{COMB} delta {st['delta']:+.2f} [{st['ci95'][0]:+.2f}, {st['ci95'][1]:+.2f}] fails the gate "
                    f"(delta >= -3 and lower bound >= -6)")


def unit_input(man: Manifest, model: str, p: int, tasks: list[int]) -> tuple[list[dict], list[str]]:
    """the dense rows of gate inputs (units of one (model, p)), with the problems that make them unavailable"""
    rows, probs = pass_rows(man, model, p, tasks)
    if not probs:
        cov, inv, _ = unit_problems(rows, {i for t in tasks for i in pass_ids(t, p)}, CFGS, model)
        probs = cov + inv
    if probs:
        return [], probs
    return sorted((r for r in rows if r["config"] == "dense"), key=lambda r: r["id"]), probs


def live_gates(man: Manifest, sc: Scorer) -> dict[str, object]:
    f: dict[str, object] = {}
    for model, (p, tasks) in F_UNITS.items():
        dense, probs = unit_input(man, model, p, tasks)
        if probs:
            g = {"status": "UNAVAILABLE", "reason": f"input units missing or invalid: {probs[:3]}"}
        else:
            g = gate_f([sc.kind(model, r) for r in dense])
            g["rows"] = [{"id": r["id"], "kind": sc.kind(model, r), "n_tokens": len(r["out"])} for r in dense]
        g.update(run=str(man[(model, p)]), units=[f"{p}.{t}" for t in tasks])
        f[model] = g
    sts = [f[m]["status"] for m in F_UNITS]
    f["status"] = "UNAVAILABLE" if "UNAVAILABLE" in sts else "FAIL" if "FAIL" in sts else "PASS"
    d: dict[str, object] = {}
    for model, (p, t) in D_UNITS.items():
        dense, probs = unit_input(man, model, p, [t])
        g = ({"status": "UNAVAILABLE", "score": None, "reason": f"task-0 unit missing or invalid: {probs[:3]}"} if probs
             else gate_d([sc.score(model, r) for r in dense]))
        g.update(run=str(man[(model, p)]), unit=f"{p}.{t}")
        d[model] = g
    return {"gate_f": f, "gate_d": d}


def stop_info(runs: list[Path], gates: dict[str, object], late: list[str]) -> dict[str, object]:
    per = {str(r): {"stop_file": (r / "STOP").exists(),
                    "lead_stop_lines": [l for l in status_lines(r) if l.startswith("FAILED stopped by the lead")]}
           for r in runs}
    happened = any(v["stop_file"] or v["lead_stop_lines"] for v in per.values())
    notes = []
    if gates["gate_f"]["status"] == "FAIL" and not happened:
        notes.append("gate F fails on the rows and no live STOP happened")
    for m in MODELS:
        if gates["gate_d"][m]["status"] == "FAIL" and not happened:
            notes.append(f"gate D fails on the rows for {m} and no live STOP happened")
    return {"live_stop": happened, "per_run": per, "late_gates": sorted(set(late)), "notes": notes}


def ignored_units(man: Manifest) -> list[str]:
    """unit files of a (model, pass) that sit in another manifest dir than its own (not read)"""
    dirs = sorted({d for d in man.values() if d is not None}, key=str)
    out = []
    for (m, p), own in sorted(man.items()):
        for d in dirs:
            if d != own:
                out += [str(f) for f in sorted((d / "ruler").glob(f"{m}_p{p}_t*.jsonl"))]
    return out


def vs_raw(model: str, res: dict[str, object], path: Path) -> dict[str, object]:
    out: dict[str, object] = {"raw": str(path), "note": RAW_NOTE[model]}
    if not path.is_file():
        out["missing"] = True
        return out
    raw = json.loads(path.read_text())
    if "per_task" not in res or "per_task" not in raw:
        out["compared"] = False
        return out
    here, rp = res["per_task"], raw["per_task"]
    out["per_task"] = {c: {t: {"here": here[c][t], "raw": rp[c][t], "here_minus_raw": here[c][t] - rp[c][t]} for t in TASKS}
                       for c in CFGS if c in rp}
    out["per_task_delta"] = {c: {t: {"here": here[c][t] - here["dense"][t], "raw": rp[c][t] - rp["dense"][t]} for t in TASKS}
                             for c in CFGS if c != "dense" and c in rp}
    out["delta"] = {c: {"here": res["stats"][c]["delta"], "raw": raw.get("stats", {}).get(c, {}).get("delta")}
                    for c in (COMB, Q, PF)}
    out["mean"] = {c: {"here": res["mean"][c], "raw": raw.get("mean", {}).get(c)} for c in CFGS}
    return out


def identity(man: Manifest) -> dict[str, object]:
    out = {}
    for m in MODELS:
        rows = [r for p in PASSES for r in pass_rows(man, m, p, list(range(13)))[0]]
        out[m] = {r["id"]: r["out"] for r in rows if schema_problem(r) is None and r["config"] == "dense"}
    ids = sorted(set(out["b38"]) & set(out["q38"]))
    same = [i for i in ids if out["b38"][i] == out["q38"][i]]
    return {"identical": len(same), "n": len(ids), "complete": len(ids) == 260,
            "per_task": {TASKS[t]: f"{sum(1 for i in same if i // 1000 == t)}/{sum(1 for i in ids if i // 1000 == t)}"
                         for t in range(13)}}


def decide(a: argparse.Namespace) -> int:
    man = load_manifest(Path(a.manifest))
    runs = sorted({d for d in man.values() if d is not None}, key=str)
    sc = Scorer(a.meta, a.score_field, a.text_field)
    res: dict[str, object] = {"manifest": str(a.manifest), "runs": {f"{m} pass {p}": str(d) for (m, p), d in sorted(man.items())},
                              "terminal_line": {str(r): terminal(r) for r in runs}, "ignored_unit_files": ignored_units(man)}
    gates = live_gates(man, sc)
    res.update(gates)
    res["stop"] = stop_info(runs, gates, a.late)
    q: dict[str, tuple[str, str]] = {}
    res["models"] = {}
    for m in MODELS:
        rm: dict[str, object] = {}
        q[m] = quality(a, man, m, sc, gates["gate_d"][m], rm)
        rm.update(decision=q[m][0], reason=q[m][1])
        rm["vs_raw"] = vs_raw(m, rm, Path(getattr(a, f"raw_{m}")))
        res["models"][m] = rm
    mb, mq = res["models"]["b38"], res["models"]["q38"]
    if "stats" in mb and "stats" in mq:
        res["b38_minus_q38"] = {"combined_delta": mb["stats"][COMB]["delta"] - mq["stats"][COMB]["delta"],
                                "per_task_combined_delta": {t: mb["per_task_delta"][COMB][t] - mq["per_task_delta"][COMB][t]
                                                            for t in TASKS}, "note": "descriptive"}
    res["dense_token_identity_b38_q38"] = identity(man)

    qb, qq, gf = q["b38"][0], q["q38"][0], gates["gate_f"]
    if gf["status"] == "FAIL":
        oc, why = "FORMAT_FAIL", "gate F failed: " + "; ".join(f"{m} {gf[m]['status']} ({gf[m]['reason']})" for m in F_UNITS)
    elif gf["status"] == "UNAVAILABLE" or qb not in ("PASS", "FAIL") or qq not in ("PASS", "FAIL"):
        oc, why = "NO_DECISION", f"gate F {gf['status']}, b38 {qb}, q38 {qq}"
    else:
        oc, why = OUTCOME[(qb, qq)], f"gate F PASS, b38 {qb}, q38 {qq}"
    res.update(outcome=oc, outcome_reason=why)
    Path(a.out).write_text(json.dumps(res, indent=1))
    gd = gates["gate_d"]
    print(f"GATE F {gf['status']}: " + "; ".join(f"{m} {gf[m]['status']} ({gf[m]['reason']})" for m in F_UNITS))
    print("GATE D " + "; ".join(f"{m} {gd[m]['status']} ({gd[m]['reason']})" for m in MODELS))
    print(f"STOP {'yes' if res['stop']['live_stop'] else 'no'}" + (f": {'; '.join(res['stop']['notes'])}" if res['stop']['notes'] else ""))
    for m in MODELS:
        print(f"DECISION {m} {q[m][0]}: {q[m][1]}")
    print(f"OUTCOME {oc}: {why}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True, help="analysis/manifest.json: one run dir per (model, pass)")
    ap.add_argument("--meta", default=str(META))
    ap.add_argument("--raw-b38", default=str(RAW["b38"])); ap.add_argument("--raw-q38", default=str(RAW["q38"]))
    ap.add_argument("--late", action="append", default=[], choices=("F", "D"), help="a gate the lead read after the fact")
    ap.add_argument("--out", required=True); ap.add_argument("--score-field"); ap.add_argument("--text-field")
    ap.add_argument("--boot", type=int, default=10000)
    return decide(ap.parse_args())


if __name__ == "__main__":
    sys.exit(main())
