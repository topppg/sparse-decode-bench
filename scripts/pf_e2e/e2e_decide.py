"""PF-E2E decision (on the Mac, after the pulls): the speed decision of each speed arm (A: RTX 5090, 256k and 128k; D:
RTX 4090, 128k), the RULER-at-256k quality decision (boxes B and C), and the registered consequence row.

  cd experiments/src
  ../.venv/bin/python ../run_scripts/pf_e2e/e2e_decide.py --out ../results/pf_e2e/analysis/decision.json
        [--speed-a <run dir>] [--speed-d <run dir>] [--ruler-manifest ../results/pf_e2e/analysis/manifest.json]
        [--book-meta ...] [--ruler-meta ...] [--late F] [--late D] [--amend4] [--score-field F --text-field T] [--boot 10000]
A part that is not given is NOT_RUN. Relative manifest paths resolve against the manifest's directory.

SPEED, per arm (run dir of run_pfe_speed.sh):
  INVALID   a required status line is missing (GATE P PASS, PROMPTS OK, SOURCE OK, UPSTREAM OK, BUILD DONE fork and
            upstream, the eight GATE K lines, MODEL OK, SETUP DONE, REF <len> OK per length), no terminal line (COMPLETE,
            INCOMPLETE, FAILED), the run's matrix SHA-256 differs from the registered pfe_matrix.tsv, or the prompt
            lengths differ from the book meta
  per (length, block, config) of the registered matrix: OK | OOM | INVALID | MISSING, recomputed from the files with
            e2e_check.srv_verdict (two valid timed rows, n_predict 1000, prompt_n = the meta length, MTP drafted, the
            conversion-count rule against the reference's count recomputed with e2e_check.ref_check); a config with sparse
            decode is INVALID in a block where its decode tok/s is below SD_MIN x fork_dense's (1.30 for pfsd27648 at 256,
            1.15 otherwise; the engagement check, since the server has no sparse-decode counter); a config OOM in block 1
            and absent in block 2 is OOM there
  best dense of a block: the lowest median wall among the OK dense candidates (fork_dense, fork_dense_ub2048, fork_mtp,
            up_dense, up_mtp); none if any candidate is INVALID or MISSING (an OOM candidate is left out), or none is OK
  quoted ratio R_L(config) = min over blocks 1, 2 of (best dense median wall / config median wall); none if either block
            lacks a best dense or the config is not OK in both blocks. Also: the TTFT ratio against the same best dense and
            against fork_dense; decode ratio; drift (block 2 / block 1 median wall, flagged beyond +-3%); determinism (all
            timed outputs of a config identical, flagged otherwise); gate N (fork_dense's first timed output: >= 50%
            distinct 8-grams; first end-of-turn position) with MTP rows labelled on a failure; gate B per length (the two
            saved-logit SHA-256s equal); S (informative: first-64-token prefix of each config against its sparse-tail
            reference); peak memory.used per row from gpu_log.csv and stages.txt (informative)
  arm D also: D-043's rule on the registered measurement: fork_dense TTFT / pfsd27648 TTFT at 128k, lower of the two
            blocks, >= 1.30 cleared, else triggered (sparse prefill stops; negative speed result)
  The 4090 arm's quality rests on GATE B (GATHER=2 bit-identical to GATHER=1) plus PF-NT and PF-CONF at 128k only when
  gate B passed on that card; otherwise the inherited quality is unavailable and the speed result stands alone.

QUALITY (RULER at 256k), manifest {"q38": {"1": <box B run dir>, "2": <box C run dir>}}, first match:
  INVALID       a manifest dir lacks SETUP DONE, SOURCE OK, a GATE K line, GATE A PASS or Q38 UNITS COMPLETE, or has no
                terminal line
  INCONCLUSIVE  cohort not exact (13 x 20 x 5, pfn_check.unit_problems), a unit's selective-conversion line missing or
                wrong (e2e_check ruler-sel rule), or any degenerate row (PF-NT amendment 1: >= 4 tokens, half or more id 0)
  FORMAT_FAIL   gate F fails (B's units 1.11 and 1.8; pfn_gates.gate_f); UNAVAILABLE inputs give NO_DECISION
  UNINFORMATIVE gate D fails (B's 1.0 dense < 80) or the dense 13-task mean < 60
  PASS_27648    pf+quest:27648 passes the PORT-R gate (delta >= -3 and 95% lower bound >= -6; pfn_decide.gate_stats)
  PASS_55296    else pf+quest:55296 passes
  PASS_PF_ONLY  else pf:0.1:512:d512 alone passes
  FAIL_ALL      otherwise
  --amend4 (PF-E2E amendment 4, written after the data): pass 1's run may end "FAILED pass 1 task 12" without
                Q38 UNITS COMPLETE when that unit's three files are in the run's tmp/ (where the box left the unit it
                rejected); the unit is then read from tmp/ and re-checked here with the corrected count rule like every other.
  Outcome NO_DECISION for INVALID, INCONCLUSIVE or gate F UNAVAILABLE. Reported: every config's stats (both passes and per
  pass), per-task scores and deltas, empty outputs and kinds, the 12-task delta without qa_1 (descriptive; RULER repeats
  SQuAD documents at this length), gate D and F on box C's units (descriptive), dense per task against PF-NT's 128k q38
  dense (descriptive: other length, KV type and card).

CONSEQUENCE (protocol table), from arm A and the quality outcome. The claim config comes from the quality outcome first
  (PASS_27648 -> pfsd27648, PASS_55296 -> pfsd55296, PASS_PF_ONLY -> pf_only; otherwise pfsd27648 for the speed-only rows),
  then its own R_256 is required (none -> NO_RATIO). Gain band of that R_256 (absolute 256k gain, not a length claim):
  >= 1.8 HIGH, 1.4-1.8 MODERATE, < 1.4 LOW. Rows: <quality>_<gain> for a PASS; NO_QUALITY_CLAIM for FAIL_ALL; SPEED_ONLY
  for NOT_RUN, UNINFORMATIVE, FORMAT_FAIL or NO_DECISION. A fallback row says the primary config did not pass the gate,
  not that quality was lost. Length effect, separate (arm A, each sparse config): R_256, R_128, R_256 - R_128 and
  R_256 / R_128; GROWS iff R_256 - R_128 >= 0.3 (P7's registered margin), else NOT_SHOWN; UNAVAILABLE with the reason when
  either ratio is missing. Wording that the gain grows with context is allowed only if the claim config's own length
  effect is GROWS; pfsd27648's is reported alongside.
  Arm D: the quality basis sentence only with gate B PASS on that card; otherwise "inherited quality unavailable".
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import re
import statistics
import sys
from pathlib import Path

import numpy as np

H = Path(__file__).resolve().parent
EXP = H.parents[1]
sys.path.insert(0, str(H / "box" / "in"))
sys.path.insert(0, str(H.parent / "pf_nt"))
from e2e_check import BLOCK_CONFIGS, CONFIGS, read_matrix, ref_check, sel_counts, srv_verdict  # noqa: E402
from pfe_check import read_rows  # noqa: E402
from pfn_check import cohort_rows, pass_ids, schema_problem, unit_problems  # noqa: E402
from pfn_decide import degenerate, gate_stats, status_lines, terminal  # noqa: E402
from pfn_gates import EPS, KINDS, Scorer, gate_d, gate_f  # noqa: E402

MATRIX = H / "box" / "in" / "pfe_matrix.tsv"
BOOK_META = EXP / "results" / "pf_e2e" / "inputs" / "pfe_book_q38.i32.meta.json"
RULER_META = EXP / "results" / "pf_e2e" / "inputs" / "pfe_ruler256_q38.i32.meta.json"
PFNT_DECISION = EXP / "results" / "pf_nt" / "analysis" / "decision.json"
LENS = {"A": (256, 128), "D": (128,)}
PF = "pf:0.1:512:d512"
REFCFG = ["dense", PF, f"{PF}+quest:27648:f3", f"{PF}+quest:55296:f3"]
REF_OF = {"pfsd27648": f"{PF}+quest:27648:f3", "pfsd55296": f"{PF}+quest:55296:f3", "pf_only": PF}
DENSE = [c for c in BLOCK_CONFIGS if CONFIGS[c]["role"] == "dense"]
SPARSE = [c for c in BLOCK_CONFIGS if CONFIGS[c]["role"] == "sparse"]
GATE_K = ["blocks", "sparse", "pages", "ops", "select_c1", "select_c0", "select_c2", "fa_all"]
REPS, N_PREDICT, DRIFT, N_MIN, D043, LE_MARGIN = 2, 1000, 0.03, 0.5, 1.30, 0.3

Q, Q2 = "quest:27648:f3", "quest:55296:f3"
C27, C55 = f"{PF}+{Q}", f"{PF}+{Q2}"
QCFGS = ["dense", Q, PF, C27, C55]
TASKS = ["niah_single_1", "niah_single_2", "niah_single_3", "niah_multikey_1", "niah_multikey_2", "niah_multikey_3",
         "niah_multivalue", "niah_multiquery", "vt", "cwe", "fwe", "qa_1", "qa_2"]
QREQ = ["SETUP DONE", "SOURCE OK", *[f"GATE K {k} PASS" for k in GATE_K], "GATE A PASS", "Q38 UNITS COMPLETE"]
F_UNITS, D_UNIT = (1, [11, 8]), (1, 0)
DENSE_MIN = 60.0
AMEND4_UNIT = (1, 12)
AMEND4_TERM = re.compile(r"^FAILED pass 1 task 12 \(rc 2, see status\)$")


def sd_min(length: int, cfg: str) -> float:
    return 1.30 if (length == 256 and cfg == "pfsd27648") else 1.15


def has(lines: list[str], prefix: str) -> bool:
    return any(l.startswith(prefix) for l in lines)


def med(xs: list[float]) -> float:
    return float(statistics.median(xs))


def distinct_8grams(t: list[int]) -> float:
    g = [tuple(t[i:i + 8]) for i in range(len(t) - 7)]
    return len(set(g)) / len(g) if g else 0.0


def prefix_eq(x: list[int], y: list[int]) -> int:
    n = 0
    for p, q in zip(x, y):
        if p != q:
            break
        n += 1
    return n


def peaks(run: Path) -> dict[str, int]:
    """peak memory.used (MiB) per stage from gpu_log.csv (1 s samples) and stages.txt; informative"""
    out: dict[str, int] = {}
    g, s = run / "gpu_log.csv", run / "stages.txt"
    if not (g.is_file() and s.is_file()):
        return out
    samples = []
    for line in g.read_text().splitlines():
        f = [x.strip() for x in line.split(",")]
        try:
            t = dt.datetime.strptime(f[0], "%Y/%m/%d %H:%M:%S.%f").replace(tzinfo=dt.timezone.utc).timestamp()
            samples.append((t, int(f[1].split()[0])))
        except (ValueError, IndexError):
            continue
    st = dict(l.split()[:2] for l in s.read_text().splitlines() if len(l.split()) >= 2)
    for k, v in st.items():
        if k.endswith("_start") and k[:-6] + "_end" in st:
            a, b = int(v), int(st[k[:-6] + "_end"])
            m = [x for t, x in samples if a <= t <= b]
            if m:
                out[k[:-6]] = max(m)
    return out


def speed_arm(arm: str, run: Path, meta: dict) -> dict[str, object]:
    res: dict[str, object] = {"run": str(run)}
    lines = status_lines(run)
    lens = LENS[arm]
    req = ["GATE P PASS", "PROMPTS OK", "SOURCE OK", "UPSTREAM OK", "BUILD DONE fork", "BUILD DONE upstream",
           *[f"GATE K {k} PASS" for k in GATE_K], "MODEL OK", "SETUP DONE", *[f"REF {L} OK" for L in lens]]
    missing = [x for x in req if not has(lines, x)]
    term = [l for l in lines if l.split(" ")[0] in ("COMPLETE", "INCOMPLETE") or l.startswith("FAILED")]
    res["terminal_line"] = term[-1] if term else None
    res["power_line"] = next((l for l in lines if l.startswith("GATE P PASS")), None)
    reg_sha = hashlib.sha256(MATRIX.read_bytes()).hexdigest()
    run_sha = [l.split()[0] for l in (run / "sha.txt").read_text().splitlines() if "pfe_matrix.tsv" in l] if (run / "sha.txt").is_file() else []
    nlen = {L: meta[str(L)]["len"] for L in lens}
    plen = next((l for l in lines if l.startswith("PROMPTS OK")), "")
    why = []
    if missing:
        why.append(f"missing status lines {missing}")
    if not term:
        why.append("no terminal line")
    if run_sha != [reg_sha]:
        why.append(f"matrix SHA-256 {run_sha} != registered {reg_sha[:12]}")
    if plen and any(f"{L}:{nlen[L]}" not in plen for L in lens):
        why.append(f"prompt lengths {plen!r} != meta {nlen}")
    if why:
        res.update(status="INVALID", reason="; ".join(why))
        return res

    rows, _ = read_matrix(MATRIX)
    mrows = [(L, b, c) for a, L, b, c in rows if a == arm]
    ends = list(meta["end_ids"].values())
    # gate B and the references
    gb, refs = {}, {}
    shas = {}
    if (run / "bit" / "bin_sha256.txt").is_file():
        for l in (run / "bit" / "bin_sha256.txt").read_text().splitlines():
            if len(l.split()) == 2:
                shas[l.split()[1].lstrip("*")] = l.split()[0]
    for L in lens:
        a, b = shas.get(f"lg_{L}_g1.bin"), shas.get(f"lg_{L}_g2.bin")
        gb[str(L)] = {"status": "MISSING" if not (a and b) else "PASS" if a == b else "FAIL", "sha_g1": a, "sha_g2": b,
                      "status_line": next((l for l in lines if l.startswith(f"GATE B {L} ")), None)}
        sel, probs, rrows = ref_check(run / "ref" / f"gen_{L}.jsonl", run / "ref" / f"gen_{L}.err", nlen[L], REFCFG)
        refs[L] = {"sel": sel if not probs else None, "problems": probs[:5],
                   "out": {r["config"]: r.get("out", []) for r in rrows if isinstance(r, dict) and "config" in r}}
    res["gate_b"] = gb
    res["references"] = {str(L): {"selective_per_prefill": v["sel"], "problems": v["problems"]} for L, v in refs.items()}

    cell: dict[tuple[int, str, str], dict[str, object]] = {}
    for L, b, c in mrows:
        f = run / "e2e" / f"{L}_b{b}_{c}.jsonl"
        log = run / "e2e" / f"srv_{L}_b{b}_{c}.log"
        d: dict[str, object] = {}
        if not f.exists() and not log.exists():
            skip = b == "2" and cell.get((L, "1", c), {}).get("status") == "OOM"
            d["status"] = "OOM" if skip else "MISSING"
            if skip:
                d["note"] = "skipped after OOM in block 1"
        else:
            v = srv_verdict(f, log, c, REPS, N_PREDICT, nlen[L], refs[L]["sel"])
            d["status"] = v.split(" ")[0]
            d["verdict"] = v[:300]
            if d["status"] == "OK":
                timed = [r for r in read_rows(f)[0] if r.get("kind") == "timed"]
                d["wall"] = med([r["wall_s"] for r in timed])
                d["ttft"] = med([r["prompt_ms"] / 1e3 for r in timed])
                d["tps"] = med([r["predicted_n"] / r["predicted_ms"] * 1e3 for r in timed])
                dn = sum(r.get("draft_n") or 0 for r in timed)
                d["acceptance"] = sum(r.get("draft_n_accepted") or 0 for r in timed) / dn if dn else None
                d["tokens"] = [r["tokens"] for r in timed]
        cell[(L, b, c)] = d
    # decode engagement (blocks 1, 2 and i), against fork_dense of the same block (block i: of block 1)
    for (L, b, c), d in cell.items():
        if d["status"] != "OK" or not CONFIGS[c]["sd"] or CONFIGS[c]["kv"] != "q8_0":
            continue
        ref = cell.get((L, b if b != "i" else "1", "fork_dense"), {})
        if ref.get("status") != "OK":
            d["decode_engaged"] = "unchecked (fork_dense not OK in that block)"
            continue
        ratio = d["tps"] / ref["tps"]
        d["decode_ratio"] = ratio
        if ratio < sd_min(L, c) - EPS:
            d["status"] = "INVALID"
            d["verdict"] = f"sparse decode not engaged: decode {ratio:.3f} x fork_dense < {sd_min(L, c)}"
        else:
            d["decode_engaged"] = f"{ratio:.3f} x fork_dense >= {sd_min(L, c)}"

    per_len: dict[str, object] = {}
    for L in lens:
        best = {}
        for b in ("1", "2"):
            ds = {c: cell[(L, b, c)] for c in DENSE}
            if any(v["status"] in ("INVALID", "MISSING") for v in ds.values()):
                best[b] = {"config": None, "reason": "a dense candidate is INVALID or MISSING: "
                           + ", ".join(f"{c} {v['status']}" for c, v in ds.items() if v["status"] in ("INVALID", "MISSING"))}
                continue
            ok = {c: v for c, v in ds.items() if v["status"] == "OK"}
            if not ok:
                best[b] = {"config": None, "reason": "no dense candidate OK"}
                continue
            bc = min(ok, key=lambda c: ok[c]["wall"])
            best[b] = {"config": bc, "wall": ok[bc]["wall"], "ttft": ok[bc]["ttft"], "tps": ok[bc]["tps"]}
        ratios = {}
        for c in SPARSE:
            r = {}
            if all(best[b]["config"] for b in ("1", "2")) and all(cell[(L, b, c)]["status"] == "OK" for b in ("1", "2")):
                blk = {b: best[b]["wall"] / cell[(L, b, c)]["wall"] for b in ("1", "2")}
                r = {"quoted": min(blk.values()), "per_block": blk,
                     "ttft_vs_best": min(best[b]["ttft"] / cell[(L, b, c)]["ttft"] for b in ("1", "2")),
                     "decode_vs_best": min(cell[(L, b, c)]["tps"] / best[b]["tps"] for b in ("1", "2"))}
            else:
                r = {"quoted": None, "reason": f"best dense {[best[b].get('config') for b in ('1', '2')]}, "
                                              f"{c} {[cell[(L, b, c)]['status'] for b in ('1', '2')]}"}
            fd = [cell[(L, b, "fork_dense")] for b in ("1", "2")]
            if all(x["status"] == "OK" for x in fd) and all(cell[(L, b, c)]["status"] == "OK" for b in ("1", "2")):
                r["ttft_vs_fork_dense"] = min(fd[i]["ttft"] / cell[(L, b, c)]["ttft"] for i, b in enumerate(("1", "2")))
                r["wall_vs_fork_dense"] = min(fd[i]["wall"] / cell[(L, b, c)]["wall"] for i, b in enumerate(("1", "2")))
            ratios[c] = r
        drift, determ = {}, {}
        for c in BLOCK_CONFIGS:
            x, y = cell[(L, "1", c)], cell[(L, "2", c)]
            if x["status"] == "OK" and y["status"] == "OK":
                dr = y["wall"] / x["wall"] - 1
                drift[c] = {"b2_over_b1_minus_1": dr, "flag": abs(dr) > DRIFT}
                toks = x["tokens"] + y["tokens"]
                determ[c] = all(t == toks[0] for t in toks)
        gn = {"status": "UNAVAILABLE"}
        fd1 = cell[(L, "1", "fork_dense")]
        if fd1["status"] == "OK":
            t = fd1["tokens"][0]
            fr = distinct_8grams(t)
            gn = {"status": "PASS" if fr >= N_MIN else "FAIL", "distinct_8gram_fraction": fr,
                  "first_end_position": next((i for i, x in enumerate(t) if x in ends), None)}
        s_info = {}
        for (LL, b, c), d in cell.items():
            if LL == L and d["status"] == "OK" and CONFIGS[c]["kv"] == "q8_0":
                rc = REF_OF.get(c) or ("dense" if CONFIGS[c]["role"] == "dense" else None)
                ro = refs[L]["out"].get(rc) if rc else None
                if ro:
                    s_info[f"b{b} {c}"] = prefix_eq(d["tokens"][0], ro)
        per_len[str(L)] = {"best_dense": best, "ratios": ratios, "drift": drift, "deterministic": determ, "gate_n": gn,
                           "mtp_label": None if gn["status"] == "PASS" else "MTP rows measured on degenerate or unscreened text",
                           "server_vs_reference_prefix_64": s_info}
    res["lengths"] = per_len
    res["cells"] = {f"{L} b{b} {c}": {k: v for k, v in d.items() if k != "tokens"} for (L, b, c), d in cell.items()}
    res["peak_memory_mib"] = peaks(run)
    if len(lens) == 2:
        le = {}
        for c in SPARSE:
            r2, r1 = per_len["256"]["ratios"][c].get("quoted"), per_len["128"]["ratios"][c].get("quoted")
            if r2 is None or r1 is None:
                le[c] = {"verdict": "UNAVAILABLE", "reason": f"R_256 {r2}, R_128 {r1}"}
            else:
                le[c] = {"r_256": r2, "r_128": r1, "difference": r2 - r1, "r_256_over_r_128": r2 / r1,
                         "verdict": "GROWS" if r2 - r1 >= LE_MARGIN - EPS else "NOT_SHOWN"}
        res["length_effect"] = le
    if arm == "D":
        r = per_len["128"]["ratios"]["pfsd27648"].get("ttft_vs_fork_dense")
        gbd = gb.get("128", {}).get("status")
        res["d043"] = {"ttft_ratio": r, "line": D043,
                       "outcome": None if r is None else "cleared" if r >= D043 - EPS else "triggered",
                       "quality_basis": ("quality rests on the bit identity of GATHER=2 with GATHER=1 (gate B PASS on this "
                                         "card) plus PF-NT and PF-CONF at 128k; no RULER on this card") if gbd == "PASS" else
                                        (f"inherited quality unavailable: gate B {gbd or 'missing'} on this card; the speed "
                                         "result stands, without a quality statement"),
                       "gate_b": gbd or "MISSING"}
    res["status"] = "OK"
    return res


def load_manifest(path: Path) -> dict[int, Path | None]:
    raw = json.loads(path.read_text())
    if not isinstance(raw, dict) or set(raw) != {"q38"} or not isinstance(raw["q38"], dict) or set(raw["q38"]) != {"1", "2"}:
        raise SystemExit(f"manifest {path}: must be exactly {{\"q38\": {{\"1\": <run dir|null>, \"2\": <run dir|null>}}}}")
    out: dict[int, Path | None] = {}
    for p in (1, 2):
        v = raw["q38"][str(p)]
        if v is None:
            out[p] = None
            continue
        if not isinstance(v, str):
            raise SystemExit(f"manifest {path}: pass {p} must be one run dir, got {v!r}")
        d = Path(v) if Path(v).is_absolute() else path.parent / v
        if not (d / "status.txt").is_file():
            raise SystemExit(f"manifest {path}: pass {p}: no status.txt in {d}")
        out[p] = d
    return out


def stats_tasks(s: dict, c: str, prompts: list[int], boot: int, tasks: list[int]) -> dict[str, object]:
    """pfn_decide.gate_stats over a subset of tasks (descriptive)"""
    rng = np.random.default_rng(0)
    n = len(prompts)
    idx = [rng.integers(0, n, (boot, n)) for _ in tasks]
    diff = [np.array([s[(t * 1000 + i, c)] - s[(t * 1000 + i, "dense")] for i in prompts]) for t in tasks]
    delta = 100 * float(np.mean([d.mean() for d in diff]))
    boots = 100 * np.mean([diff[k][idx[k]].mean(1) for k in range(len(tasks))], axis=0)
    return {"delta": delta, "ci95": [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))],
            "n_pairs": len(tasks) * n}


def amend4_applies(a: argparse.Namespace, p: int, d: Path) -> bool:
    p4, t4 = AMEND4_UNIT
    term = re.sub(r" 20[0-9-]+T[0-9:+.-]+$", "", terminal(d) or "")
    return (a.amend4 and p == p4 and bool(AMEND4_TERM.match(term)) and
            all((d / "tmp" / f"q38_p{p4}_t{t4}.{x}").is_file() for x in ("jsonl", "err", "check")))


def quality(a: argparse.Namespace, man: dict[int, Path | None], sc: Scorer) -> dict[str, object]:
    res: dict[str, object] = {"runs": {str(p): str(d) for p, d in man.items()}}
    a4 = {p: d is not None and amend4_applies(a, p, d) for p, d in man.items()}
    res["amendment_4"] = {"requested": bool(a.amend4), "applied": [p for p, v in a4.items() if v]}

    def udirs(p: int, t: int) -> list[Path]:
        """where a unit's files are read: ruler/, plus tmp/ for amendment 4's unit only"""
        return [man[p] / "ruler", man[p] / "tmp"] if a4[p] and (p, t) == AMEND4_UNIT else [man[p] / "ruler"]

    def urows(p: int, tasks: list[int]) -> tuple[list[dict], list[str]]:
        rows, probs = [], []
        for t in tasks:
            r, pr = cohort_rows(udirs(p, t), "q38", [(p, t)])
            rows += r; probs += pr
        return rows, probs

    for p, d in man.items():
        if d is None:
            continue
        lines = status_lines(d)
        miss = [x for x in QREQ if not has(lines, x) and not (a4[p] and x == "Q38 UNITS COMPLETE")]
        if miss or terminal(d) is None:
            res.update(decision="INVALID", reason=f"{d}: missing {miss}" if miss else f"{d}: no terminal line", outcome="NO_DECISION")
            return res
    rows, probs = [], []
    for p in (1, 2):
        if man[p] is None:
            probs.append(f"pass {p}: no run dir")
            continue
        r, pr = urows(p, list(range(13)))
        rows += r; probs += pr
        for t in range(13):
            e = next((d / f"q38_p{p}_t{t}.err" for d in udirs(p, t) if (d / f"q38_p{p}_t{t}.err").exists()),
                     man[p] / "ruler" / f"q38_p{p}_t{t}.err")
            if a4[p] and (p, t) == AMEND4_UNIT and e.exists():
                # the box's unit check (pfn_check task) also rejects a reused prefill graph; the recovered unit gets it here
                reused = [l for l in e.read_text().splitlines() if re.match(r"^prefill .*graphs reused", l)]
                if reused:
                    probs.append(f"pass {p} task {t}: {len(reused)} prefills reused a graph: {reused[0]}")
            c, _ = sel_counts(e)
            if e.exists() and not (len(c) == 1 and c[0][0] > 0 and c[0][1] == 0):
                probs.append(f"pass {p} task {t}: conversion count lines {c}")
    ids = {i for p in (1, 2) for t in range(13) for i in pass_ids(t, p)}
    cov, inv, ratios = unit_problems(rows, ids, QCFGS, "q38")
    probs += cov + inv
    res["pf_dense_prefill_median"] = float(np.median(ratios)) if ratios else None

    def unit_dense(p: int, tasks: list[int]) -> tuple[list[dict], list[str]]:
        if man[p] is None:
            return [], [f"pass {p}: no run dir"]
        rr, pp = urows(p, tasks)
        if not pp:
            cv, iv, _ = unit_problems(rr, {i for t in tasks for i in pass_ids(t, p)}, QCFGS, "q38")
            pp = cv + iv
        return (sorted((r for r in rr if r["config"] == "dense"), key=lambda r: r["id"]) if not pp else []), pp

    gates: dict[str, object] = {}
    for tag, p in (("B", 1), ("C", 2)):
        dense, pp = unit_dense(p, F_UNITS[1])
        gates[f"gate_f_{tag}"] = ({"status": "UNAVAILABLE", "reason": str(pp[:3])} if pp else gate_f([sc.kind("q38", r) for r in dense]))
        dense, pp = unit_dense(p, [D_UNIT[1]])
        gates[f"gate_d_{tag}"] = ({"status": "UNAVAILABLE", "score": None, "reason": str(pp[:3])} if pp
                                  else gate_d([sc.score("q38", r) for r in dense]))
    res["gates"] = gates
    res["gates_note"] = "registered: box B (pass 1); box C's are descriptive"
    res["late_gates"] = sorted(set(a.late))
    if probs:
        res.update(decision="INCONCLUSIVE", reason=f"cohort not exact: {probs[:3]}", outcome="NO_DECISION")
        return res
    deg = sorted((r["id"], r["config"]) for r in rows if degenerate(r))
    res["degenerate_rows"] = [f"{i}:{c}" for i, c in deg]
    if deg:
        res.update(decision="INCONCLUSIVE", reason=f"{len(deg)} degenerate rows", outcome="NO_DECISION")
        return res
    s = {(r["id"], r["config"]): sc.score("q38", r) for r in rows}
    all20, p1, p2 = list(range(20)), list(range(10)), list(range(10, 20))
    per_task = {c: {TASKS[t]: 100 * float(np.mean([s[(t * 1000 + i, c)] for i in all20])) for t in range(13)} for c in QCFGS}
    res["per_task"] = per_task
    res["mean"] = {c: float(np.mean(list(per_task[c].values()))) for c in QCFGS}
    res["per_task_delta"] = {c: {t: per_task[c][t] - per_task["dense"][t] for t in TASKS} for c in QCFGS if c != "dense"}
    res["empty_outputs"] = {c: sum(1 for r in rows if r["config"] == c and not r["out"]) for c in QCFGS}
    kinds = {(r["id"], r["config"]): sc.kind("q38", r) for r in rows}
    res["kinds"] = {c: {k: sum(1 for (i, cc), v in kinds.items() if cc == c and v == k) for k in KINDS} for c in QCFGS}
    stats = {c: gate_stats(s, c, all20, a.boot) for c in (C27, C55, PF, Q)}
    res["stats"] = stats
    res["stats_pass1"] = {c: gate_stats(s, c, p1, a.boot) for c in (C27, C55, PF, Q)}
    res["stats_pass2"] = {c: gate_stats(s, c, p2, a.boot) for c in (C27, C55, PF, Q)}
    res["stats_without_qa_1"] = {c: stats_tasks(s, c, all20, a.boot, [t for t in range(13) if TASKS[t] != "qa_1"])
                                 for c in (C27, C55, PF, Q)}
    if PFNT_DECISION.is_file():
        nt = json.loads(PFNT_DECISION.read_text()).get("models", {}).get("q38", {}).get("per_task", {}).get("dense")
        if nt:
            res["dense_vs_pf_nt_128k"] = {t: {"256k_q8_0": per_task["dense"][t], "128k_f16_pf_nt": nt[t]} for t in TASKS}
    gf, gd = gates["gate_f_B"], gates["gate_d_B"]
    if gf["status"] == "FAIL":
        res.update(decision="FORMAT_FAIL", reason=gf["reason"], outcome="FORMAT_FAIL")
        return res
    if gf["status"] != "PASS":
        res.update(decision="NO_DECISION", reason=f"gate F {gf['status']}: {gf['reason']}", outcome="NO_DECISION")
        return res
    if gd["status"] != "PASS":
        res.update(decision="UNINFORMATIVE", reason=f"gate D {gd['status']}: {gd['reason']}", outcome="UNINFORMATIVE")
        return res
    if res["mean"]["dense"] < DENSE_MIN - EPS:
        res.update(decision="UNINFORMATIVE", reason=f"dense 13-task mean {res['mean']['dense']:.2f} < {DENSE_MIN:g}",
                   outcome="UNINFORMATIVE")
        return res
    for c, oc in ((C27, "PASS_27648"), (C55, "PASS_55296"), (PF, "PASS_PF_ONLY")):
        if stats[c]["pass"]:
            st = stats[c]
            res.update(decision=oc, outcome=oc, reason=f"{c} delta {st['delta']:+.2f} [{st['ci95'][0]:+.2f}, {st['ci95'][1]:+.2f}] passes")
            return res
    res.update(decision="FAIL_ALL", outcome="FAIL_ALL", reason="no config passes the gate: " + "; ".join(
        f"{c} {stats[c]['delta']:+.2f} [{stats[c]['ci95'][0]:+.2f}, {stats[c]['ci95'][1]:+.2f}]" for c in (C27, C55, PF)))
    return res


CLAIM = {"PASS_27648": "pfsd27648", "PASS_55296": "pfsd55296", "PASS_PF_ONLY": "pf_only"}
FALLBACK_TEXT = ("the primary config pf+quest:27648 did not pass the gate at 256k (non-inferiority not shown; not a "
                 "demonstrated loss); the claim uses the registered fallback {c} with its own quoted ratio")


def gain(r: float) -> str:
    """the absolute 256k gain band of a quoted ratio (the length comparison is reported separately)"""
    return "HIGH" if r >= 1.8 - EPS else "MODERATE" if r >= 1.4 - EPS else "LOW"


def consequence(speed_a: dict | None, qual: dict | None) -> dict[str, object]:
    if not speed_a or speed_a.get("status") != "OK":
        return {"row": "NO_RATIO", "reason": "arm A not OK or not run"}
    rat = speed_a["lengths"]["256"]["ratios"]
    q = (qual or {}).get("outcome", "NOT_RUN")
    # the claim config comes from the quality outcome first; only then is its own ratio required (REV-032)
    cfg = CLAIM.get(q, "pfsd27648")
    # the length wording rests on that config's own R_256 - R_128 (REV-033); the primary's is reported alongside
    lea = speed_a.get("length_effect", {})
    le = lea.get(cfg, {})
    out: dict[str, object] = {"quality": q, "length_effect": le.get("verdict", "UNAVAILABLE"), "length_effect_config": cfg,
                              "length_effect_detail": le, "length_effect_primary": lea.get("pfsd27648", {})}
    r = rat[cfg].get("quoted")
    out["claim_config" if q in CLAIM else "speed_config"] = cfg
    if r is None:
        return out | {"row": "NO_RATIO", "reason": f"{cfg}: {rat[cfg].get('reason')}"}
    out.update(r_256=r, gain=gain(r))
    if q in CLAIM:
        out["row"] = f"{q}_{gain(r)}"
        if cfg != "pfsd27648":
            out["text"] = FALLBACK_TEXT.format(c=cfg)
    elif q == "FAIL_ALL":
        out.update(row="NO_QUALITY_CLAIM", text="no 256k quality claim; the speed is reported next to the gate that was not passed")
    else:
        out.update(row="SPEED_ONLY", text="256k is speed only; quality as measured at 128k (PF-NT f16 chat; PF-CONF q8_0 raw), the gap named")
    if out["gain"] == "LOW":
        out["program"] = "ends at the 128k write-up plus this result"
    out["length_wording"] = (f"allowed for {cfg}: the gain grows with context (R_256 - R_128 >= 0.3)" if out["length_effect"] == "GROWS"
                             else "not allowed: report R_256 and R_128 side by side only")
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--speed-a"); ap.add_argument("--speed-d"); ap.add_argument("--ruler-manifest")
    ap.add_argument("--book-meta", default=str(BOOK_META)); ap.add_argument("--ruler-meta", default=str(RULER_META))
    ap.add_argument("--late", action="append", default=[], choices=("F", "D"))
    ap.add_argument("--amend4", action="store_true")
    ap.add_argument("--score-field"); ap.add_argument("--text-field"); ap.add_argument("--boot", type=int, default=10000)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    meta = json.loads(Path(a.book_meta).read_text())
    out: dict[str, object] = {}
    for arm, d in (("A", a.speed_a), ("D", a.speed_d)):
        out[f"speed_{arm}"] = speed_arm(arm, Path(d), meta) if d else {"status": "NOT_RUN"}
    if a.ruler_manifest:
        out["quality"] = quality(a, load_manifest(Path(a.ruler_manifest)), Scorer(a.ruler_meta, a.score_field, a.text_field))
    else:
        out["quality"] = {"outcome": "NOT_RUN"}
    out["consequence"] = consequence(out["speed_A"] if a.speed_a else None, out["quality"])
    Path(a.out).write_text(json.dumps(out, indent=1))
    for arm in ("A", "D"):
        s = out[f"speed_{arm}"]
        if s.get("status") != "OK":
            print(f"SPEED {arm} {s.get('status')}: {s.get('reason', '')}")
            continue
        for L, v in s["lengths"].items():
            r = v["ratios"]
            print(f"SPEED {arm} {L}: best dense {[v['best_dense'][b].get('config') for b in ('1', '2')]}; "
                  + "; ".join(f"{c} {r[c]['quoted']:.3f}" if r[c].get("quoted") else f"{c} none" for c in SPARSE)
                  + f"; gate B {s['gate_b'][L]['status']}; gate N {v['gate_n']['status']}")
        if arm == "D":
            print(f"D-043 {s['d043']['outcome']} (TTFT ratio {s['d043']['ttft_ratio']})")
    q = out["quality"]
    print(f"QUALITY {q.get('outcome')}: {q.get('reason', '')}")
    if out["speed_A"].get("length_effect"):
        le = out["speed_A"]["length_effect"]["pfsd27648"]
        print("LENGTH EFFECT pfsd27648 " + (f"R_256 {le['r_256']:.3f} R_128 {le['r_128']:.3f} difference {le['difference']:+.3f} "
                                          f"-> {le['verdict']}" if "r_256" in le else f"UNAVAILABLE ({le['reason']})"))
    print(f"CONSEQUENCE {out['consequence']['row']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
