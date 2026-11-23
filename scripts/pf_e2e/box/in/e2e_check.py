"""PF-E2E checks on the box and on the Mac (standard library only; reuses PF-EMUL's pfe_check, copied unchanged).

  e2e_check.py cfg <config>                      shell assignments for one config of the matrix (C_BIN C_KV C_UB C_MTP C_PF
                                                 C_SD C_ROLE)
  e2e_check.py matrix <pfe_matrix.tsv> --arm A|D the arm's rows "len block config" in file order, after checking the whole
        file: known configs; per (arm, length) block 2 is block 1 reversed and holds the eight registered configs; arm D
        has length 128 only; informative rows (block i) only after both blocks of every length of that arm
  e2e_check.py prompt <pfe_book_q38.i32> --id 128|256 --max-new N --out <gen.i32> [--raw <tokens.i32>]
        one prompt in --gen format with the given max_new (and its raw int32 tokens for tail mode); prints its length
  e2e_check.py sel <log>                          the page-select conversion counts printed at exit
        ("ggml_cuda: page select K/V conversion: N selective, M whole"): prints "N M", "none" if no line, exit 1 if >1 line
  e2e_check.py ref <rows.jsonl> --err <stderr> --len N --configs C
        the sparse-tail --gen reference of one prompt: every config once, len N, pf counts as pfe_check expects for 16 layers
        and 4 KV heads, sparse engaged on all 16 layers for pf configs, exactly one exit count line with selective > 0 and
        whole = 0 when a pf config ran; prints the selective count of the process (one pf prefill) and exits 0, else 1
  e2e_check.py srv <rows.jsonl> --log <server log> --cfg C --reps R --n-predict P --len N [--ref-sel S]
        one (length, block, config) through llama-server: OOM (an out-of-memory message in the server log, or a failed
        start with one) | INVALID <why> | OK <summary>. OK needs R valid timed rows (sp_e2e's row rules: prompt_n = N, exactly
        P output ids, timings present; MTP configs: draft_n > 0) and the conversion count rule: fork configs with a sparse
        prefill and q8_0 KV print "R*S selective, 0 whole" at exit (S = the reference's count for the same prompt; the 512-token smoke
        request selects nothing); every other fork config (dense, or f16 KV, which needs no conversion: the counter counts
        q8_0 conversions only) prints no count line or "0 selective, 0 whole". Rows are shape-checked first (row_problem:
        objects, valid, int token lists, finite positive wall_s, prompt_ms and predicted_ms, MTP draft counts):
        malformed rows are INVALID, never an exception
  e2e_check.py gate-a <rows.jsonl>                gate A (RULER boxes): dense and quest:262144:f3 greedy outputs identical
  e2e_check.py ruler-sel <stderr>                 a RULER unit's process printed exactly one count line, selective > 0,
                                                  whole = 0 (the GATHER=2 selective conversion engaged)
"""
from __future__ import annotations

import argparse
import json
import math
import re
import struct
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pfe_check import check_rows, read_rows  # noqa: E402

LAYERS, KV_HEADS = 16, 4                       # Qwen3.8-27B full-attention layers and KV heads
PF = "pf:0.1:512:d512"
# role: dense = a best-dense candidate; sparse = a quoted sparse config; info = informative (one block)
CONFIGS: dict[str, dict[str, object]] = {
    "fork_dense":        {"bin": "fork", "kv": "q8_0", "ub": 512,  "mtp": 0, "pf": 0, "sd": 0,     "role": "dense"},
    "fork_dense_ub2048": {"bin": "fork", "kv": "q8_0", "ub": 2048, "mtp": 0, "pf": 0, "sd": 0,     "role": "dense"},
    "fork_mtp":          {"bin": "fork", "kv": "q8_0", "ub": 512,  "mtp": 1, "pf": 0, "sd": 0,     "role": "dense"},
    "up_dense":          {"bin": "up",   "kv": "q8_0", "ub": 512,  "mtp": 0, "pf": 0, "sd": 0,     "role": "dense"},
    "up_mtp":            {"bin": "up",   "kv": "q8_0", "ub": 512,  "mtp": 1, "pf": 0, "sd": 0,     "role": "dense"},
    "pfsd27648":         {"bin": "fork", "kv": "q8_0", "ub": 512,  "mtp": 0, "pf": 1, "sd": 27648, "role": "sparse"},
    "pfsd55296":         {"bin": "fork", "kv": "q8_0", "ub": 512,  "mtp": 0, "pf": 1, "sd": 55296, "role": "sparse"},
    "pf_only":           {"bin": "fork", "kv": "q8_0", "ub": 512,  "mtp": 0, "pf": 1, "sd": 0,     "role": "sparse"},
    "sd27648":           {"bin": "fork", "kv": "q8_0", "ub": 512,  "mtp": 0, "pf": 0, "sd": 27648, "role": "info"},
    "fork_dense_f16":    {"bin": "fork", "kv": "f16",  "ub": 512,  "mtp": 0, "pf": 0, "sd": 0,     "role": "info"},
    "fork_mtp_f16":      {"bin": "fork", "kv": "f16",  "ub": 512,  "mtp": 1, "pf": 0, "sd": 0,     "role": "info"},
    "pfsd27648_f16":     {"bin": "fork", "kv": "f16",  "ub": 512,  "mtp": 0, "pf": 1, "sd": 27648, "role": "info"},
}
BLOCK_CONFIGS = [c for c, v in CONFIGS.items() if v["role"] in ("dense", "sparse")]
ARMS = {"A": (256, 128), "D": (128,)}
SEL = re.compile(r"page select K/V conversion: (\d+) selective, (\d+) whole")
OOM = re.compile(r"out of memory|cudaMalloc failed|failed to allocate|unable to allocate|ggml_backend_cuda_buffer_type_alloc_buffer",
                 re.IGNORECASE)


def cmd_cfg(a: argparse.Namespace) -> int:
    if a.config not in CONFIGS:
        print(f"echo unknown config {a.config}; false")
        return 1
    print(" ".join(f"C_{k.upper()}={v}" for k, v in CONFIGS[a.config].items()))
    return 0


def read_matrix(path: Path) -> tuple[list[tuple[str, int, str, str]], list[str]]:
    rows, probs = [], []
    for i, line in enumerate(path.read_text().splitlines()):
        if not line.strip() or line.startswith("#"):
            continue
        f = line.split("\t")
        if len(f) != 4 or f[0] not in ARMS or f[1] not in ("128", "256") or f[2] not in ("1", "2", "i") or f[3] not in CONFIGS:
            probs.append(f"line {i + 1}: bad row {line!r}")
            continue
        rows.append((f[0], int(f[1]), f[2], f[3]))
    for arm, lens in ARMS.items():
        ar = [r for r in rows if r[0] == arm]
        if sorted({r[1] for r in ar}) != sorted(lens):
            probs.append(f"arm {arm}: lengths {sorted({r[1] for r in ar})}, expected {sorted(lens)}")
        for ln in lens:
            b1 = [r[3] for r in ar if r[1] == ln and r[2] == "1"]
            b2 = [r[3] for r in ar if r[1] == ln and r[2] == "2"]
            if sorted(b1) != sorted(BLOCK_CONFIGS) or b2 != b1[::-1]:
                probs.append(f"arm {arm} length {ln}: block 1 {b1} must hold the registered configs once, block 2 its reverse {b2}")
            if any(CONFIGS[r[3]]["role"] == "info" for r in ar if r[1] == ln and r[2] != "i"):
                probs.append(f"arm {arm} length {ln}: an informative config in a timed block")
            if any(CONFIGS[r[3]]["role"] != "info" for r in ar if r[1] == ln and r[2] == "i"):
                probs.append(f"arm {arm} length {ln}: a block config in the informative block")
        last_block = max((i for i, r in enumerate(ar) if r[2] != "i"), default=-1)
        if any(r[2] == "i" for r in ar[:last_block + 1]):
            probs.append(f"arm {arm}: informative rows must come after every timed block")
        seen = set()
        for r in ar:
            if r[1:] in seen:
                probs.append(f"arm {arm}: duplicated row {r}")
            seen.add(r[1:])
    return rows, probs


def cmd_matrix(a: argparse.Namespace) -> int:
    rows, probs = read_matrix(Path(a.path))
    if probs:
        print("MATRIX INVALID " + "; ".join(probs[:5]))
        return 1
    for arm, ln, b, c in rows:
        if arm == a.arm:
            print(ln, b, c)
    return 0


def records(path: Path) -> list[tuple[int, int, list[int]]]:
    raw = path.read_bytes()
    (n,) = struct.unpack_from("<i", raw, 0)
    off, out = 4, []
    for _ in range(n):
        pid, mx, ln = struct.unpack_from("<3i", raw, off)
        out.append((pid, mx, list(struct.unpack_from(f"<{ln}i", raw, off + 12))))
        off += 12 + 4 * ln
    if off != len(raw):
        raise SystemExit(f"{path}: {len(raw) - off} trailing bytes")
    return out


def cmd_prompt(a: argparse.Namespace) -> int:
    got = [r for r in records(Path(a.path)) if r[0] == a.id]
    if len(got) != 1:
        print(f"prompt {a.id}: {len(got)} records")
        return 1
    pid, _, ids = got[0]
    Path(a.out).write_bytes(struct.pack("<4i", 1, pid, a.max_new, len(ids)) + struct.pack(f"<{len(ids)}i", *ids))
    if a.raw:
        Path(a.raw).write_bytes(struct.pack(f"<{len(ids)}i", *ids))
    print(len(ids))
    return 0


def sel_counts(path: Path) -> tuple[list[tuple[int, int]], str]:
    text = path.read_text(errors="replace") if path.exists() else ""
    return [(int(m[1]), int(m[2])) for m in SEL.finditer(text)], text


def cmd_sel(a: argparse.Namespace) -> int:
    c, _ = sel_counts(Path(a.log))
    if len(c) > 1:
        print(f"{len(c)} count lines")
        return 1
    print("none" if not c else f"{c[0][0]} {c[0][1]}")
    return 0


def ref_check(rows_path: Path, err: Path, n: int, cfgs: list[str]) -> tuple[int | None, list[str], list[dict]]:
    """(selective count of the reference process, problems, rows) of a sparse-tail --gen reference of one prompt"""
    rows, bad = read_rows(rows_path) if rows_path.exists() else ([], [f"no file {rows_path}"])
    probs = list(bad)
    for r in rows:
        if r.get("len") != n:
            probs.append(f"({r.get('id')}, {r.get('config')}): len {r.get('len')} != {n}")
    ids = {r.get("id") for r in rows} or {None}
    if len(ids) != 1:
        probs.append(f"{len(ids)} prompt ids, expected one")
    cov, inv = check_rows(rows, ids, cfgs, {"layers": LAYERS, "kv_heads": KV_HEADS})
    probs += cov + inv
    for r in rows:
        pf = r.get("pf") if isinstance(r.get("pf"), dict) else {}
        if str(r.get("config", "")).startswith("pf:") and not (pf.get("sparse", 0) > 0 and pf.get("layers") == LAYERS):
            probs.append(f"{r.get('config')}: sparse prefill did not run: {r.get('pf')}")
    c, _ = sel_counts(err)
    has_pf = any(x.startswith("pf:") for x in cfgs)
    if has_pf and not (len(c) == 1 and c[0][0] > 0 and c[0][1] == 0):
        probs.append(f"conversion count lines {c}: expected one with selective > 0 and whole = 0")
    return (c[0][0] if has_pf and len(c) == 1 else 0 if not has_pf else None), probs, rows


def cmd_ref(a: argparse.Namespace) -> int:
    sel, probs, _ = ref_check(Path(a.rows), Path(a.err), a.len, a.configs.split(","))
    if probs:
        print("REF INVALID " + "; ".join(str(p) for p in probs[:5]))
        return 1
    print(sel)
    return 0


def pos_num(x: object) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x) and x > 0


def row_problem(r: object, n: int, n_predict: int, mtp: bool) -> str | None:
    """why one timed row is unusable (malformed, invalid, wrong counts, non-finite or non-positive timings), or None;
    checked before any arithmetic, so no row content can raise"""
    if not isinstance(r, dict):
        return f"row is a {type(r).__name__}, not an object"
    rep = r.get("rep")
    if r.get("valid") is not True:
        return f"rep {rep} invalid: {r.get('why')}"
    toks = r.get("tokens")
    if not (isinstance(toks, list) and all(isinstance(t, int) and not isinstance(t, bool) for t in toks)):
        return f"rep {rep}: tokens not a list of ints"
    if r.get("prompt_n") != n or r.get("predicted_n") != n_predict or len(toks) != n_predict:
        return f"rep {rep}: prompt_n {r.get('prompt_n')} predicted_n {r.get('predicted_n')} tokens {len(toks)}"
    for f in ("wall_s", "prompt_ms", "predicted_ms"):
        if not pos_num(r.get(f)):
            return f"rep {rep}: {f} {r.get(f)!r:.20} is not a finite positive number"
    dn, da = r.get("draft_n", 0), r.get("draft_n_accepted", 0)
    ok_int = lambda x: isinstance(x, int) and not isinstance(x, bool) and x >= 0
    if not (ok_int(dn) and ok_int(da) and da <= dn) or (mtp and dn == 0):
        return f"rep {rep}: draft counts {dn!r:.12} / {da!r:.12}" + (" (MTP drafted nothing)" if mtp and dn == 0 else "")
    return None


def srv_verdict(rows_path: Path, log: Path, cfg: str, reps: int, n_predict: int, n: int, ref_sel: int | None) -> str:
    sp = CONFIGS[cfg]
    c, text = sel_counts(log)
    rows, bad = read_rows(rows_path) if rows_path.exists() else ([], [])
    objs = [r for r in rows if isinstance(r, dict)]
    timed = [r for r in objs if r.get("kind") == "timed"]
    probs = [row_problem(r, n, n_predict, bool(sp["mtp"])) for r in timed]
    if OOM.search(text) and sum(p is None for p in probs) < reps:
        return "OOM " + OOM.search(text)[0]
    why = list(bad) + [f"{len(rows) - len(objs)} rows are not JSON objects"] * (len(rows) != len(objs))
    if len(timed) != reps:
        why.append(f"{len(timed)} timed rows, expected {reps}")
    why += [p for p in probs if p]
    if sp["bin"] == "fork":
        if len(c) > 1:
            why.append(f"{len(c)} conversion count lines")
        elif sp["pf"] and sp["kv"] == "q8_0":
            want = None if ref_sel is None else reps * ref_sel
            if ref_sel is None or ref_sel <= 0:
                why.append("no reference conversion count")
            elif not c or c[0] != (want, 0):
                why.append(f"conversion count {c[0] if c else 'none'} != ({want}, 0) ({reps} x the reference's {ref_sel})")
        elif c and c[0] != (0, 0):
            # f16 KV needs no conversion (the counter counts q8_0 conversions only); a dense config selects nothing
            why.append(f"conversion count {c[0]} where none is expected ({'f16 KV' if sp['pf'] else 'no sparse prefill'})")
    if why:
        return "INVALID " + "; ".join(str(w) for w in why[:4])
    wall = sorted(r["wall_s"] for r in timed)
    ttft = sorted(r["prompt_ms"] / 1e3 for r in timed)
    tps = sorted(r["predicted_n"] / r["predicted_ms"] * 1e3 for r in timed)
    acc = (sum(r.get("draft_n_accepted", 0) for r in timed) / sum(r.get("draft_n", 0) for r in timed)) if sp["mtp"] else None
    return (f"OK wall {' '.join(f'{x:.2f}' for x in wall)} ttft {' '.join(f'{x:.2f}' for x in ttft)} "
            f"tok/s {' '.join(f'{x:.2f}' for x in tps)}" + (f" acceptance {acc:.3f}" if sp["mtp"] else "")
            + (f" sel {c[0][0]}" if c else ""))


def cmd_srv(a: argparse.Namespace) -> int:
    v = srv_verdict(Path(a.rows), Path(a.log), a.cfg, a.reps, a.n_predict, a.len, a.ref_sel)
    print(v)
    return 0 if v.startswith("OK") else 3 if v.startswith("OOM") else 2


def cmd_gate_a(a: argparse.Namespace) -> int:
    rows, bad = read_rows(Path(a.rows))
    by = {r.get("config"): r for r in rows}
    if bad or set(by) != {"dense", "quest:262144:f3"} or len(rows) != 2:
        print(f"FAIL rows {sorted(by)} ({len(rows)}), bad {bad}")
        return 1
    x, y = by["dense"].get("out"), by["quest:262144:f3"].get("out")
    if not x or x != y:
        first = next((i for i, (u, v) in enumerate(zip(x or [], y or [])) if u != v), min(len(x or []), len(y or [])))
        print(f"FAIL outputs differ at token {first} of {len(x or [])}")
        return 1
    print(f"PASS {len(x)} greedy tokens identical (id {by['dense'].get('id')}, len {by['dense'].get('len')})")
    return 0


def cmd_ruler_sel(a: argparse.Namespace) -> int:
    c, _ = sel_counts(Path(a.err))
    ok = len(c) == 1 and c[0][0] > 0 and c[0][1] == 0
    print(("SEL OK " if ok else "SEL INVALID ") + (f"{c[0][0]} selective, {c[0][1]} whole" if len(c) == 1 else f"{len(c)} count lines"))
    return 0 if ok else 2


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("cfg"); s.add_argument("config")
    s = sub.add_parser("matrix"); s.add_argument("path"); s.add_argument("--arm", choices=ARMS, required=True)
    s = sub.add_parser("prompt"); s.add_argument("path"); s.add_argument("--id", type=int, required=True)
    s.add_argument("--max-new", type=int, required=True); s.add_argument("--out", required=True); s.add_argument("--raw")
    s = sub.add_parser("sel"); s.add_argument("log")
    s = sub.add_parser("ref"); s.add_argument("rows"); s.add_argument("--err", required=True)
    s.add_argument("--len", type=int, required=True); s.add_argument("--configs", required=True)
    s = sub.add_parser("srv"); s.add_argument("rows"); s.add_argument("--log", required=True); s.add_argument("--cfg", choices=CONFIGS, required=True)
    s.add_argument("--reps", type=int, required=True); s.add_argument("--n-predict", type=int, required=True)
    s.add_argument("--len", type=int, required=True); s.add_argument("--ref-sel", type=int)
    s = sub.add_parser("gate-a"); s.add_argument("rows")
    s = sub.add_parser("ruler-sel"); s.add_argument("err")
    a = ap.parse_args()
    sys.exit({"cfg": cmd_cfg, "matrix": cmd_matrix, "prompt": cmd_prompt, "sel": cmd_sel, "ref": cmd_ref, "srv": cmd_srv,
              "gate-a": cmd_gate_a, "ruler-sel": cmd_ruler_sel}[a.cmd](a))


if __name__ == "__main__":
    main()
