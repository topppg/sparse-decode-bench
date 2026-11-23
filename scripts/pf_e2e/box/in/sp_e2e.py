"""PORT-SP end to end: one long prompt plus N generated tokens through a fresh llama-server, timed at the client.

  sp_e2e.py run --i32 <file> --id <prompt id> --tag <kv_config> --block <n> --out <jsonl> --port <p> --server-pid <pid>
                --expect-model <gguf name> --expect-ctx <n> [--warmup] [--reps 2] [--n-predict 1000] [--smoke] [--expect-draft]
  sp_e2e.py summary <jsonl> --expect-tags a,b,... --reps 4

run: waits for /health while the server process lives, checks /props (model file, context, one slot), then
  --smoke: the first 2048 prompt tokens with 64 to generate, once (exit 3 with --expect-draft if nothing was drafted);
  otherwise: an untimed full-prompt warmup (64 tokens) with --warmup, then --reps timed requests of n_predict tokens,
  greedy, ignore_eos, no prompt cache. Each timed row is validated: every prompt token processed, exactly n_predict output ids,
  timings present, and with --expect-draft some tokens drafted. Exit 0 only if every timed row is valid.
summary: per tag exactly --reps valid timed rows with unique (block, rep) keys; medians of per-repetition wall times and rates;
  MTP acceptance; output identity against the same-KV dense tag. "status" is COMPLETE only if every expected tag is complete.
Standard library only.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import struct
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path


def prompt_ids(path: Path, pid: int) -> list[int]:
    raw = path.read_bytes()
    (n,) = struct.unpack_from("<i", raw, 0)
    off = 4
    for _ in range(n):
        i, _max_new, ln = struct.unpack_from("<3i", raw, off)
        if i == pid:
            return list(struct.unpack_from(f"<{ln}i", raw, off + 12))
        off += 12 + 4 * ln
    raise SystemExit(f"prompt {pid} not in {path}")


def get_json(port: int, path: str) -> dict[str, object]:
    with urllib.request.urlopen(f"http://127.0.0.1:{port}{path}", timeout=10) as r:
        return json.loads(r.read())


def wait_health(port: int, timeout_s: float, pid: int) -> None:
    t_end = time.time() + timeout_s
    while time.time() < t_end:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            raise SystemExit(f"server pid {pid} exited before it was healthy")
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=5) as r:
                if r.status == 200:
                    return
        except (urllib.error.URLError, ConnectionError, TimeoutError):
            pass
        time.sleep(2)
    raise SystemExit(f"server on {port} not healthy after {timeout_s} s")


def check_identity(port: int, model: str, ctx: int) -> None:
    p = get_json(port, "/props")
    path = str(p.get("model_path", ""))
    dgs = p.get("default_generation_settings", {})
    n_ctx = dgs.get("n_ctx") if isinstance(dgs, dict) else None
    if Path(path).name != model or n_ctx != ctx or p.get("total_slots") != 1:
        raise SystemExit(f"/props mismatch: model_path={path} n_ctx={n_ctx} total_slots={p.get('total_slots')}")


def complete(port: int, ids: list[int], n_predict: int) -> tuple[float, dict[str, object]]:
    body = {"prompt": ids, "n_predict": n_predict, "ignore_eos": True, "temperature": 0.0, "cache_prompt": False,
            "return_tokens": True, "stream": False}
    req = urllib.request.Request(f"http://127.0.0.1:{port}/completion", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=1800) as r:
        res = json.loads(r.read())
    return time.perf_counter() - t0, res


def row_of(a: argparse.Namespace, kind: str, rep: int, ids: list[int], n_predict: int, wall: float,
           res: dict[str, object]) -> dict[str, object]:
    t = res.get("timings", {})
    assert isinstance(t, dict)
    toks = res.get("tokens") or []
    why = []
    if t.get("prompt_n") != len(ids):
        why.append(f"prompt_n {t.get('prompt_n')} != {len(ids)}")
    if t.get("predicted_n") != n_predict or len(toks) != n_predict:
        why.append(f"predicted_n {t.get('predicted_n')} / ids {len(toks)} != {n_predict}")
    if not t.get("prompt_ms") or not t.get("predicted_ms"):
        why.append("timings missing")
    if a.expect_draft and not t.get("draft_n"):
        why.append("no tokens drafted")
    return {"tag": a.tag, "block": a.block, "kind": kind, "rep": rep, "id": a.id, "n_prompt": len(ids), "wall_s": round(wall, 3),
            "prompt_n": t.get("prompt_n"), "prompt_ms": t.get("prompt_ms"), "predicted_n": t.get("predicted_n"),
            "predicted_ms": t.get("predicted_ms"), "draft_n": t.get("draft_n", 0), "draft_n_accepted": t.get("draft_n_accepted", 0),
            "valid": not why, "why": why, "tokens": toks}


def run(a: argparse.Namespace) -> None:
    ids = prompt_ids(Path(a.i32), a.id)
    wait_health(a.port, a.health_timeout, a.server_pid)
    check_identity(a.port, a.expect_model, a.expect_ctx)
    plan: list[tuple[str, list[int], int]] = []
    if a.smoke:
        plan.append(("smoke", ids[:2048], 64))
    else:
        if a.warmup:
            plan.append(("warmup", ids, 64))
        plan += [("timed", ids, a.n_predict)] * a.reps
    ok = True
    with open(a.out, "a") as f:
        for rep, (kind, p, n) in enumerate(plan):
            wall, res = complete(a.port, p, n)
            row = row_of(a, kind, rep, p, n, wall, res)
            if kind != "warmup":
                ok &= bool(row["valid"])
            f.write(json.dumps(row) + "\n")
            f.flush()
            print(json.dumps({k: v for k, v in row.items() if k != "tokens"}))
    sys.exit(0 if ok else 3)


def prefix_eq(x: list[int], y: list[int]) -> int:
    n = 0
    for p, q in zip(x, y):
        if p != q:
            break
        n += 1
    return n


def summary(path: Path, tags: list[str], reps: int) -> dict[str, object]:
    rows = [json.loads(x) for x in path.read_text().splitlines() if x.strip()] if path.exists() else []
    timed = [r for r in rows if r.get("kind") == "timed"]
    out: dict[str, object] = {"expected_tags": tags, "reps": reps}
    complete_tags = []
    for t in tags:
        rs = [r for r in timed if r["tag"] == t]
        keys = [(r["block"], r["rep"]) for r in rs]
        valid = [r for r in rs if r["valid"]]
        s: dict[str, object] = {"rows": len(rs), "valid": len(valid), "duplicate_keys": len(keys) - len(set(keys)),
                                "invalid_reasons": sorted({w for r in rs for w in r["why"]})}
        s["complete"] = len(rs) == reps and len(valid) == reps and s["duplicate_keys"] == 0
        if valid:
            s["wall_s_median"] = round(statistics.median(r["wall_s"] for r in valid), 3)
            s["wall_s_all"] = [r["wall_s"] for r in valid]
            s["prefill_tps_median"] = round(statistics.median(r["prompt_n"] / r["prompt_ms"] * 1e3 for r in valid), 1)
            s["decode_tps_median"] = round(statistics.median(r["predicted_n"] / r["predicted_ms"] * 1e3 for r in valid), 2)
            dn = sum(int(r["draft_n"] or 0) for r in valid)
            s["acceptance"] = round(sum(int(r["draft_n_accepted"] or 0) for r in valid) / dn, 3) if dn else None
            s["reps_identical"] = all(r["tokens"] == valid[0]["tokens"] for r in valid)
        if s["complete"]:
            complete_tags.append(t)
        out[t] = s
    # ratios and identity only between complete tags of the same KV type
    for t in complete_tags:
        kv = t.split("_")[0]
        d = f"{kv}_dense"
        if t == d or d not in complete_tags:
            continue
        s, ds = out[t], out[d]
        assert isinstance(s, dict) and isinstance(ds, dict)
        s["wall_dense_over_this"] = round(ds["wall_s_median"] / s["wall_s_median"], 3)
        s["decode_this_over_dense"] = round(s["decode_tps_median"] / ds["decode_tps_median"], 3)
        x = next(r for r in timed if r["tag"] == t and r["valid"])["tokens"]
        y = next(r for r in timed if r["tag"] == d and r["valid"])["tokens"]
        s["prefix_equal_to_dense"] = prefix_eq(x, y)
        if "mtp_dense" in t and s["prefix_equal_to_dense"] < 50:
            s["flag"] = "MTP dense output leaves dense greedy output within 50 tokens"
    mtp_d, mtp_s = [f"{kv}_mtp_dense" for kv in ("q8", "f16")], [f"{kv}_mtp_sparse" for kv in ("q8", "f16")]
    for a_, b_ in zip(mtp_s, mtp_d):
        if a_ in complete_tags and b_ in complete_tags:
            sa, sb = out[a_], out[b_]
            assert isinstance(sa, dict) and isinstance(sb, dict)
            sa["wall_mtp_dense_over_this"] = round(sb["wall_s_median"] / sa["wall_s_median"], 3)
    out["status"] = "COMPLETE" if len(complete_tags) == len(tags) else "INCOMPLETE"
    out["incomplete_tags"] = [t for t in tags if t not in complete_tags]
    return out


def main() -> None:
    if len(sys.argv) > 1 and sys.argv[1] == "summary":
        ap = argparse.ArgumentParser()
        ap.add_argument("cmd")
        ap.add_argument("jsonl")
        ap.add_argument("--expect-tags", required=True)
        ap.add_argument("--reps", type=int, default=4)
        a = ap.parse_args()
        res = summary(Path(a.jsonl), a.expect_tags.split(","), a.reps)
        print(json.dumps(res, indent=1))
        sys.exit(0 if res["status"] == "COMPLETE" else 1)
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["run"])
    ap.add_argument("--i32", required=True)
    ap.add_argument("--id", type=int, required=True)
    ap.add_argument("--tag", required=True, help="<kv>_<config>: q8_dense, q8_sparse, q8_mtp_dense, q8_mtp_sparse, f16_...")
    ap.add_argument("--block", type=int, required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--port", type=int, required=True)
    ap.add_argument("--server-pid", type=int, required=True)
    ap.add_argument("--expect-model", required=True)
    ap.add_argument("--expect-ctx", type=int, required=True)
    ap.add_argument("--n-predict", type=int, default=1000)
    ap.add_argument("--reps", type=int, default=2)
    ap.add_argument("--health-timeout", type=float, default=900)
    ap.add_argument("--warmup", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--expect-draft", action="store_true")
    run(ap.parse_args())


if __name__ == "__main__":
    main()
