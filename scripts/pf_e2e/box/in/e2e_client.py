"""PF-E2E client: one long prompt plus N generated tokens through a fresh llama-server, timed at the client. PORT-SP's
sp_e2e.py (copied unchanged next to this file) supplies the prompt reader, the health and /props checks, the request and
the row validation; this file changes only the warmup.

  e2e_client.py --i32 <gen file> --id <prompt id> --tag <arm>_<len>_<config> --block <0|1|2> --out <jsonl> --port <p>
                --server-pid <pid> --expect-model <gguf name> --expect-ctx <n> [--reps 2] [--n-predict 1000]
                [--smoke-tokens 512] [--expect-draft]

Plan per server: one untimed smoke request (the first --smoke-tokens prompt tokens, 64 generated; with 512 tokens one
ubatch at position 0, where the sparse prefill has no selectable page, so the page-select conversion count stays 0), then
--reps timed requests of the full prompt: n_predict tokens, greedy, ignore_eos, no prompt cache (sp_e2e.complete). Every
row is written (kind smoke|timed) with sp_e2e.row_of's fields and validity (every prompt token processed, exactly n_predict
output ids, timings present, with --expect-draft some tokens drafted). Exit 0 only if every timed row is valid, 3 otherwise.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from sp_e2e import check_identity, complete, prompt_ids, row_of, wait_health  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--i32", required=True); ap.add_argument("--id", type=int, required=True)
    ap.add_argument("--tag", required=True); ap.add_argument("--block", type=int, required=True)
    ap.add_argument("--out", required=True); ap.add_argument("--port", type=int, required=True)
    ap.add_argument("--server-pid", type=int, required=True); ap.add_argument("--expect-model", required=True)
    ap.add_argument("--expect-ctx", type=int, required=True); ap.add_argument("--n-predict", type=int, default=1000)
    ap.add_argument("--reps", type=int, default=2); ap.add_argument("--smoke-tokens", type=int, default=512)
    ap.add_argument("--health-timeout", type=float, default=900); ap.add_argument("--expect-draft", action="store_true")
    a = ap.parse_args()
    ids = prompt_ids(Path(a.i32), a.id)
    wait_health(a.port, a.health_timeout, a.server_pid)
    check_identity(a.port, a.expect_model, a.expect_ctx)
    plan = [("smoke", ids[:a.smoke_tokens], 64)] + [("timed", ids, a.n_predict)] * a.reps
    ok = True
    with open(a.out, "a") as f:
        for rep, (kind, p, n) in enumerate(plan):
            wall, res = complete(a.port, p, n)
            row = row_of(a, kind, rep if kind == "smoke" else rep - 1, p, n, wall, res)
            if kind == "timed":
                ok &= bool(row["valid"])
            f.write(json.dumps(row) + "\n")
            f.flush()
            print(json.dumps({k: v for k, v in row.items() if k != "tokens"}))
    return 0 if ok else 3


if __name__ == "__main__":
    sys.exit(main())
