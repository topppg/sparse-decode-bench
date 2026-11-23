"""Print the README's sparse-prefill numbers from the PF-E2E and PF-NT decision files and the rebase development run.

  python3 e2e_summary.py --decision <decision.json> --pf-nt <PF-NT decision.json> --dev <results/cuda_5090_pfrebase_dev>
"""
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

C27 = "pf:0.1:512:d512+quest:27648:f3"
C55 = "pf:0.1:512:d512+quest:55296:f3"
PF = "pf:0.1:512:d512"
Q = "quest:27648:f3"


def speed(d: dict) -> None:
    for arm, card in (("speed_A", "RTX 5090"), ("speed_D", "RTX 4090")):
        s = d[arm]
        for L, v in s["lengths"].items():
            r = v["ratios"]["pfsd27648"]
            b = min(r["per_block"], key=r["per_block"].get)  # the quoted ratio is the lower block's
            bd = v["best_dense"][b]
            sp = s["cells"][f"{L} b{b} pfsd27648"]
            print(f"{card} {L}k: best dense {bd['config']} {bd['wall']:.1f} s -> pfsd27648 {sp['wall']:.1f} s = {r['quoted']:.2f}x "
                  f"(block {b}); TTFT {bd['ttft']:.1f} -> {sp['ttft']:.1f} s; decode {bd['tps']:.1f} -> {sp['tps']:.1f} tok/s; "
                  f"pfsd55296 {v['ratios']['pfsd55296']['quoted']:.2f}x, pf_only {v['ratios']['pf_only']['quoted']:.2f}x; "
                  f"vs fork_dense {r['wall_vs_fork_dense']:.2f}x; gate B {s['gate_b'][L]['status']}")
            other = "2" if b == "1" else "1"
            print(f"  other block {other}: best dense {v['best_dense'][other]['config']} {v['best_dense'][other]['wall']:.2f} s; "
                  f"max |drift| {100 * max(abs(x['b2_over_b1_minus_1']) for x in v['drift'].values()):.2f}%")
            if L == "256":
                fd = s["cells"]["256 b1 fork_dense"]
                print(f"  fork_dense block 1: {fd['wall']:.1f} s, TTFT {fd['ttft']:.1f} s")
                pk = s["peak_memory_mib"]
                print("  peak memory.used, block 1 (MiB / GiB): " + ", ".join(
                    f"{c} {pk[f'256_b1_{c}']} / {pk[f'256_b1_{c}'] / 1024:.1f}" for c in ("up_dense", "fork_dense_ub2048", "up_mtp", "pfsd27648")))
    le = d["speed_A"]["length_effect"]["pfsd27648"]
    print(f"length effect pfsd27648: R_256 - R_128 = {le['difference']:+.2f} -> {le['verdict']}")
    print(f"RTX 4090 TTFT ratio fork_dense / pfsd27648 (registered stopping rule, line 1.30): {d['speed_D']['d043']['ttft_ratio']:.2f}")


def quality(d: dict, nt: dict) -> None:
    q = d["quality"]
    print(f"RULER 256k: outcome {q['outcome']} (amendment 4 applied to pass {q['amendment_4']['applied']}); dense mean {q['mean']['dense']:.2f}")
    for c in (Q, PF, C27, C55):
        st = q["stats"][c]
        print(f"  {c:32s} {st['delta']:+.2f} [{st['ci95'][0]:+.2f}, {st['ci95'][1]:+.2f}]")
    p1, wo = q["stats_pass1"][C27], q["stats_without_qa_1"][C27]
    print(f"  primary, pass 1 alone {p1['delta']:+.2f} [{p1['ci95'][0]:+.2f}, {p1['ci95'][1]:+.2f}]; "
          f"without qa_1 {wo['delta']:+.2f} [{wo['ci95'][0]:+.2f}, {wo['ci95'][1]:+.2f}]")
    print("  primary per task: " + ", ".join(f"{t} {x:+.2f}" for t, x in q["per_task_delta"][C27].items() if abs(x) > 1e-9))
    for m, name in (("b38", "BF16"), ("q38", "UD-Q4_K_M")):
        s = nt["models"][m]
        print(f"RULER 128k (PF-NT, f16 KV) {name}: dense {s['mean']['dense']:.2f}; " + "; ".join(
            f"{c} {s['stats'][c]['delta']:+.2f} [{s['stats'][c]['ci95'][0]:+.2f}, {s['stats'][c]['ci95'][1]:+.2f}]" for c in (Q, PF, C27)))


def dev(run: Path) -> None:
    w: dict[tuple[str, str], tuple[float, float]] = {}
    for f in sorted((run / "e2e").glob("*_b1_*.jsonl")):
        rows = [json.loads(x) for x in f.read_text().splitlines() if x.strip()]
        t = [r for r in rows if r.get("kind") == "timed"]
        L, _, c = f.stem.split("_", 2)
        w[(L, c)] = (statistics.median(r["wall_s"] for r in t), statistics.median(r["prompt_ms"] / 1e3 for r in t))
    for L, best in (("256", "up_dense"), ("128", "up_mtp")):
        print(f"rebase dev run {L}k: {best} {w[(L, best)][0]:.1f} s -> pfsd27648 {w[(L, 'pfsd27648')][0]:.1f} s = "
              f"{w[(L, best)][0] / w[(L, 'pfsd27648')][0]:.2f}x; fork_dense TTFT {w[(L, 'fork_dense')][1]:.1f} s")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--decision", required=True)
    ap.add_argument("--pf-nt", required=True)
    ap.add_argument("--dev", required=True)
    a = ap.parse_args()
    d = json.loads(Path(a.decision).read_text())
    speed(d)
    quality(d, json.loads(Path(a.pf_nt).read_text()))
    dev(Path(a.dev))


if __name__ == "__main__":
    main()
