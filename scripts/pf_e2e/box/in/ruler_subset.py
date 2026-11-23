"""Keep the first K samples of each RULER task in a llama-sparse-tail --gen file.

  ruler_subset.py <in.i32> <out.i32> --per-task K [--tasks 0,5,12] [--ids 1,2001]

Format: int32 n, then per prompt int32 [id, max_new, len] and len token ids. Sample ids are task*1000 + index.
"""

import argparse
import struct


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("dst")
    ap.add_argument("--per-task", type=int, required=True)
    ap.add_argument("--tasks", default="", help="task indices (id // 1000) to keep, default all")
    ap.add_argument("--ids", default="", help="sample ids to keep, default all")
    args = ap.parse_args()
    ids = {int(i) for i in args.ids.split(",")} if args.ids else None
    tasks = {int(t) for t in args.tasks.split(",")} if args.tasks else None
    raw = open(args.src, "rb").read()
    (n,) = struct.unpack_from("<i", raw, 0)
    off = 4
    keep: list[bytes] = []
    for _ in range(n):
        pid, max_new, ln = struct.unpack_from("<3i", raw, off)
        end = off + 12 + 4 * ln
        if pid % 1000 < args.per_task and (tasks is None or pid // 1000 in tasks) and (ids is None or pid in ids):
            keep.append(raw[off:end])
        off = end
    if off != len(raw):
        raise SystemExit(f"{args.src}: {len(raw) - off} trailing bytes")
    with open(args.dst, "wb") as f:
        f.write(struct.pack("<i", len(keep)))
        f.writelines(keep)
    print(f"{n} -> {len(keep)} prompts")


if __name__ == "__main__":
    main()
