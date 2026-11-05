"""E01: where does arithmetic actually break?

Runs a held-out grid of operations and digit lengths in two prompt formats,
plus a small format comparison on the hard cells. Exact final-integer scoring.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import common as C

N = 20
SEED = 20261105

CELLS = [
    ("add", 1, 1), ("add", 2, 1), ("add", 2, 2), ("add", 3, 2),
    ("add", 3, 3), ("add", 4, 4), ("add", 5, 5), ("add", 7, 7),
    ("sub", 2, 1), ("sub", 2, 2), ("sub", 3, 2), ("sub", 3, 3), ("sub", 4, 4),
    ("mul", 1, 1), ("mul", 2, 1), ("mul", 2, 2), ("mul", 3, 1),
    ("mul", 3, 2), ("mul", 5, 1), ("mul", 5, 3),
    ("div", 2, 1), ("div", 3, 1), ("div", 4, 2),
]

FORMAT_CELLS = [
    ("add", 3, 3), ("add", 5, 5), ("sub", 2, 2), ("sub", 3, 3),
    ("mul", 2, 2), ("mul", 3, 2),
]


def run(p, rows, fmt, max_new):
    prompts = [C.chat_text(p.tok, *C.prompt(r, fmt)) for r in rows]
    replies = C.generate_batch(p, prompts, max_new_tokens=max_new)
    return [
        {**r, "fmt": fmt, "reply": rep, **C.score(rep, r["want"])}
        for r, rep in zip(rows, replies)
    ]


def heldout(op, da, db, n):
    rows = [r for r in C.make(op, da, db, n * 3, SEED) if r["split"] == "heldout"]
    return rows[:n]


def main() -> int:
    p = C.load()
    print("loaded", p.n_layers, "layers")

    out = {"n": N, "seed": SEED, "grid": {}, "formats": {}}

    print("\nheld-out grid, answer prefill")
    print(f"{'cell':10s} {'n':>3s} {'ok':>3s} {'acc':>6s}  {'95% CI':>14s}")
    for op, da, db in CELLS:
        rows = heldout(op, da, db, N)
        got = run(p, rows, "answer", 16)
        k = sum(g["ok"] for g in got)
        lo, hi = C.wilson(k, len(got))
        key = f"{op} {da}x{db}"
        out["grid"][key] = got
        print(f"{key:10s} {len(got):3d} {k:3d} {k/len(got):6.2f}  [{lo:.2f}, {hi:.2f}]")

    print("\nheld-out grid, bare")
    print(f"{'cell':10s} {'n':>3s} {'ok':>3s} {'acc':>6s}")
    for op, da, db in CELLS:
        rows = heldout(op, da, db, N)
        got = run(p, rows, "bare", 48)
        k = sum(g["ok"] for g in got)
        out["grid"][f"{op} {da}x{db} bare"] = got
        cell = f"{op} {da}x{db}"
        print(f"{cell:10s} {len(got):3d} {k:3d} {k/len(got):6.2f}")

    print("\nformat comparison on hard cells")
    print(f"{'cell':10s} {'answer':>8s} {'bare':>6s} {'steps':>6s} {'column':>7s}")
    for op, da, db in FORMAT_CELLS:
        rows = heldout(op, da, db, N)
        accs = {}
        for fmt, mx in [("answer", 16), ("bare", 48), ("steps", 192), ("column", 192)]:
            got = run(p, rows, fmt, mx)
            k = sum(g["ok"] for g in got)
            accs[fmt] = k / len(got)
            out["formats"][f"{op} {da}x{db} {fmt}"] = got
        cell = f"{op} {da}x{db}"
        print(f"{cell:10s} {accs['answer']:8.2f} {accs['bare']:6.2f} "
              f"{accs['steps']:6.2f} {accs['column']:7.2f}")

    path = C.save(out, C.ROOT / "findings" / "e01_failure_map.json")
    print("\nwrote", path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
