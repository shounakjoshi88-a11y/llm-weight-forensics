"""E12: the base-100 scaffold across digit cells.

Reuses the e11 algorithm. For each cell, 30 held-out problems, exact match on
the assembled integer. Every atom is a 2-digit by 1-digit product or a 2-digit
plus 2-digit sum, each asked three ways with a majority.

The point of the sweep is to show that the scaffold accuracy is set by the atom
reliability and the number of atoms, not by the size of the problem.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C
from e11_scaffold_v2 import Engine, chunk_mul

CELLS = [(2, 2), (3, 2), (4, 2), (3, 3), (5, 3)]
N = 30
SEEDS = [121, 122]


def main() -> int:
    p = C.load()
    eng = Engine(p)
    print("loaded", p.n_layers, "layers")

    results = {}
    for da, db in CELLS:
        rows = []
        for s in SEEDS:
            pool = C.make("mul", da, db, N, s)
            rows += [r for r in pool if r["split"] == "heldout"]
        rows = rows[:N]
        ok, calls, per = 0, 0, []
        for r in rows:
            before = eng.asks
            v = chunk_mul(eng, r["a"], r["b"])
            calls += eng.asks - before
            good = v == r["want"]
            ok += good
            per.append({"a": r["a"], "b": r["b"], "want": r["want"], "got": v, "ok": good})
        acc = ok / len(rows)
        results[f"{da}x{db}"] = {
            "acc": acc, "ok": ok, "n": len(rows),
            "asks_per_problem": calls / len(rows), "rows": per,
        }
        print(f"  {da}x{db}: {ok}/{len(rows)} = {acc:.2f}   "
              f"{calls/len(rows):.1f} asks/problem")

    C.save(results, C.ROOT / "findings" / "e12_scaffold_sweep.json")
    print("wrote findings/e12_scaffold_sweep.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
