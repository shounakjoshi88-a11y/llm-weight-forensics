"""E13: does asking three ways help the atoms?

The scaffold asks every atom three ways and takes the majority. That only helps
if the model's errors vary with the phrasing. This measures first-answer
accuracy against three-way majority on the two atoms the scaffold uses, plus
3-digit addition as a reference.

100 held-out problems per cell.
"""

from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C

ADD_T = ["What is {a} + {b}?", "Compute {a} plus {b}.", "Add {a} and {b}."]
MUL_T = ["What is {a} x {b}?", "Compute {a} times {b}.", "Multiply {a} by {b}."]
CELLS = [("add", 2, 2, ADD_T), ("add", 3, 3, ADD_T), ("mul", 2, 1, MUL_T)]
N = 100
SEEDS = [131, 132]


def majority(vals):
    vals = [v for v in vals if v is not None]
    if not vals:
        return None
    c = Counter(vals).most_common()
    if len(c) > 1 and c[0][1] == c[1][1]:
        return vals[0]
    return c[0][0]


def main() -> int:
    p = C.load()
    out = {}
    for op, da, db, T in CELLS:
        rows = []
        for s in SEEDS:
            pool = C.make(op, da, db, N, s)
            rows += [r for r in pool if r["split"] == "heldout"]
        rows = rows[:N]
        prompts = [C.chat_text(p.tok, t.format(a=r["a"], b=r["b"]), "The answer is")
                   for r in rows for t in T]
        reps = C.generate_batch(p, prompts, max_new_tokens=12, batch_size=60)

        first = maj = any_ok = disagree = 0
        for i, r in enumerate(rows):
            vals = [C.last_int(reps[i * len(T) + j]) for j in range(len(T))]
            first += vals[0] == r["want"]
            maj += majority(vals) == r["want"]
            any_ok += any(v == r["want"] for v in vals)
            disagree += len(set(v for v in vals if v is not None)) > 1
        n = len(rows)
        out[f"{op}{da}x{db}"] = {
            "n": n, "first": first / n, "majority": maj / n,
            "any": any_ok / n, "disagreement_rate": disagree / n,
        }
        print(f"  {op} {da}x{db}: n={n}  first {first/n:.2f}  majority {maj/n:.2f}  "
              f"any {any_ok/n:.2f}  disagree {disagree/n:.2f}")

    C.save(out, C.ROOT / "findings" / "e13_atom_votes.json")
    print("wrote findings/e13_atom_votes.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
