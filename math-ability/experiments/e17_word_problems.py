"""E17: word problems, immediate answer against chain of thought.

E15 used 15 hand-written problems and found 4/15 against 12/15. This is the
same test on 60 generated two-step problems with exact answers, so the effect
has an interval.

Every problem needs one product and one addition or subtraction, both in the
range the model handles. Nothing here is a raw multi-digit calculation.

Conditions, greedy, 256 tokens:
  answer     prefill "The answer is"   (the format used everywhere else)
  bare       no prefill
  cot        prefill "Let me work through this step by step."
"""

from __future__ import annotations

import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C

MAX_NEW = 256
COT = "Let me work through this step by step.\n"
N = 60


def problems(n: int, seed: int) -> list[tuple[str, int]]:
    rng = random.Random(seed)
    out = []
    while len(out) < n:
        k = len(out) % 5
        a = rng.randint(12, 96)
        b = rng.randint(3, 9)
        c = rng.randint(11, 89)
        if k == 0:
            d = rng.randint(1, min(9, a * b // 10))
            q = (f"A shop sells pens in packs of {a}. Maya buys {b} packs and "
                 f"gives {d} pens away. How many pens does she have left?")
            out.append((q, a * b - d))
        elif k == 1:
            d = rng.randint(2, 7)
            q = (f"A train travels {a} km per hour for {b} hours, then {c} km "
                 f"per hour for {d} hours. How far does it travel in total?")
            out.append((q, a * b + c * d))
        elif k == 2:
            d = rng.randint(2, min(9, a * b // 8))
            q = (f"There are {a} boxes with {b} apples each. {d} apples are "
                 f"rotten. How many good apples are there?")
            out.append((q, a * b - d))
        elif k == 3:
            d = rng.randint(1, a * b // 6)
            q = (f"Liam saves {a} dollars a week for {b} weeks. He spends {d} "
                 f"dollars on a bike. How much money does he have left?")
            out.append((q, a * b - d))
        else:
            d = rng.randint(1, a * b - 1)
            q = (f"A tank holds {a * b} litres. It already has {d} litres. "
                 f"How many more litres can it hold?")
            out.append((q, a * b - d))
    return out


def main() -> int:
    p = C.load()
    print("loaded", p.n_layers, "layers")

    rows = problems(N, 20261114)
    wants = [w for _, w in rows]

    conds = [
        ("answer", [C.chat_text(p.tok, q, "The answer is") for q, _ in rows]),
        ("bare", [C.chat_text(p.tok, q, "") for q, _ in rows]),
        ("cot", [C.chat_text(p.tok, q, COT) for q, _ in rows]),
    ]

    out = {}
    print(f"\n{'condition':9s} {'last':>8s} {'95% CI':>16s} {'contains':>10s} "
          f"{'median chars':>13s}")
    for name, prompts in conds:
        reps = C.generate_batch(p, prompts, max_new_tokens=MAX_NEW, batch_size=30)
        last = sum(C.last_int(r) == w for r, w in zip(reps, wants))
        has = sum(w in C.integers(r) for r, w in zip(reps, wants))
        lo, hi = C.wilson(last, len(wants))
        chars = sorted(len(r) for r in reps)
        print(f"{name:9s} {last:3d}/{len(wants):<3d}  [{lo:.2f}, {hi:.2f}] "
              f"{has:7d}/{len(wants)} {chars[len(chars)//2]:13d}")
        out[name] = {"last": last / len(wants), "ci": [lo, hi],
                     "contains": has / len(wants),
                     "median_chars": chars[len(chars) // 2], "replies": reps}

    C.save({"n": N, "problems": rows, "results": out},
           C.ROOT / "findings" / "e17_word_problems.json")
    print("\nwrote findings/e17_word_problems.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
