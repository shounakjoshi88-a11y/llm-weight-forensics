"""E14: does this model loop?

The Astra result on Qwen3.5-0.8B is a loop-removal result: 93 percent of its
MATH-500 answers hit the 32,768 token limit, and subtracting one MLP direction
took that to 33 percent and MATH-500 from 7.9 to 48.1. Their own Gemma control,
which does not loop, did not improve.

Every experiment in this folder so far used a 12 to 16 token budget with an
answer prefill, which makes a loop impossible by construction. This checks the
regime the Astra edit lives in: open chain of thought, long budget, greedy.

No edit here. This is only a measurement of whether there is a loop to remove.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C

MAX_NEW = 512

WORDS = [
    ("A shop sells pens in packs of 12. Maya buys 7 packs and gives 15 pens away. How many pens does she have left?", 69),
    ("A train travels 68 km per hour for 4 hours, then 45 km per hour for 3 hours. How far does it travel in total?", 407),
    ("There are 23 boxes with 14 apples each. 37 apples are rotten. How many good apples are there?", 285),
    ("Liam saves 35 dollars a week for 19 weeks. He spends 240 dollars on a bike. How much is left?", 425),
    ("A rectangle is 47 cm long and 26 cm wide. What is its area in square centimetres?", 1222),
    ("A tank holds 850 litres. It is filled at 65 litres per minute for 9 minutes. How much more can it hold?", 265),
    ("There are 96 students split into 8 equal groups. Each group needs 3 helpers. How many helpers in total?", 36),
    ("A book has 312 pages. Sam reads 48 pages a day for 5 days. How many pages are left?", 72),
    ("A farmer has 74 sheep and buys 39 more. He then sells 28. How many does he have?", 85),
    ("Each crate holds 24 bottles. A truck carries 15 crates. 46 bottles break. How many are intact?", 314),
    ("A worker earns 17 dollars an hour and works 38 hours. How much does she earn?", 646),
    ("A field is 93 metres by 57 metres. A fence runs around the edge. How many metres of fence?", 300),
    ("A jar has 245 sweets. They are shared equally among 7 children. How many does each get?", 35),
    ("A cyclist rides 26 km a day for 14 days, then 19 km a day for 6 days. Total distance?", 478),
    ("A school orders 64 boxes of 25 pencils. They give 130 pencils away. How many remain?", 1470),
]

DIRECT = [
    ("What is 57892 x 512? Work through it step by step.", 29640704),
    ("What is 4837 x 296? Work through it step by step.", 1431752),
    ("What is 87654 + 98765? Work through it step by step.", 186419),
    ("What is 70213 - 48697? Work through it step by step.", 21516),
    ("What is 394 x 87? Work through it step by step.", 34278),
]


def loop_score(text: str) -> float:
    """Fraction of the reply taken up by its most repeated 40-char window."""
    t = re.sub(r"\s+", " ", text)
    if len(t) < 120:
        return 0.0
    from collections import Counter
    grams = [t[i:i + 40] for i in range(0, len(t) - 40, 20)]
    if not grams:
        return 0.0
    _, n = Counter(grams).most_common(1)[0]
    return (n * 20) / len(t)


def main() -> int:
    p = C.load()
    print("loaded", p.n_layers, "layers, budget", MAX_NEW)

    cases = [("word", q, w) for q, w in WORDS] + [("direct", q, w) for q, w in DIRECT]
    prompts = [C.chat_text(p.tok, q, "") for _, q, _ in cases]
    reps = C.generate_batch(p, prompts, max_new_tokens=MAX_NEW, batch_size=20)

    out = []
    for (kind, q, want), rep in zip(cases, reps):
        ls = loop_score(rep)
        got = C.first_int(rep)
        last = C.last_int(rep)
        hit = len(rep) > 0.95 * MAX_NEW * 4
        out.append({"kind": kind, "q": q, "want": want, "got": got, "last": last,
                    "loop": ls, "chars": len(rep), "ok": got == want})
        print(f"  {kind:6s} loop {ls:.2f} chars {len(rep):4d} first {str(got):>9s} "
              f"last {str(last):>9s} want {want:>9d} {'OK' if got == want else 'X'}")

    loops = [o for o in out if o["loop"] > 0.25]
    print(f"\nrepetitive (loop score > 0.25): {len(loops)}/{len(out)}")
    print(f"first integer correct: {sum(o['ok'] for o in out)}/{len(out)}")
    print(f"longest reply: {max(o['chars'] for o in out)} chars")

    C.save(out, C.ROOT / "findings" / "e14_loop_probe.json")
    print("wrote findings/e14_loop_probe.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
