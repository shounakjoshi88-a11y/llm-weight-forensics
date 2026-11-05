"""E20: which prefilled opener actually produces good reasoning?

E19 showed that a weight edit can force the model to start with an opener, but
the reasoning that follows is worse than the chain-of-thought prefill in e17.
Two candidate explanations: the edit is too blunt, or the opener it forces is
the wrong one.

This measures the opener directly, with no weight edit. Each condition is the
same 60 word problems with a different first phrase prefilled, at the same
token budget, so the comparison is clean.

The result decides how to weight the rows in the next edit: uniformly, or in
proportion to how well each opener works.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C
from e17_word_problems import problems

MAX_NEW = 384
N = 60

CONDS = [
    ("none", ""),
    ("answer", "The answer is"),
    ("First", "First, "),
    ("Let me", "Let me work through this step by step.\n"),
    ("Lets", "Let's work through this step by step.\n"),
    ("To solve", "To solve this, "),
    ("Step", "Step 1: "),
]


def main() -> int:
    p = C.load()
    print("loaded", p.n_layers, "layers, budget", MAX_NEW)

    rows = problems(N, 20261121)
    wants = [w for _, w in rows]

    out = {}
    print(f"\n{'opener':10s} {'exact':>8s} {'95% CI':>16s} {'median chars':>13s}")
    for name, pre in CONDS:
        prompts = [C.chat_text(p.tok, q, pre) for q, _ in rows]
        reps = C.generate_batch(p, prompts, max_new_tokens=MAX_NEW, batch_size=20)
        exact = sum(C.last_int(r) == w for r, w in zip(reps, wants))
        lo, hi = C.wilson(exact, len(wants))
        med = sorted(len(r) for r in reps)[len(reps) // 2]
        print(f"{name:10s} {exact:3d}/{len(wants):<3d}  [{lo:.2f}, {hi:.2f}] "
              f"{med:13d}")
        out[name] = {"prefill": pre, "exact": exact / len(wants),
                     "ci": [lo, hi], "median_chars": med, "replies": reps}

    C.save({"n": N, "results": out}, C.ROOT / "findings" / "e20_openers.json")
    print("\nwrote findings/e20_openers.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
