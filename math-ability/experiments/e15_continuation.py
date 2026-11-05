"""E15: is the word-problem failure a missing ability or a premature stop?

E14 found that this model rarely writes any reasoning. Ten of twenty replies
were two to four characters: one number, then EOS. When it did write working it
usually landed on the right answer.

That is the opposite of the Qwen3.5-0.8B failure in the Astra post, which
looped. This measures the failure this model actually has, by forcing it to
keep generating.

Four conditions, greedy, 256 tokens:
  bare            no prefill, normal stopping
  bare forced     no prefill, EOS suppressed for 128 tokens
  cot             prefill "Let me work through this step by step."
  cot forced      that prefill, EOS suppressed for 128 tokens
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C
from e14_loop_probe import WORDS, loop_score

MAX_NEW = 256
FORCE = 128
COT = "Let me work through this step by step.\n"


def run(p, prompts, min_new):
    return C.generate_batch(p, prompts, max_new_tokens=MAX_NEW,
                            batch_size=15, min_new_tokens=min_new)


def main() -> int:
    p = C.load()
    print("loaded", p.n_layers, "layers")

    qs = [q for q, _ in WORDS]
    wants = [w for _, w in WORDS]

    conds = [
        ("bare", [C.chat_text(p.tok, q, "") for q in qs], 0),
        ("bare forced", [C.chat_text(p.tok, q, "") for q in qs], FORCE),
        ("cot", [C.chat_text(p.tok, q, COT) for q in qs], 0),
        ("cot forced", [C.chat_text(p.tok, q, COT) for q in qs], FORCE),
    ]

    out = {}
    print(f"\n{'condition':12s} {'first':>7s} {'last':>7s} {'contains':>9s} "
          f"{'median chars':>13s} {'loop':>6s}")
    for name, prompts, mn in conds:
        reps = run(p, prompts, mn)
        first = sum(C.first_int(r) == w for r, w in zip(reps, wants))
        last = sum(C.last_int(r) == w for r, w in zip(reps, wants))
        has = sum(w in C.integers(r) for r, w in zip(reps, wants))
        chars = sorted(len(r) for r in reps)
        loops = sum(loop_score(r) > 0.25 for r in reps)
        n = len(wants)
        print(f"{name:12s} {first:4d}/{n} {last:4d}/{n} {has:6d}/{n} "
              f"{chars[len(chars)//2]:13d} {loops:5d}/{n}")
        out[name] = {"first": first / n, "last": last / n, "contains": has / n,
                     "median_chars": chars[len(chars) // 2], "loops": loops,
                     "replies": reps}

    C.save(out, C.ROOT / "findings" / "e15_continuation.json")
    print("\nwrote findings/e15_continuation.json")

    best = out["cot forced"]["replies"]
    print("\nthree cot forced replies:")
    for q, w, r in list(zip(qs, wants, best))[:3]:
        print(f"\n  want {w}\n  {r.strip()[:300]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
