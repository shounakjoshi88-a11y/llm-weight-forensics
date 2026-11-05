"""E16: is the stop decision editable?

E15 showed that the model gets 4/15 word problems when it answers immediately
and 12/15 when it is pushed into chain of thought, with no weight change. E14
showed it does not loop. So the failure is a stopping policy: it emits one
number and EOS.

This is the Astra-shaped question. Their edit removed a loop direction from two
MLP layers and MATH-500 went 7.9 to 48.1 because the baseline was crippled by
the loop. Here the crippling behaviour is an early EOS. If that is a policy and
not a missing computation, a small edit to the EOS row of lm_head should
recover it.

Edit: scale the EOS row of lm_head by a constant. That is 896 numbers, a
rank-one change with v = e_eos. The embedding table is separated from lm_head
first so the input side is untouched.

Regression: non-math prompts, to check the model does not ramble.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C
import edits as E
from e14_loop_probe import WORDS, loop_score
from e15_continuation import MAX_NEW

OTHER = [
    "What is the capital of France?",
    "Name three primary colours.",
    "Write one sentence about the sea.",
    "What does HTTP stand for?",
    "Say hello in Japanese.",
    "Is 7 a prime number?",
    "Summarise what a for loop does in one sentence.",
    "What is the boiling point of water in Celsius?",
    "Give one synonym for happy.",
    "Who wrote Hamlet?",
]

SCALES = [1.0, 0.5, 0.25, 0.0]


def main() -> int:
    p = C.load()
    with torch.no_grad():
        p.model.lm_head.weight = torch.nn.Parameter(
            p.model.model.embed_tokens.weight.detach().clone())

    eos = p.tok.eos_token_id
    W = p.model.lm_head.weight
    orig = W.detach().clone()
    print("loaded", p.n_layers, "layers; eos id", eos,
          "row norm", round(orig[eos].float().norm().item(), 3))

    wants = [w for _, w in WORDS]
    math_prompts = [C.chat_text(p.tok, q, "") for q, _ in WORDS]
    other_prompts = [C.chat_text(p.tok, q, "") for q in OTHER]

    out = {}
    print(f"\n{'scale':>6s} {'last':>8s} {'contains':>9s} {'median chars':>13s} "
          f"{'other median chars':>19s}")
    for s in SCALES:
        with torch.no_grad():
            W.copy_(orig)
            W[eos] *= s
        reps = C.generate_batch(p, math_prompts, max_new_tokens=MAX_NEW, batch_size=15)
        oth = C.generate_batch(p, other_prompts, max_new_tokens=MAX_NEW, batch_size=10)
        with torch.no_grad():
            W.copy_(orig)

        last = sum(C.last_int(r) == w for r, w in zip(reps, wants))
        has = sum(w in C.integers(r) for r, w in zip(reps, wants))
        chars = sorted(len(r) for r in reps)
        ochars = sorted(len(r) for r in oth)
        n = len(wants)
        print(f"{s:6.2f} {last:5d}/{n} {has:6d}/{n} {chars[len(chars)//2]:13d} "
              f"{ochars[len(ochars)//2]:19d}")
        out[f"scale{s}"] = {
            "last": last / n, "contains": has / n,
            "median_chars": chars[len(chars) // 2],
            "other_median_chars": ochars[len(ochars) // 2],
            "math_replies": reps, "other_replies": oth,
        }

    C.save(out, C.ROOT / "findings" / "e16_stop_edit.json")
    print("\nwrote findings/e16_stop_edit.json")

    print("\none reply at scale 0.5:")
    print(out["scale0.5"]["math_replies"][0].strip()[:300])
    print("\none non-math reply at scale 0.5:")
    print(out["scale0.5"]["other_replies"][0].strip()[:300])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
