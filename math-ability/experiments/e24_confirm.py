"""E24: confirmation of the "Step" row edit on fresh problems.

E23 found alpha 0.10 on 60 held-out problems, which were also used to pick it.
This scores the same edit once on 100 problems from a different seed, next to
the base model and the "Step 1: " prefill, and runs a paired exact test between
base and edit.

Also checks the non-math prompts, since the edit is a constant boost to one
output row and has no notion of task.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C
from e17_word_problems import problems
from e18_math_mode import OTHER, first_h

MAX_NEW = 384
ALPHA = 0.10
N = 100


def mcnemar(b: int, c: int) -> float:
    """Exact two-sided p for discordant pairs b and c."""
    from math import comb
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(comb(n, i) for i in range(k + 1)) / (2 ** n)
    return min(1.0, 2 * tail)


def main() -> int:
    p = C.load()
    with torch.no_grad():
        p.model.lm_head.weight = torch.nn.Parameter(
            p.model.model.embed_tokens.weight.detach().clone())
    W = p.model.lm_head.weight
    orig = W.detach().clone()
    step = p.tok("Step", add_special_tokens=False).input_ids[0]
    print("loaded", p.n_layers, "layers; Step id", step)

    train = problems(80, 20261120)
    rows = problems(N, 20261122)
    wants = [w for _, w in rows]
    prompts = [C.chat_text(p.tok, q, "") for q, _ in rows]
    pre_prompts = [C.chat_text(p.tok, q, "Step 1: ") for q, _ in rows]
    other_prompts = [C.chat_text(p.tok, q, "") for q in OTHER]

    starts, mids = [], []
    for q, _ in train:
        base = C.chat_text(p.tok, q, "")
        starts.append(first_h(p, base))
        mids.append(first_h(p, base + "Step"))
    d = torch.stack(starts).mean(0) - torch.stack(mids).mean(0)
    d = d / d.norm()
    dd = d.to(W.device, W.dtype)

    res = {}
    print(f"\n{'condition':10s} {'exact':>8s} {'95% CI':>16s}")
    for label, pr, alpha in [("base", prompts, 0.0),
                             ("edit", prompts, ALPHA),
                             ("prefill", pre_prompts, 0.0)]:
        with torch.no_grad():
            W.copy_(orig)
            if alpha:
                W[step] += alpha * dd
        reps = C.generate_batch(p, pr, max_new_tokens=MAX_NEW, batch_size=20)
        with torch.no_grad():
            W.copy_(orig)
        ok = [C.last_int(r) == w for r, w in zip(reps, wants)]
        lo, hi = C.wilson(sum(ok), len(ok))
        print(f"{label:10s} {sum(ok):3d}/{len(ok):<3d}  [{lo:.2f}, {hi:.2f}]")
        res[label] = {"exact": sum(ok) / len(ok), "ci": [lo, hi],
                      "ok": ok, "replies": reps}

    b = sum(1 for x, y in zip(res["base"]["ok"], res["edit"]["ok"]) if y and not x)
    c = sum(1 for x, y in zip(res["base"]["ok"], res["edit"]["ok"]) if x and not y)
    pv = mcnemar(b, c)
    print(f"\nbase -> edit: {b} fixed, {c} broken, exact p = {pv:.2e}")

    with torch.no_grad():
        W.copy_(orig)
        W[step] += ALPHA * dd
    oth = C.generate_batch(p, other_prompts, max_new_tokens=256, batch_size=20)
    with torch.no_grad():
        W.copy_(orig)
    res["other_edited"] = oth
    print("\nnon-math prompts, edited:")
    for q, r in list(zip(OTHER, oth))[:6]:
        print(f"  {q[:34]:34s} -> {r.strip()[:90].replace(chr(10), ' | ')}")

    C.save({"alpha": ALPHA, "n": N, "results": res,
            "mcnemar": {"fixed": b, "broken": c, "p": pv}},
           C.ROOT / "findings" / "e24_confirm.json")
    print("\nwrote findings/e24_confirm.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
