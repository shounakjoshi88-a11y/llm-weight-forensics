"""E21: can one row of lm_head reproduce the best prefill?

E20 measured the openers directly. Prefilling "Step 1: " gives 0.92 on 60 word
problems, against 0.17 with no prefill. That is the target.

E19 failed because it boosted six opener rows uniformly and at every position,
so the reply drifted and the reasoning degraded. This boosts a single row, with
the start-of-reply direction from e19, so the boost is large at the first token
and near zero afterwards.

If this works, the model starts reasoning on its own, from a 896-number change
in one tensor, with no prompt engineering.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C
from e17_word_problems import problems
from e18_math_mode import OTHER, first_h, first_logits

MAX_NEW = 384
ALPHAS = [1.0, 2.0, 3.0, 4.0, 6.0, 8.0]
ROWS = ["Step", "Lets"]


def main() -> int:
    p = C.load()
    with torch.no_grad():
        p.model.lm_head.weight = torch.nn.Parameter(
            p.model.model.embed_tokens.weight.detach().clone())
    W = p.model.lm_head.weight
    orig = W.detach().clone()

    row_ids = {}
    for name in ROWS:
        row_ids[name] = p.tok(name, add_special_tokens=False).input_ids[0]
    print("loaded", p.n_layers, "layers; rows", row_ids)

    train = problems(80, 20261120)
    held = problems(60, 20261121)
    wants = [w for _, w in held]

    starts, mids = [], []
    for q, _ in train:
        base = C.chat_text(p.tok, q, "")
        starts.append(first_h(p, base))
        mids.append(first_h(p, base + "Step"))
    d = torch.stack(starts).mean(0) - torch.stack(mids).mean(0)
    d = d / d.norm()

    held_prompts = [C.chat_text(p.tok, q, "") for q, _ in held]
    other_prompts = [C.chat_text(p.tok, q, "") for q in OTHER]
    dd = d.to(W.device, W.dtype)

    def rate(prompts, tid):
        k = 0
        for text in prompts:
            if int(first_logits(p, text).argmax()) == tid:
                k += 1
        return k / len(prompts)

    print(f"\n{'rows':10s} {'alpha':>5s} {'math Step':>10s} {'other Step':>11s}")
    best = None
    for name in ROWS:
        tid = row_ids[name]
        for a in ALPHAS:
            with torch.no_grad():
                W.copy_(orig)
                W[tid] += a * dd
            mr, orr = rate(held_prompts, tid), rate(other_prompts, tid)
            with torch.no_grad():
                W.copy_(orig)
            print(f"{name:10s} {a:5.1f} {mr:10.2f} {orr:11.2f}")
            if best is None and mr >= 0.85 and orr <= 0.25:
                best = (name, a)

    if best is None:
        best = ("Step", 4.0)
    print("\nchosen", best)

    out = {}
    for label, (name, alpha) in [("base", ("Step", 0.0)), ("edited", best)]:
        tid = row_ids[name]
        with torch.no_grad():
            W.copy_(orig)
            if alpha:
                W[tid] += alpha * dd
        reps = C.generate_batch(p, held_prompts, max_new_tokens=MAX_NEW, batch_size=20)
        oth = C.generate_batch(p, other_prompts, max_new_tokens=MAX_NEW, batch_size=20)
        with torch.no_grad():
            W.copy_(orig)
        exact = sum(C.last_int(r) == w for r, w in zip(reps, wants))
        lo, hi = C.wilson(exact, len(wants))
        print(f"  {label:7s} exact {exact}/{len(wants)} = {exact/len(wants):.2f} "
              f"[{lo:.2f}, {hi:.2f}]")
        out[label] = {"row": name, "alpha": alpha, "exact": exact / len(wants),
                      "ci": [lo, hi], "replies": reps, "other_replies": oth}

    C.save({"rows": row_ids, "chosen": best, "results": out},
           C.ROOT / "findings" / "e21_step_row.json")
    print("\nwrote findings/e21_step_row.json")
    print("\nthree edited replies:")
    for r in out["edited"]["replies"][:3]:
        print("  " + r.strip()[:240].replace("\n", " | "))
    print("\nthree edited non-math replies:")
    for r in out["edited"]["other_replies"][:3]:
        print("  " + r.strip()[:160].replace("\n", " | "))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
