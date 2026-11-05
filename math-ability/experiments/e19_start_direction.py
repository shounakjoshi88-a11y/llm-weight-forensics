"""E19: a start-of-reply direction, not a math direction.

E18 boosted the opener rows of lm_head with a math-vs-other direction. It fired
everywhere, so once the model wrote "First" it kept writing openers and looped.

A weight edit cannot be position-conditional, but the direction can be. Capture
the residual at two places:

    h_start  last prompt token, about to write the first reply token
    h_mid    the same prompt with "First" appended, about to write the second

and take d = mean(h_start) - mean(h_mid). Boosting the opener rows by d then
raises them at the start of a reply and does nothing in the middle of one.

Same edit surface as e18: six rows of lm_head, 896 numbers each.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C
from e17_word_problems import problems
from e18_math_mode import OPENERS, OTHER, first_h, first_logits, opener_ids

ALPHAS = [0.5, 1.0, 2.0, 3.0, 4.0, 6.0]


def main() -> int:
    p = C.load()
    with torch.no_grad():
        p.model.lm_head.weight = torch.nn.Parameter(
            p.model.model.embed_tokens.weight.detach().clone())
    W = p.model.lm_head.weight
    orig = W.detach().clone()
    oids = opener_ids(p)
    print("loaded", p.n_layers, "layers; opener rows", len(oids))

    train = problems(80, 20261120)
    held = problems(60, 20261121)

    starts, mids = [], []
    for q, _ in train:
        base = C.chat_text(p.tok, q, "")
        starts.append(first_h(p, base))
        mids.append(first_h(p, base + "First"))
    d = torch.stack(starts).mean(0) - torch.stack(mids).mean(0)
    d = d / d.norm()
    print("start direction: cos to start",
          round(float((torch.stack(starts) @ d).mean()), 2),
          "to mid", round(float((torch.stack(mids) @ d).mean()), 2))

    held_prompts = [C.chat_text(p.tok, q, "") for q, _ in held]
    other_prompts = [C.chat_text(p.tok, q, "") for q in OTHER]

    def opener_rate(prompts, suffix=""):
        k = 0
        for text in prompts:
            lg = first_logits(p, text + suffix)
            if int(lg.argmax()) in oids:
                k += 1
        return k / len(prompts)

    dd = d.to(W.device, W.dtype)
    print(f"\n{'alpha':>5s} {'start math':>11s} {'start other':>12s} {'mid math':>9s}")
    print(f"{0:5.1f} {opener_rate(held_prompts):11.2f} "
          f"{opener_rate(other_prompts):12.2f} {opener_rate(held_prompts, 'First'):9.2f}")
    chosen = None
    for a in ALPHAS:
        with torch.no_grad():
            W.copy_(orig)
            for r in oids:
                W[r] += a * dd
        sm = opener_rate(held_prompts)
        so = opener_rate(other_prompts)
        mm = opener_rate(held_prompts, "First")
        with torch.no_grad():
            W.copy_(orig)
        print(f"{a:5.1f} {sm:11.2f} {so:12.2f} {mm:9.2f}")
        if chosen is None and sm >= 0.8 and mm <= 0.2:
            chosen = a
    if chosen is None:
        chosen = 2.0
    print("chosen alpha", chosen)

    wants = [w for _, w in held]
    out = {}
    for label, alpha in [("base", 0.0), ("edited", chosen)]:
        with torch.no_grad():
            W.copy_(orig)
            for r in oids:
                W[r] += alpha * dd
        reps = C.generate_batch(p, held_prompts, max_new_tokens=256, batch_size=20)
        oth = C.generate_batch(p, other_prompts, max_new_tokens=256, batch_size=20)
        with torch.no_grad():
            W.copy_(orig)
        exact = sum(C.last_int(r) == w for r, w in zip(reps, wants))
        lo, hi = C.wilson(exact, len(wants))
        med = sorted(len(r) for r in reps)[len(reps) // 2]
        omed = sorted(len(r) for r in oth)[len(oth) // 2]
        print(f"  {label:7s} exact {exact}/{len(wants)} = {exact/len(wants):.2f} "
              f"[{lo:.2f}, {hi:.2f}]  median chars {med}  other median {omed}")
        out[label] = {"alpha": alpha, "exact": exact / len(wants), "ci": [lo, hi],
                      "median_chars": med, "other_median": omed,
                      "replies": reps, "other_replies": oth}

    C.save({"opener_ids": oids, "chosen_alpha": chosen, "results": out},
           C.ROOT / "findings" / "e19_start_direction.json")
    print("\nwrote findings/e19_start_direction.json")
    print("\nthree edited replies:")
    for r in out["edited"]["replies"][:3]:
        print("  " + r.strip()[:220].replace("\n", " | "))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
