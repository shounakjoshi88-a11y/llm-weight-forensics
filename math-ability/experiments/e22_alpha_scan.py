"""E22: how small can the boost be and still win the first token?

E21 boosted the "Step" row by a start-of-reply direction. At alpha 1.0 and
above the model starts with "Step" every time, then repeats it, because the
same boost is still winning at later positions.

If the boost is only just enough to win the first token, the continuation
should be left alone. This scans small alphas and measures two things without
generating:

    P(first token is Step)
    P(second token is Step | first token is Step)

The second number is the loop. It should be near zero for a usable edit.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C
from e17_word_problems import problems
from e18_math_mode import first_h, first_logits

ALPHAS = [0.0, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0]
N = 20


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
    held = problems(N, 20261121)

    starts, mids = [], []
    for q, _ in train:
        base = C.chat_text(p.tok, q, "")
        starts.append(first_h(p, base))
        mids.append(first_h(p, base + "Step"))
    d = torch.stack(starts).mean(0) - torch.stack(mids).mean(0)
    d = d / d.norm()
    dd = d.to(W.device, W.dtype)

    prompts = [C.chat_text(p.tok, q, "") for q, _ in held]

    print(f"\n{'alpha':>6s} {'P(Step first)':>14s} {'P(Step again)':>14s} "
          f"{'logit Step @start':>18s}")
    for a in ALPHAS:
        with torch.no_grad():
            W.copy_(orig)
            W[step] += a * dd
        first_ok = again = 0
        margins = []
        for text in prompts:
            lg = first_logits(p, text)
            margins.append(float(lg[step] - lg.max()))
            if int(lg.argmax()) == step:
                first_ok += 1
                lg2 = first_logits(p, text + "Step")
                if int(lg2.argmax()) == step:
                    again += 1
        with torch.no_grad():
            W.copy_(orig)
        denom = max(1, first_ok)
        print(f"{a:6.2f} {first_ok/N:14.2f} {again/denom:14.2f} "
              f"{sum(margins)/len(margins):18.2f}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
