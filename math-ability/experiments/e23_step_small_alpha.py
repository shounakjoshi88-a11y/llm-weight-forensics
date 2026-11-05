"""E23: generation at the small alphas that only just win the first token.

E22 found the transition: at alpha 0.05 the "Step" row wins 15 percent of first
tokens, at 0.10 it wins all of them, and at 0.10 the next token is not "Step".
E21 jumped straight to alpha 1.0 and above, where the boost also dominates
later positions and the reply collapses into "Step Step Step".

This runs the three alphas at the transition and scores the whole reply.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C
from e17_word_problems import problems
from e18_math_mode import OTHER, first_h

MAX_NEW = 384
ALPHAS = [0.05, 0.10, 0.20]


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
    held = problems(60, 20261121)
    wants = [w for _, w in held]

    starts, mids = [], []
    for q, _ in train:
        base = C.chat_text(p.tok, q, "")
        starts.append(first_h(p, base))
        mids.append(first_h(p, base + "Step"))
    d = torch.stack(starts).mean(0) - torch.stack(mids).mean(0)
    d = d / d.norm()
    dd = d.to(W.device, W.dtype)

    held_prompts = [C.chat_text(p.tok, q, "") for q, _ in held]
    other_prompts = [C.chat_text(p.tok, q, "") for q in OTHER]

    out = {}
    print(f"\n{'alpha':>6s} {'exact':>8s} {'95% CI':>16s} {'median chars':>13s}")
    for a in [0.0] + ALPHAS:
        with torch.no_grad():
            W.copy_(orig)
            if a:
                W[step] += a * dd
        reps = C.generate_batch(p, held_prompts, max_new_tokens=MAX_NEW, batch_size=20)
        oth = C.generate_batch(p, other_prompts, max_new_tokens=MAX_NEW, batch_size=20)
        with torch.no_grad():
            W.copy_(orig)
        exact = sum(C.last_int(r) == w for r, w in zip(reps, wants))
        lo, hi = C.wilson(exact, len(wants))
        med = sorted(len(r) for r in reps)[len(reps) // 2]
        print(f"{a:6.2f} {exact:3d}/{len(wants):<3d}  [{lo:.2f}, {hi:.2f}] {med:13d}")
        out[f"alpha{a}"] = {"alpha": a, "exact": exact / len(wants),
                            "ci": [lo, hi], "median_chars": med,
                            "replies": reps, "other_replies": oth}

    C.save(out, C.ROOT / "findings" / "e23_step_small_alpha.json")
    print("\nwrote findings/e23_step_small_alpha.json")
    best = max(out.items(), key=lambda kv: kv[1]["exact"])
    print(f"\nbest {best[0]}: {best[1]['exact']:.2f}")
    for r in best[1]["replies"][:3]:
        print("  " + r.strip()[:240].replace("\n", " | "))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
