"""E18: a math-mode direction in lm_head.

Finding 04 showed the failure is that the model answers before it reasons, and
that a chain-of-thought prefill fixes it (0.23 to 0.78 on word problems). The
EOS row edit failed because it was not conditional on the prompt.

This is the conditional version. Build a direction in the final normed residual
that separates math prompts from other prompts:

    d = mean h(math) - mean h(other),  normalised

Then boost only the rows of lm_head for reasoning openers, in proportion to how
much the residual looks like math:

    W_lm[opener] += alpha * d

When the residual is math-like, the opener logits rise and the model starts
writing "First, ..." instead of a digit. When it is not, d is near orthogonal
and nothing happens. Only a handful of rows change: 896 numbers each.

No gradients, no new parameters. lm_head is separated from the embedding table
first so the input side is untouched.
"""

from __future__ import annotations

import random
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C
import edits as E
from e17_word_problems import problems

OPENERS = ["First", "Let", "To", "Step", "We", "The"]
ALPHAS = [0.5, 1.0, 2.0, 4.0, 8.0]

OTHER = [
    "What is the capital of France?", "Name three primary colours.",
    "Write one sentence about the sea.", "What does HTTP stand for?",
    "Say hello in Japanese.", "Is 7 a prime number?",
    "Summarise what a for loop does.", "What is the boiling point of water?",
    "Give one synonym for happy.", "Who wrote Hamlet?",
    "What language is spoken in Brazil?", "Name a planet with rings.",
    "What is the largest ocean?", "Who painted the Mona Lisa?",
    "What is the chemical symbol for gold?", "How many days in February?",
    "Name a mammal that lays eggs.", "What is the tallest mountain?",
    "Translate good morning into Spanish.", "What is a haiku?",
    "Name the author of Pride and Prejudice.", "What gas do plants absorb?",
    "What is the currency of Japan?", "Name a desert in Africa.",
    "What does CPU stand for?", "Who was the first person on the moon?",
    "Name a string instrument.", "What is the capital of Canada?",
    "What is photosynthesis?", "Name three metals.",
    "What is the hardest natural substance?", "Who wrote Romeo and Juliet?",
    "What is the speed of light roughly?", "Name a Nordic country.",
    "What does DNA stand for?", "Give an example of a renewable energy.",
    "What is the smallest prime number?", "Name a famous composer.",
    "What is the main ingredient in bread?", "Translate thank you into German.",
]


@torch.no_grad()
def first_h(p, prompt: str) -> torch.Tensor:
    """Final normed residual at the position that picks the first token."""
    ids = p.tok(prompt, return_tensors="pt").input_ids.to(p.device)
    store = {}

    def hook(module, args, out):
        store["h"] = out[0, -1].detach().float().cpu()

    h = p.model.model.norm.register_forward_hook(hook)
    try:
        p.model(ids)
    finally:
        h.remove()
    return store["h"]


def first_logits(p, prompt: str) -> torch.Tensor:
    ids = p.tok(prompt, return_tensors="pt").input_ids.to(p.device)
    with torch.no_grad():
        return p.model(ids).logits[0, -1].float().cpu()


def opener_ids(p) -> list[int]:
    ids = []
    for o in OPENERS:
        t = p.tok(o, add_special_tokens=False).input_ids
        if t:
            ids.append(t[0])
    return sorted(set(ids))


def main() -> int:
    p = C.load()
    with torch.no_grad():
        p.model.lm_head.weight = torch.nn.Parameter(
            p.model.model.embed_tokens.weight.detach().clone())
    W = p.model.lm_head.weight
    orig = W.detach().clone()

    oids = opener_ids(p)
    names = [p.tok.decode([i]) for i in oids]
    print("loaded", p.n_layers, "layers")
    print("opener rows:", list(zip(oids, names)))

    train = problems(80, 20261120)
    held = problems(60, 20261121)
    print(len(train), "train,", len(held), "held out")

    # math-mode direction
    mh = torch.stack([first_h(p, C.chat_text(p.tok, q, "")) for q, _ in train])
    oh = torch.stack([first_h(p, C.chat_text(p.tok, q, "")) for q in OTHER])
    d = mh.mean(0) - oh.mean(0)
    d = d / d.norm()
    print("math direction norm", round(d.norm().item(), 3),
          "mean cos to math", round(float((mh @ d).mean()), 2),
          "to other", round(float((oh @ d).mean()), 2))

    held_prompts = [C.chat_text(p.tok, q, "") for q, _ in held]
    other_prompts = [C.chat_text(p.tok, q, "") for q in OTHER]

    def opener_rate(prompts):
        k = 0
        for text in prompts:
            lg = first_logits(p, text)
            if int(lg.argmax()) in oids:
                k += 1
        return k / len(prompts)

    print("\nfirst-token argmax is an opener")
    print(f"  base math {opener_rate(held_prompts):.2f}  "
          f"other {opener_rate(other_prompts):.2f}")

    dd = d.to(W.device, W.dtype)
    chosen = None
    print("\nsweep on the first token only")
    for a in ALPHAS:
        with torch.no_grad():
            W.copy_(orig)
            for r in oids:
                W[r] += a * dd
        mr, orr = opener_rate(held_prompts), opener_rate(other_prompts)
        with torch.no_grad():
            W.copy_(orig)
        print(f"  alpha {a:4.1f}: math {mr:.2f}  other {orr:.2f}")
        if chosen is None and mr >= 0.8 and orr <= 0.2:
            chosen = a

    if chosen is None:
        chosen = ALPHAS[-1]
    print("\nchosen alpha", chosen)

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
           C.ROOT / "findings" / "e18_math_mode.json")
    print("\nwrote findings/e18_math_mode.json")
    print("\nthree edited replies:")
    for r in out["edited"]["replies"][:3]:
        print("  " + r.strip()[:220].replace("\n", " | "))
    print("\ntwo edited non-math replies:")
    for r in out["edited"]["other_replies"][:2]:
        print("  " + r.strip()[:160].replace("\n", " | "))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
