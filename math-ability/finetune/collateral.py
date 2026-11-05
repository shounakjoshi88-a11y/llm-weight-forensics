"""Did the arithmetic fine-tune damage anything outside arithmetic?

Exact-match on held-out cells only says the model got better at the target. It
says nothing about whether 17.6M adapter parameters turned the model into a
one-trick arithmetic parrot. Two checks:

1. Perplexity on fixed English prose the arithmetic corpus never contained.
   Catastrophic forgetting shows up here first.
2. A handful of non-arithmetic prompts, base against tuned, read side by side.

Usage: python finetune/collateral.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments"))
import common as C

ADAPTER = C.ROOT / "finetune" / "adapter"

PROSE = """
The lighthouse had stood on the point for a hundred and forty years, and in that
time it had outlived four keepers, two storms that took the roof off the shed,
and one attempt by the county to demolish it. The county lost. The lighthouse
remained, white and slightly crooked, leaning into the wind as if it had decided
long ago that standing straight was a young building's vanity.

In the spring the gulls came back and nested in the cracked mortar, and the man
who owned the place, whose name was Hollis, would climb the ninety-two steps
twice a day to check the lamp. He did not need to. The lamp had been automatic
since 1974. But Hollis had a theory that a thing left unwatched begins to
disappear, first from memory and then from the world, and he was not willing to
test the theory on the lighthouse.

The town below had changed around the point. The cannery closed in the eighties
and the pier was rebuilt for tourists, who arrived in summer with coolers and
dogs and a vague expectation of scenery. They took photographs of the lighthouse
from the parking lot and rarely walked the half mile to the door. Hollis did not
mind. He had never been much for company, and the lighthouse, whatever else it
was, was company enough.

There is a particular quality of light on the coast in late afternoon, when the
sun is low and the haze turns the water the colour of old brass. Hollis called
it the quiet hour. The wind would drop, the gulls would settle, and for a few
minutes the whole point would go still, as though the land were holding its
breath before the dark. He had tried to explain it to people and had never
managed. It was not something you explained. It was something you stood in.

The storm that finally did for the shed came in October, out of the northeast,
with a pressure so low the barometer in the kitchen gave up and pointed at the
floor. Hollis watched from the tower as the sea came over the rocks in sheets
and the shed's roof lifted, hung, and went. In the morning he found the roof in
the marsh a quarter mile inland, upright and undamaged, sitting in the reeds
like a boat that had decided to retire.
""".strip()

PROMPTS = [
    ("Who wrote Pride and Prejudice?", 60),
    ("Explain in one sentence why the sky is blue.", 80),
    ("What is the capital of Australia?", 40),
    ("Name three rivers in Europe.", 60),
    ("Write a haiku about winter.", 80),
    ("What does HTTP stand for?", 50),
    ("Give one reason a company might prefer a relational database over a document store.", 100),
    ("Translate 'good morning' into French.", 40),
    ("What is the difference between a list and a tuple in Python?", 100),
    ("Summarise the plot of Romeo and Juliet in two sentences.", 100),
    ("Is a tomato a fruit or a vegetable?", 80),
    ("What year did the Berlin Wall fall?", 40),
]


@torch.no_grad()
def perplexity(p) -> float:
    ids = p.tok(PROSE, return_tensors="pt").input_ids.to(p.model.device)
    out = p.model(input_ids=ids, labels=ids)
    return float(torch.exp(out.loss))


def main() -> int:
    p = C.load()
    base_ppl = perplexity(p)
    base_replies = C.generate_batch(
        p, [C.chat_text(p.tok, q, "") for q, _ in PROMPTS],
        max_new_tokens=max(n for _, n in PROMPTS), batch_size=6)

    from peft import PeftModel
    p.model = PeftModel.from_pretrained(p.model, str(ADAPTER))
    p.model.eval()
    tuned_ppl = perplexity(p)
    tuned_replies = C.generate_batch(
        p, [C.chat_text(p.tok, q, "") for q, _ in PROMPTS],
        max_new_tokens=max(n for _, n in PROMPTS), batch_size=6)

    print(f"prose perplexity   base {base_ppl:8.3f}   tuned {tuned_ppl:8.3f}   "
          f"ratio {tuned_ppl / base_ppl:.2f}x\n")
    for (q, _), b, t in zip(PROMPTS, base_replies, tuned_replies):
        print(f"--- {q}")
        print(f"  base : {b.strip()[:200].replace(chr(10), ' | ')}")
        print(f"  tuned: {t.strip()[:200].replace(chr(10), ' | ')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
