"""E04: head importance by exhaustive ablation.

E03 showed the top-3 induction heads are nearly irrelevant to behavior while
random heads are not. The obvious explanation is that some other heads matter
far more. This measures all 336 heads (24 layers x 14 heads) one at a time and
builds an importance map.

Importance is the perplexity increase caused by zeroing that single head on a
150-token passage. Absolute increase, so a head that matters only because the
model is already fragile is not flattered.

The map is then correlated against the E01 attention scores, to test the claim
"heads that look like induction heads are the ones that matter".

Run: python experiments/e04_head_importance.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from probe import Probe, save_json  # noqa: E402
from probe.ablate import ablate  # noqa: E402
from probe.attention import build_repeat, head_scores  # noqa: E402

PASSAGE = """The deep ocean is the largest habitat on the planet, yet most of it has never
been seen. Below roughly two hundred meters sunlight fades to nothing, and the
only light left is produced by living things. Animals there have adapted in
strange directions. Some have eyes that point upward, tuned to catch the faint
silhouette of a predator passing overhead. A compiler translates source code into
machine instructions. It first parses the text into a tree, then checks that the
tree is well formed, then rewrites it into simpler forms before emitting
instructions for the target processor. The stock market fell sharply on Tuesday
after a major bank reported losses larger than analysts expected. Investors moved
money into government bonds, and the central bank signalled it would consider
cutting rates."""


@torch.no_grad()
def perplexity(p, ids):
    out = p.model(ids, labels=ids, attention_mask=torch.ones_like(ids))
    return float(torch.exp(out.loss))


def main() -> int:
    p = Probe(str(ROOT / "Qwen2.5-0.5B-Instruct"))
    ids = p.encode(PASSAGE)
    base = perplexity(p, ids)
    print(f"passage {ids.shape[1]} tokens | baseline ppl {base:.4f}")
    print(f"sweeping {p.n_layers * p.n_heads} heads\n")

    grid = np.zeros((p.n_layers, p.n_heads))
    for li in range(p.n_layers):
        for h in range(p.n_heads):
            with ablate(p, heads=[(li, h)]):
                grid[li, h] = perplexity(p, ids) - base
        if li % 4 == 0:
            print(f"  layer {li:2d} done", flush=True)

    flat = [(float(grid[li, h]), li, h)
            for li in range(p.n_layers) for h in range(p.n_heads)]
    flat.sort(reverse=True)

    print("\n== 20 most important heads ==")
    print(f"{'rank':>5}{'layer':>7}{'head':>6}{'ppl delta':>12}")
    for r, (d, li, h) in enumerate(flat[:20], 1):
        print(f"{r:>5}{li:>7}{h:>6}{d:>12.3f}")

    print("\n== 10 least important heads ==")
    for d, li, h in flat[-10:]:
        print(f"{'':>5}{li:>7}{h:>6}{d:>12.3f}")

    # ---- correlate with the E01 attention scores ----
    cap = p.capture(build_repeat(half_len=24).to(p.device))
    scores = [head_scores(a, 24) for a in cap["attentions"]]

    def stack(key):
        return np.stack([s[key] for s in scores])  # [L, H]

    ind = stack("induction")
    sink = stack("sink")
    prev = stack("prev")
    ent = stack("entropy")

    def corr(a, b):
        a = a.ravel() - a.mean()
        b = b.ravel() - b.mean()
        den = np.sqrt((a * a).sum() * (b * b).sum())
        return float((a * b).sum() / den) if den else float("nan")

    print("\n== correlation with E01 attention scores ==")
    print(f"  importance vs induction score : {corr(grid, ind):+.3f}")
    print(f"  importance vs sink score      : {corr(grid, sink):+.3f}")
    print(f"  importance vs prev-token score: {corr(grid, prev):+.3f}")
    print(f"  importance vs entropy         : {corr(grid, ent):+.3f}")

    # where do the known induction heads rank?
    print("\n== rank of the top induction heads ==")
    rank_of = {(li, h): r for r, (d, li, h) in enumerate(flat, 1)}
    for li, h in [(9, 13), (11, 12), (16, 3)]:
        print(f"  L{li}H{h}: rank {rank_of[(li, h)]} of {len(flat)}, "
              f"delta {grid[li, h]:+.3f}")

    print("\n== layer totals ==")
    for li in range(p.n_layers):
        print(f"  L{li:2d} total {grid[li].sum():>8.2f}  max {grid[li].max():>7.3f}")

    save_json({"baseline_ppl": base,
               "importance": grid.tolist(),
               "top20": [{"delta": d, "layer": li, "head": h} for d, li, h in flat[:20]],
               "rank_of_induction_heads": {f"L{li}H{h}": rank_of[(li, h)]
                                           for li, h in [(9, 13), (11, 12), (16, 3)]},
               "correlations": {"induction": corr(grid, ind), "sink": corr(grid, sink),
                                "prev": corr(grid, prev), "entropy": corr(grid, ent)}},
              ROOT / "findings" / "e04_head_importance.json")

    print("\nwrote findings/e04_head_importance.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
