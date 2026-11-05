"""E05: which heads actually drive induction.

E04 ranked heads by perplexity on prose. That is the wrong task for an induction
head, so a low rank there does not disprove the circuit. This ranks all 336 heads
by their effect on the induction margin instead, which is the task the circuit is
supposed to perform.

If the attention-based detection in E01 is sound, L9H13, L11H12 and L16H3 should
rank near the top here. If they do not, attention pattern and function are
genuinely different things in this model.

Run: python experiments/e05_induction_importance.py
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

HALF = 64


def build_repeat(half=HALF, seed=0):
    rng = np.random.default_rng(seed)
    toks = rng.integers(1000, 100_000, size=half).tolist()
    return torch.tensor([toks + toks])


@torch.no_grad()
def induction_margin(p, ids):
    logits = p.model(ids).logits[0]
    margins = []
    for i in range(HALF, 2 * HALF):
        row = logits[i - 1]
        target = int(ids[0][i])
        t = float(row[target])
        other = row.clone()
        other[target] = -1e9
        margins.append(t - float(other.max()))
    return float(np.mean(margins))


def main() -> int:
    p = Probe(str(ROOT / "Qwen2.5-0.5B-Instruct"))
    ids = build_repeat().to(p.device)
    base = induction_margin(p, ids)
    print(f"baseline induction margin {base:+.3f}")
    print(f"sweeping {p.n_layers * p.n_heads} heads on the induction task\n")

    grid = np.zeros((p.n_layers, p.n_heads))
    for li in range(p.n_layers):
        for h in range(p.n_heads):
            with ablate(p, heads=[(li, h)]):
                grid[li, h] = induction_margin(p, ids) - base
        if li % 4 == 0:
            print(f"  layer {li:2d} done", flush=True)

    flat = [(float(grid[li, h]), li, h)
            for li in range(p.n_layers) for h in range(p.n_heads)]
    flat.sort()

    print("\n== 20 heads whose removal hurts induction most ==")
    print(f"{'rank':>5}{'layer':>7}{'head':>6}{'margin delta':>14}")
    for r, (d, li, h) in enumerate(flat[:20], 1):
        print(f"{r:>5}{li:>7}{h:>6}{d:>+14.3f}")

    rank_of = {(li, h): r for r, (d, li, h) in enumerate(flat, 1)}
    print("\n== where the E01 induction heads land ==")
    for li, h in [(9, 13), (11, 12), (16, 3)]:
        print(f"  L{li}H{h}: rank {rank_of[(li, h)]} of {len(flat)}, "
              f"delta {grid[li, h]:+.3f}")

    print("\n== where the E01 sink heads land ==")
    for li, h in [(9, 7), (11, 7)]:
        print(f"  L{li}H{h}: rank {rank_of[(li, h)]} of {len(flat)}, "
              f"delta {grid[li, h]:+.3f}")

    # ---- does ablation compound, or is the circuit redundant? ----
    print("\n== cumulative ablation of the top-K induction-critical heads ==")
    print(f"{'K':>4}{'margin':>11}{'delta':>10}")
    cum = {}
    for K in (1, 2, 3, 5, 8, 12):
        heads = [(li, h) for _, li, h in flat[:K]]
        with ablate(p, heads=heads):
            m = induction_margin(p, ids)
        print(f"{K:>4}{m:>+11.3f}{m - base:>+10.3f}")
        cum[K] = {"margin": m, "delta": m - base, "heads": heads}

    # how much total signal is there?
    pos = float(grid[grid > 0].sum())
    neg = float(grid[grid < 0].sum())
    print(f"\nsum of positive deltas {pos:+.1f} over {int((grid > 0).sum())} heads")
    print(f"sum of negative deltas {neg:+.1f} over {int((grid < 0).sum())} heads")

    print("\n== per-layer total effect on induction ==")
    for li in range(p.n_layers):
        print(f"  L{li:2d} {grid[li].sum():>+8.2f}  max hurt {grid[li].min():>+7.3f}")

    save_json({"baseline": base, "grid": grid.tolist(),
               "top20": [{"delta": d, "layer": li, "head": h} for d, li, h in flat[:20]],
               "rank_of": {f"L{li}H{h}": rank_of[(li, h)]
                           for li, h in [(9, 13), (11, 12), (16, 3), (9, 7), (11, 7)]},
               "cumulative": {str(k): v for k, v in cum.items()},
               "pos_sum": pos, "neg_sum": neg},
              ROOT / "findings" / "e05_induction_importance.json")

    print("\nwrote findings/e05_induction_importance.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
