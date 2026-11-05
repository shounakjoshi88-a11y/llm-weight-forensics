"""E07: separate general importance from induction-specific importance.

E04 measured each head's effect on perplexity. E05 measured each head's effect on
the induction margin. A head can score high on both simply by being generally
load-bearing, which is what the layer-0 heads turned out to be.

This joins the two sweeps and computes specificity:

    z(induction damage) - z(perplexity damage)

A head with high specificity hurts in-context copying more than it hurts ordinary
language modeling, which is the signature of a real induction component.

Reads the two saved sweeps, so it needs no GPU and no model.

Run: python experiments/e07_specificity.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def load(name):
    return json.loads((ROOT / "findings" / name).read_text(encoding="utf-8"))


def main() -> int:
    ppl = np.array(load("e04_head_importance.json")["importance"], dtype=float)
    ind = -np.array(load("e05_induction_importance.json")["grid"], dtype=float)
    # ind is now "damage": positive means removal hurt

    def z(a):
        return (a - a.mean()) / (a.std() + 1e-12)

    spec = z(ind) - z(ppl)

    flat = [(float(spec[li, h]), float(ind[li, h]), float(ppl[li, h]), li, h)
            for li in range(spec.shape[0]) for h in range(spec.shape[1])]
    flat.sort(reverse=True)

    print("== most induction-specific heads ==")
    print(f"{'rank':>5}{'layer':>7}{'head':>6}{'specificity':>13}"
          f"{'ind damage':>12}{'ppl damage':>12}")
    for r, (s, di, dp, li, h) in enumerate(flat[:15], 1):
        print(f"{r:>5}{li:>7}{h:>6}{s:>+13.2f}{di:>+12.3f}{dp:>+12.3f}")

    print("\n== most language-modeling-specific heads ==")
    for s, di, dp, li, h in flat[-8:]:
        print(f"{'':>5}{li:>7}{h:>6}{s:>+13.2f}{di:>+12.3f}{dp:>+12.3f}")

    print("\n== the E01 induction heads, by specificity ==")
    rank = {(li, h): r for r, (_, _, _, li, h) in enumerate(flat, 1)}
    for li, h in [(9, 13), (11, 12), (16, 3)]:
        print(f"  L{li}H{h}: specificity {spec[li, h]:+.2f}, "
              f"rank {rank[(li, h)]} of {spec.size}, "
              f"ind {ind[li, h]:+.3f}, ppl {ppl[li, h]:+.3f}")

    # how much of the variance is shared?
    r = float(np.corrcoef(ppl.ravel(), ind.ravel())[0, 1])
    print(f"\ncorrelation between ppl damage and induction damage: {r:+.3f}")
    print(f"shared variance: {100 * r * r:.1f}%")
    print(f"so {100 * (1 - r * r):.1f}% of the variation is task-specific")

    print("\n== layer summary ==")
    print(f"{'layer':>6}{'ind damage':>12}{'ppl damage':>12}{'specificity':>13}")
    for li in range(spec.shape[0]):
        print(f"{li:>6}{ind[li].sum():>+12.2f}{ppl[li].sum():>+12.2f}"
              f"{spec[li].mean():>+13.2f}")

    out = {"correlation": r, "r2": r * r,
           "top_specific": [{"spec": s, "ind": di, "ppl": dp, "layer": li, "head": h}
                            for s, di, dp, li, h in flat[:15]],
           "induction_heads": {f"L{li}H{h}": {"spec": float(spec[li, h]),
                                              "ind": float(ind[li, h]),
                                              "ppl": float(ppl[li, h]),
                                              "rank": rank[(li, h)]}
                               for li, h in [(9, 13), (11, 12), (16, 3)]}}
    (ROOT / "findings" / "e07_specificity.json").write_text(
        json.dumps(out, indent=2), encoding="utf-8")
    print("\nwrote findings/e07_specificity.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
