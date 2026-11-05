"""E06: what are the layer-0 heads that dominate induction?

E05 found that the largest single contributors to the induction margin are heads
in layer 0, not the mid-network heads that E01 flagged. Layer 0 attends directly
to token embeddings, so this asks what offset they look at.

For each candidate head this reports the mean attention weight at relative
offsets: -1 is the previous token, -2 the one before, and 0 the first token
(the sink position).

Run: python experiments/e06_layer0.py
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
from probe.attention import head_scores  # noqa: E402

HALF = 64
CANDIDATES = [(0, 8), (0, 3), (0, 2), (2, 5), (8, 7), (15, 2), (10, 2),
              (9, 13), (11, 12), (16, 3)]


def main() -> int:
    p = Probe(str(ROOT / "Qwen2.5-0.5B-Instruct"))
    rng = np.random.default_rng(0)
    toks = rng.integers(1000, 100_000, size=HALF).tolist()
    ids = torch.tensor([toks + toks]).to(p.device)

    cap = p.capture(ids)
    scores = [head_scores(a, HALF) for a in cap["attentions"]]

    print(f"input {2*HALF} tokens, two copies of {HALF} random tokens")
    print("offsets are relative to the query position\n")

    header = f"{'head':<10}{'prev':>7}{'ind':>7}{'sink':>7}{'ent':>7}"
    header += "".join(f"{f'@{k}':>7}" for k in (1, 2, 3, 4))
    print(header)

    rows = []
    for li, h in CANDIDATES:
        a = cap["attentions"][li][0].numpy()[h]  # [Q, K]
        s = scores[li]
        off = {}
        for k in (1, 2, 3, 4):
            vals = [a[i, i - k] for i in range(k, 2 * HALF)]
            off[k] = float(np.mean(vals))
        line = (f"L{li}H{h:<8}{s['prev'][h]:>7.3f}{s['induction'][h]:>7.3f}"
                f"{s['sink'][h]:>7.3f}{s['entropy'][h]:>7.3f}")
        line += "".join(f"{off[k]:>7.3f}" for k in (1, 2, 3, 4))
        print(line)
        rows.append({"head": f"L{li}H{h}", "prev": float(s["prev"][h]),
                     "induction": float(s["induction"][h]),
                     "sink": float(s["sink"][h]), "entropy": float(s["entropy"][h]),
                     "offsets": {str(k): v for k, v in off.items()}})

    # what does a layer-0 head do to the residual stream?
    print("\n== contribution of layer-0 heads to the residual stream ==")
    print(f"{'head':<10}{'output norm':>14}{'as pct of stream':>18}")
    cap0 = cap
    stream = cap0["resid_in"][1][0]  # stream entering layer 1
    base_norm = float(stream.norm(dim=-1).mean())
    for li, h in [(0, 8), (0, 3), (0, 2), (0, 1)]:
        # recompute this head's output by zeroing it and differencing the stream
        from probe.ablate import ablate
        with ablate(p, heads=[(li, h)]):
            cap2 = p.capture(ids)
        new_norm = float(cap2["resid_in"][1][0].norm(dim=-1).mean())
        delta = abs(new_norm - base_norm)
        print(f"L{li}H{h:<8}{delta:>14.4f}{100 * delta / base_norm:>17.2f}%")

    save_json({"half": HALF, "rows": rows,
               "stream_norm_into_layer1": base_norm},
              ROOT / "findings" / "e06_layer0.json")
    print("\nwrote findings/e06_layer0.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
