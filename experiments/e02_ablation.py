"""E02: causal ablation of the structures found in E01.

E01 found three near-perfect induction heads, two perfect attention sinks, and
massive activations in three layers. None of that proves they are used. This
removes them and measures the damage.

Three measurements:

  perplexity        on a natural text passage, the general "is the model still
                    working" number
  induction acc     on a repeated random sequence, the fraction of second-copy
                    positions where the model correctly predicts the token that
                    followed the previous occurrence. A direct behavioral test of
                    prefix matching, not an attention-pattern proxy.
  shuffled acc      same computation on a non-repeated sequence. Should be near
                    zero. This is the control that proves induction acc means
                    something.

Ablation is zeroing, not scaling: head h is zeroed by clearing its 64 columns at
the o_proj input, a neuron is zeroed by clearing its column at the down_proj
input. Random-head ablation with the same head count is the control.

Run: python experiments/e02_ablation.py
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
from probe.attention import build_repeat  # noqa: E402

HALF = 24

PASSAGE = """The deep ocean is the largest habitat on the planet, yet most of it has never
been seen. Below roughly two hundred meters sunlight fades to nothing, and the
only light left is produced by living things. Animals there have adapted in
strange directions. Some have eyes that point upward, tuned to catch the faint
silhouette of a predator passing overhead. Others have given up on eyes entirely
and read the water through pressure and vibration instead. The pressure itself is
extraordinary. At the bottom of the deepest trench a human body would be crushed
in an instant, and yet fish live there, their cells packed with molecules that
keep proteins from folding wrong. Nothing about this environment is generous, so
every adaptation is a compromise, and the animals that survive are the ones that
made the cheapest trade."""

INDUCTION_HEADS = [(9, 13), (11, 12), (16, 3)]
SINK_HEADS = [(9, 7), (11, 7)]
MASSIVE_LAYERS = [2, 3, 21]
HEAD_DIM = 64


@torch.no_grad()
def perplexity(p, ids):
    out = p.model(ids, labels=ids, attention_mask=torch.ones_like(ids))
    return float(torch.exp(out.loss))


@torch.no_grad()
def induction_accuracy(p, repeat=True):
    """Fraction of second-half positions predicted correctly.

    On a repeated sequence the only way to get these right is to notice the
    repeat and copy what followed last time. On a shuffled sequence the same
    computation should score near zero.
    """
    ids = build_repeat(seed=0, half_len=HALF, repeat=repeat).to(p.device)
    logits = p.model(ids).logits[0]
    correct = 0
    total = 0
    for i in range(HALF, 2 * HALF):
        pred = int(logits[i - 1].argmax())
        correct += int(pred == int(ids[0][i]))
        total += 1
    return correct / total


def main() -> int:
    p = Probe(str(ROOT / "Qwen2.5-0.5B-Instruct"))
    text_ids = p.encode(PASSAGE)
    print(f"passage: {text_ids.shape[1]} tokens\n")

    base_ppl = perplexity(p, text_ids)
    base_ind = induction_accuracy(p, repeat=True)
    base_shuf = induction_accuracy(p, repeat=False)
    print(f"baseline          ppl={base_ppl:9.4f}  induction={base_ind:.3f}  "
          f"shuffled={base_shuf:.3f}\n")

    results = {"baseline": {"ppl": base_ppl, "induction": base_ind,
                            "shuffled": base_shuf}}

    def run(label, **kw):
        with ablate(p, **kw):
            ppl = perplexity(p, text_ids)
            ind = induction_accuracy(p, repeat=True)
            shuf = induction_accuracy(p, repeat=False)
        print(f"{label:<18} ppl={ppl:9.4f} ({ppl / base_ppl:5.2f}x)  "
              f"induction={ind:.3f}  shuffled={shuf:.3f}")
        results[label] = {"ppl": ppl, "ratio": ppl / base_ppl,
                          "induction": ind, "shuffled": shuf}

    print("== head ablations ==")
    run("induction heads", heads=INDUCTION_HEADS)
    run("sink heads", heads=SINK_HEADS)
    run("both", heads=INDUCTION_HEADS + SINK_HEADS)

    rng = np.random.default_rng(0)
    rand_heads = [(int(rng.integers(0, 24)), int(rng.integers(0, 14))) for _ in range(3)]
    print(f"\nrandom control heads: {rand_heads}")
    run("random 3 heads", heads=rand_heads)

    # ---- massive channels ----
    print("\n== massive activation channels ==")
    cap = p.capture(text_ids)
    massive = {}
    for li, act in enumerate(cap["mlp_act"]):
        a = act[0].numpy()
        peak = np.abs(a).max(axis=0)
        top = np.argsort(-peak)[:5]
        massive[li] = [(int(c), float(peak[c])) for c in top]

    for li in MASSIVE_LAYERS:
        chans = massive[li]
        print(f"  layer {li}: " + ", ".join(f"ch{c}={v:.0f}" for c, v in chans))

    # stability: do the same channels dominate on a different passage?
    other = p.encode(PASSAGE[::-1])
    cap2 = p.capture(other)
    print("\n  stability across a different input:")
    stable = {}
    for li in MASSIVE_LAYERS:
        a2 = cap2["mlp_act"][li][0].numpy()
        peak2 = np.abs(a2).max(axis=0)
        top2 = set(int(c) for c in np.argsort(-peak2)[:5])
        top1 = set(c for c, _ in massive[li])
        overlap = len(top1 & top2)
        stable[li] = overlap
        print(f"    layer {li}: {overlap}/5 top channels identical")

    print()
    for li in MASSIVE_LAYERS:
        cols = [c for c, _ in massive[li]]
        run(f"massive L{li}", channels=[(li, cols)])

    run("massive all", channels=[(li, [c for c, _ in massive[li]]) for li in MASSIVE_LAYERS])

    # random neuron control: same count, different channels
    print()
    rand_chans = []
    for li in MASSIVE_LAYERS:
        cols = rng.choice(p.inter, size=5, replace=False).tolist()
        rand_chans.append((li, cols))
    run("random neurons", channels=rand_chans)

    save_json({"passage_tokens": int(text_ids.shape[1]),
               "massive_channels": {str(k): v for k, v in massive.items()},
               "stability": {str(k): v for k, v in stable.items()},
               "results": results},
              ROOT / "findings" / "e02_ablation.json")

    print("\nwrote findings/e02_ablation.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
