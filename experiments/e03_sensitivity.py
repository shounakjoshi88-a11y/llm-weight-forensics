"""E03: sharpening the two ambiguous results from E02.

E02 produced two things that cannot be reported as findings yet.

1. Ablating the induction heads changed induction accuracy by exactly zero. That
   is either redundancy or a test too coarse to see the effect. Accuracy over 24
   positions is a blunt instrument, so this replaces it with a logit margin over
   64 positions, and adds ablation sanity checks to prove the hooks work at all.

2. The massive activation channels were only 2/5 to 3/5 stable across two inputs.
   Set overlap is the wrong question. The right question is whether the single
   top channel is fixed across many inputs, which would mean it acts as a bias
   term rather than encoding content.

Run: python experiments/e03_sensitivity.py
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

PASSAGES = [
    "The deep ocean is the largest habitat on the planet, yet most of it has never "
    "been seen. Below roughly two hundred meters sunlight fades to nothing, and the "
    "only light left is produced by living things. Animals there have adapted in "
    "strange directions.",
    "A compiler translates source code into machine instructions. It first parses "
    "the text into a tree, then checks that the tree is well formed, then rewrites "
    "it into simpler forms before emitting instructions for the target processor.",
    "The stock market fell sharply on Tuesday after a major bank reported losses "
    "larger than analysts expected. Investors moved money into government bonds, "
    "and the central bank signalled it would consider cutting rates.",
    "Photosynthesis converts light into chemical energy. Chlorophyll absorbs red and "
    "blue wavelengths while reflecting green, which is why most leaves appear green "
    "to the human eye during the growing season.",
    "The treaty was signed in a small room with no windows. Neither delegation "
    "expected the talks to succeed, and both had already drafted statements "
    "explaining the failure to their governments and their newspapers.",
]

INDUCTION_HEADS = [(9, 13), (11, 12), (16, 3)]
PREV_HEADS = [(8, 11), (15, 6), (1, 12)]


def build_repeat(half=HALF, seed=0, repeat=True):
    rng = np.random.default_rng(seed)
    toks = rng.integers(1000, 100_000, size=half).tolist()
    second = toks if repeat else toks[::-1]
    return torch.tensor([toks + second])


@torch.no_grad()
def perplexity(p, ids):
    out = p.model(ids, labels=ids, attention_mask=torch.ones_like(ids))
    return float(torch.exp(out.loss))


@torch.no_grad()
def induction_margin(p, half=HALF, repeat=True):
    """Mean logit margin for the copied token, over the second copy.

    Margin, not accuracy: it keeps moving after accuracy saturates, so it can
    detect damage that argmax hides.
    """
    ids = build_repeat(half=half, repeat=repeat).to(p.device)
    logits = p.model(ids).logits[0]
    margins, correct, total = [], 0, 0
    for i in range(half, 2 * half):
        row = logits[i - 1]
        target = int(ids[0][i])
        t = float(row[target])
        other = row.clone()
        other[target] = -1e9
        margins.append(t - float(other.max()))
        correct += int(int(row.argmax()) == target)
        total += 1
    return float(np.mean(margins)), correct / total


def main() -> int:
    p = Probe(str(ROOT / "Qwen2.5-0.5B-Instruct"))
    ids = p.encode(PASSAGES[0])
    print(f"passage: {ids.shape[1]} tokens\n")

    res = {}

    # ---------- A. do the hooks work at all ----------
    print("== A. ablation sanity: does zeroing do anything ==")
    base = perplexity(p, ids)
    print(f"{'condition':<26}{'ppl':>12}{'ratio':>9}")

    def ppl_row(label, **kw):
        with ablate(p, **kw):
            v = perplexity(p, ids)
        print(f"{label:<26}{v:>12.3f}{v / base:>8.2f}x")
        res[label] = v
        return v

    print(f"{'baseline':<26}{base:>12.3f}{1.0:>8.2f}x")
    ppl_row("all heads in layer 9", heads=[(9, h) for h in range(p.n_heads)])
    ppl_row("all heads in all layers",
            heads=[(li, h) for li in range(p.n_layers) for h in range(p.n_heads)])
    ppl_row("all MLP in layer 9", channels=[(9, list(range(p.inter)))])

    # ---------- B. sensitive induction test ----------
    print("\n== B. induction with a logit margin (64 positions) ==")
    m0, a0 = induction_margin(p)
    ms0, _ = induction_margin(p, repeat=False)
    print(f"baseline        margin={m0:+8.3f}  acc={a0:.3f}   "
          f"shuffled margin={ms0:+8.3f}\n")
    print(f"{'condition':<26}{'margin':>10}{'delta':>9}{'acc':>8}")

    ind_res = {"baseline": {"margin": m0, "acc": a0, "shuffled_margin": ms0}}

    def ind_row(label, **kw):
        with ablate(p, **kw):
            m, a = induction_margin(p)
        print(f"{label:<26}{m:>+10.3f}{m - m0:>+9.3f}{a:>8.3f}")
        ind_res[label] = {"margin": m, "delta": m - m0, "acc": a}
        return m

    for li, h in INDUCTION_HEADS:
        ind_row(f"induction L{li}H{h}", heads=[(li, h)])
    ind_row("all 3 induction heads", heads=INDUCTION_HEADS)

    for li, h in PREV_HEADS:
        ind_row(f"prev-token L{li}H{h}", heads=[(li, h)])
    ind_row("all 3 prev-token heads", heads=PREV_HEADS)
    ind_row("induction + prev", heads=INDUCTION_HEADS + PREV_HEADS)

    rng = np.random.default_rng(1)
    rand = [(int(rng.integers(0, 24)), int(rng.integers(0, 14))) for _ in range(6)]
    ind_row("random 6 heads", heads=rand)

    # ---------- C. massive channel stability ----------
    print("\n== C. is the top massive channel fixed across inputs ==")
    print(f"{'layer':>6}{'input':>7}{'top channel':>13}{'peak':>10}")
    stability = {}
    for li in (2, 3, 21):
        tops = []
        for k, text in enumerate(PASSAGES):
            cap = p.capture(p.encode(text))
            a = cap["mlp_act"][li][0].numpy()
            peak = np.abs(a).max(axis=0)
            c = int(np.argmax(peak))
            tops.append(c)
            print(f"{li:>6}{k:>7}{c:>13}{peak[c]:>10.1f}")
        uniq, counts = np.unique(tops, return_counts=True)
        dominant = int(uniq[np.argmax(counts)])
        stability[li] = {"tops": tops, "dominant": dominant,
                         "agreement": int(counts.max())}
        print(f"       dominant channel {dominant}, {counts.max()}/{len(tops)} inputs\n")

    save_json({"ablation_sanity": res, "induction": ind_res,
               "massive_stability": {str(k): v for k, v in stability.items()}},
              ROOT / "findings" / "e03_sensitivity.json")

    print("wrote findings/e03_sensitivity.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
