"""E29: does the readout gain survive when the leading digit is not constant?

E28 confirmed the closed-form digit-row correction on fresh 5x5 addition:
116/200 to 128/200, +14 -2, p = 0.004. But the fit and the test were both
restricted to 6-digit sums, and a 6-digit sum of two 5-digit numbers always
starts with 1. So the readout could be winning by learning a constant, which is
an artifact of the restriction and would not transfer.

This removes the restriction. Problems are grouped by answer length, the
residuals are captured with the correct length for each group, and the fit is
pooled over 5-digit and 6-digit sums. The held-out set is scored overall and
split by length, so it is visible exactly how much of the gain was the constant.

If the gain holds on 5-digit sums, where the leading digit ranges over 2 to 9,
it is a real readout improvement and not a restriction artifact.
"""

from __future__ import annotations

import sys
from math import comb
from pathlib import Path

import torch

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C
from e26_digit_trajectory import forward_capture
from e27_readout_refit import fit_readout

N_TRAIN = 800
N_TEST = 300
LAM = 100.0
BETA = 0.25
MAX_NEW = 12


def mcnemar(b: int, c: int) -> float:
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    return min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / (2 ** n))


def collect(p, digit_id, rows):
    """Pool residuals and labels over answer lengths."""
    by_len: dict[int, list] = {}
    for r in rows:
        by_len.setdefault(len(str(r["want"])), []).append(r)
    X, y = [], []
    for nd, group in sorted(by_len.items()):
        pref = [C.chat_text(p.tok, *C.prompt(r, "answer")) for r in group]
        texts = [s + " " + str(r["want"]) for s, r in zip(pref, group)]
        R, _ = forward_capture(p, texts, nd, digit_id)
        X.append(torch.cat([R[i] for i in range(nd)], 0))
        y.append(torch.cat([torch.tensor([int(str(r["want"])[i]) for r in group])
                            for i in range(nd)], 0))
    return torch.cat(X, 0), torch.cat(y, 0)


def main() -> int:
    p = C.load()
    with torch.no_grad():
        p.model.lm_head.weight = torch.nn.Parameter(
            p.model.model.embed_tokens.weight.detach().clone())
    W = p.model.lm_head.weight
    orig = W.detach().clone()
    digit_id = [p.tok(str(d), add_special_tokens=False).input_ids[0]
                for d in range(10)]
    print("loaded", p.n_layers, "layers")

    tr = [r for r in C.make("add", 5, 5, 6000, 20261201)
          if r["split"] == "train"][:N_TRAIN]
    print("train lengths:", {k: sum(1 for r in tr if len(str(r["want"])) == k)
                             for k in (5, 6)})
    X, y = collect(p, digit_id, tr)
    Wf = fit_readout(X, y, LAM)
    g = torch.Generator().manual_seed(13)
    Ws = fit_readout(X, y[torch.randperm(len(y), generator=g)], LAM)

    orig_d = orig[digit_id].detach().float().cpu().double()
    per_digit = (X @ orig_d.T).std(0) / (X @ Wf.T).std(0).clamp_min(1e-9)
    Wf = Wf * per_digit[:, None]
    Ws = Ws * per_digit[:, None]
    base_row = orig_d

    te = [r for r in C.make("add", 5, 5, 6000, 20261210)
          if r["split"] == "heldout"][:N_TEST]
    prompts = [C.chat_text(p.tok, *C.prompt(r, "answer")) for r in te]
    wants = [r["want"] for r in te]
    lens = [len(str(w)) for w in wants]
    print(f"test: {len(te)} problems, "
          f"{lens.count(5)} five-digit, {lens.count(6)} six-digit")

    def install(mat):
        with torch.no_grad():
            W.copy_(orig)
            if mat is not None:
                W[digit_id] = mat.to(device=W.device, dtype=W.dtype)

    conds = [("base", None),
             ("beta=0.25", base_row + BETA * (Wf - base_row)),
             ("shuffled 0.25", base_row + BETA * (Ws - base_row))]

    results = {}
    base_ok = None
    print(f"\n  {'condition':16s} {'all':>12s} {'5-digit':>10s} {'6-digit':>10s}")
    for label, mat in conds:
        install(mat)
        reps = C.generate_batch(p, prompts, max_new_tokens=MAX_NEW, batch_size=40)
        install(None)
        ok = [C.last_int(r) == w for r, w in zip(reps, wants)]
        if base_ok is None:
            base_ok = ok
        b = sum(1 for x, y2 in zip(base_ok, ok) if y2 and not x)
        c = sum(1 for x, y2 in zip(base_ok, ok) if x and not y2)
        pv = mcnemar(b, c)
        s5 = [o for o, L in zip(ok, lens) if L == 5]
        s6 = [o for o, L in zip(ok, lens) if L == 6]
        print(f"  {label:16s} {sum(ok):4d}/{len(ok):<3d} {sum(s5):5d}/{len(s5):<4d} "
              f"{sum(s6):5d}/{len(s6):<4d}   +{b} -{c} p={pv:.3f}")
        results[label] = {"exact": sum(ok) / len(ok),
                          "five": sum(s5) / len(s5), "six": sum(s6) / len(s6),
                          "fixed": b, "broken": c, "p": pv, "replies": reps}

    results["meta"] = {"n_train": len(tr), "n_test": len(te),
                       "lam": LAM, "beta": BETA}
    C.save(results, C.ROOT / "findings" / "e29_unrestricted.json")
    print("\nwrote findings/e29_unrestricted.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
