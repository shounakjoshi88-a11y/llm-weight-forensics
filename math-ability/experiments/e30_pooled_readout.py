"""E30: one readout for both operations.

E29 confirmed the digit-row correction on 5x5 addition, unrestricted:
180/300 to 195/300, +18 -3, p = 0.001, and it held on 5-digit sums where the
leading digit is not constant. But the fit was addition only, and the ten digit
rows are shared, so E28 saw a small loss on 3x2 multiplication.

This fits on addition and multiplication pooled, then scores fresh held-out sets
of both. The question is whether a single closed-form readout can serve both
operations, or whether the two tasks want different digit rows and the gain is
inherently a trade.
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

N_TRAIN = 700
N_TEST = 300
LAM = 100.0
BETAS = [0.1, 0.2, 0.3]
MAX_NEW = 12


def mcnemar(b: int, c: int) -> float:
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    return min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / (2 ** n))


def collect(p, digit_id, rows):
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

    add_tr = [r for r in C.make("add", 5, 5, 6000, 20261201)
              if r["split"] == "train"][:N_TRAIN]
    mul_tr = [r for r in C.make("mul", 3, 2, 6000, 20261201)
              if r["split"] == "train"][:N_TRAIN]
    print(f"train: {len(add_tr)} addition, {len(mul_tr)} multiplication")

    Xa, ya = collect(p, digit_id, add_tr)
    Xm, ym = collect(p, digit_id, mul_tr)
    X = torch.cat([Xa, Xm], 0)
    y = torch.cat([ya, ym], 0)

    Wf = fit_readout(X, y, LAM)
    g = torch.Generator().manual_seed(13)
    Ws = fit_readout(X, y[torch.randperm(len(y), generator=g)], LAM)
    orig_d = orig[digit_id].detach().float().cpu().double()
    per_digit = (X @ orig_d.T).std(0) / (X @ Wf.T).std(0).clamp_min(1e-9)
    Wf = Wf * per_digit[:, None]
    Ws = Ws * per_digit[:, None]
    base_row = orig_d

    def install(mat):
        with torch.no_grad():
            W.copy_(orig)
            if mat is not None:
                W[digit_id] = mat.to(device=W.device, dtype=W.dtype)

    sets = {
        "add5x5": [r for r in C.make("add", 5, 5, 6000, 20261210)
                   if r["split"] == "heldout"][:N_TEST],
        "mul3x2": [r for r in C.make("mul", 3, 2, 6000, 20261210)
                   if r["split"] == "heldout"][:N_TEST],
    }

    conds = [("base", None)]
    for beta in BETAS:
        conds.append((f"beta={beta}", base_row + beta * (Wf - base_row)))
    conds.append(("shuffled 0.2", base_row + 0.2 * (Ws - base_row)))

    results = {}
    for name, te in sets.items():
        prompts = [C.chat_text(p.tok, *C.prompt(r, "answer")) for r in te]
        wants = [r["want"] for r in te]
        print(f"\n=== {name}: {len(te)} fresh problems ===")
        base_ok = None
        out = {}
        for label, mat in conds:
            install(mat)
            reps = C.generate_batch(p, prompts, max_new_tokens=MAX_NEW,
                                    batch_size=40)
            install(None)
            ok = [C.last_int(r) == w for r, w in zip(reps, wants)]
            if base_ok is None:
                base_ok = ok
            b = sum(1 for x, y2 in zip(base_ok, ok) if y2 and not x)
            c = sum(1 for x, y2 in zip(base_ok, ok) if x and not y2)
            pv = mcnemar(b, c)
            print(f"  {label:16s} {sum(ok):3d}/{len(ok):<3d}  "
                  f"+{b:2d} -{c:2d}  p={pv:.3f}")
            out[label] = {"exact": sum(ok) / len(ok), "fixed": b,
                          "broken": c, "p": pv}
        results[name] = out

    C.save(results, C.ROOT / "findings" / "e30_pooled_readout.json")
    print("\nwrote findings/e30_pooled_readout.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
