"""E28: confirm the readout correction on fresh problems.

E27 found one working point. Refitting the ten digit rows of lm_head in closed
form on 5x5 addition and installing a quarter of the correction raised 5x5
addition from 110/200 to 123/200, paired +14 -1, p = 0.001. That beta was the
best of five, so it needs a set it has never seen.

This fits on the same training split and evaluates on 200 fresh 5x5 addition
problems from a different seed, plus a sweep of nearby betas and a shuffled
label control. It also scores a fresh 3x2 multiplication set, because the ten
digit rows are shared across tasks and the fit was done on addition only. That
is where the collateral shows up.
"""

from __future__ import annotations

import sys
from math import comb
from pathlib import Path

import torch

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C
from e26_digit_trajectory import fixed_length, forward_capture
from e27_readout_refit import fit_readout

BETAS = [0.125, 0.25, 0.375]
N_TRAIN = 600
N_TEST = 200
MAX_NEW = 12
LAM = 100.0


def mcnemar(b: int, c: int) -> float:
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    return min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / (2 ** n))


def build(p, digit_id):
    """Fit the digit rows on 5x5 addition, return the scaled correction."""
    rows = fixed_length("add", 5, 5, 6, N_TRAIN + N_TEST, 20261201)
    tr = [r for r in rows if r["split"] == "train"][:N_TRAIN]
    pref = [C.chat_text(p.tok, *C.prompt(r, "answer")) for r in tr]
    texts = [s + " " + str(r["want"]) for s, r in zip(pref, tr)]
    R, _ = forward_capture(p, texts, 6, digit_id)
    X = torch.cat([R[i] for i in range(6)], 0)
    y = torch.cat([torch.tensor([int(str(r["want"])[i]) for r in tr])
                   for i in range(6)], 0)

    Wf = fit_readout(X, y, LAM)
    g = torch.Generator().manual_seed(13)
    Ws = fit_readout(X, y[torch.randperm(len(y), generator=g)], LAM)
    return Wf, Ws, X


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

    Wf, Ws, X = build(p, digit_id)
    orig_d = orig[digit_id].detach().float().cpu().double()
    lo = X @ orig_d.T
    per_digit = lo.std(0) / (X @ Wf.T).std(0).clamp_min(1e-9)
    Wf = Wf * per_digit[:, None]
    Ws = Ws * per_digit[:, None]
    base_row = orig_d

    def install(mat):
        with torch.no_grad():
            W.copy_(orig)
            if mat is not None:
                W[digit_id] = mat.to(device=W.device, dtype=W.dtype)

    results = {}

    def run(tag, op, da, db, nd, seed, conds):
        rows = fixed_length(op, da, db, nd, N_TEST * 2, seed)
        te = [r for r in rows if r["split"] == "heldout"][:N_TEST]
        prompts = [C.chat_text(p.tok, *C.prompt(r, "answer")) for r in te]
        wants = [r["want"] for r in te]
        print(f"\n=== {tag}: {len(te)} fresh problems, seed {seed} ===")
        out = {}
        base_ok = None
        for label, mat in conds:
            install(mat)
            reps = C.generate_batch(p, prompts, max_new_tokens=MAX_NEW, batch_size=40)
            install(None)
            ok = [C.last_int(r) == w for r, w in zip(reps, wants)]
            if base_ok is None:
                base_ok = ok
            b = sum(1 for x, y in zip(base_ok, ok) if y and not x)
            c = sum(1 for x, y in zip(base_ok, ok) if x and not y)
            pv = mcnemar(b, c)
            print(f"  {label:16s} {sum(ok):3d}/{len(ok):<3d}  "
                  f"+{b:2d} -{c:2d}  p={pv:.3f}")
            out[label] = {"exact": sum(ok) / len(ok), "fixed": b,
                          "broken": c, "p": pv, "replies": reps}
        return out, te

    add_conds = [("base", None)]
    for beta in BETAS:
        add_conds.append((f"beta={beta}", base_row + beta * (Wf - base_row)))
    add_conds.append(("shuffled 0.25", base_row + 0.25 * (Ws - base_row)))
    results["add5x5"], add_te = run("add5x5 fresh", "add", 5, 5, 6, 20261210,
                                    add_conds)

    mul_conds = [("base", None),
                 ("beta=0.25", base_row + 0.25 * (Wf - base_row))]
    results["mul3x2"], mul_te = run("mul3x2 collateral", "mul", 3, 2, 5,
                                    20261210, mul_conds)

    C.save(results, C.ROOT / "findings" / "e28_confirm_readout.json")
    print("\nwrote findings/e28_confirm_readout.json")

    print("\nbase vs beta=0.25 on addition:")
    for a, b in zip(results["add5x5"]["base"]["replies"],
                    results["add5x5"]["beta=0.25"]["replies"]):
        print("  base " + repr(a[:44]) + "   edit " + repr(b[:44]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
