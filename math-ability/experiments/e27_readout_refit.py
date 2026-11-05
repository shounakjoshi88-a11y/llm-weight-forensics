"""E27: replace the digit rows of lm_head with a closed-form readout.

E26 found the first real headroom in this folder. For 3x2 multiplication the
leading digit of the answer is linearly decodable from the prompt residual at
0.87, but the model's own lm_head emits it 0.24 of the time. Everywhere else
the probe matches the model, which is why every earlier edit was null: there was
nothing for a readout to recover.

This installs the readout. Collect teacher-forced (residual, true next digit)
pairs over the training split, solve the least-squares problem in closed form,

    W = (X'X + lambda I)^-1 X'Y

and write the ten digit rows of lm_head. No gradients, no new parameters, no
change to any other row, so non-digit outputs are untouched.

Three conditions on held-out problems:

    base       the shipped lm_head
    fitted     digit rows replaced by the closed-form solution
    shuffled   the same procedure with permuted labels, as the null

Also reports the fitted readout's own per-position accuracy, so it is clear
whether it beats lm_head everywhere or only at the first digit.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C
from e26_digit_trajectory import fixed_length, forward_capture

N_TRAIN = 600
N_TEST = 200
LAMBDAS = [0.01, 0.1, 1.0, 10.0, 100.0]
MAX_NEW = 12


def fit_readout(X: torch.Tensor, y: torch.Tensor, lam: float) -> torch.Tensor:
    """Closed-form least squares, no intercept. Returns (10, d)."""
    d = X.shape[1]
    Y = torch.zeros(X.shape[0], 10, dtype=torch.float64)
    Y[torch.arange(X.shape[0]), y] = 1.0
    A = X.T @ X + lam * torch.eye(d, dtype=X.dtype)
    W = torch.linalg.solve(A, X.T @ Y)
    return W.T.contiguous()


def main() -> int:
    p = C.load()
    with torch.no_grad():
        p.model.lm_head.weight = torch.nn.Parameter(
            p.model.model.embed_tokens.weight.detach().clone())
    W = p.model.lm_head.weight
    orig = W.detach().clone()
    print("loaded", p.n_layers, "layers")

    digit_id = [p.tok(str(d), add_special_tokens=False).input_ids[0]
                for d in range(10)]
    print("digit rows", digit_id)

    tasks = [("add5x5", "add", 5, 5, 6), ("mul3x2", "mul", 3, 2, 5)]
    results = {}

    for name, op, da, db, nd in tasks:
        rows = fixed_length(op, da, db, nd, N_TRAIN + N_TEST, 20261201)
        tr = [r for r in rows if r["split"] == "train"][:N_TRAIN]
        te = [r for r in rows if r["split"] == "heldout"][:N_TEST]
        print(f"\n=== {name}: {len(tr)} train, {len(te)} held out ===")
        if len(tr) < 300 or len(te) < 100:
            print("  not enough problems, skipping")
            continue

        pref_tr = [C.chat_text(p.tok, *C.prompt(r, "answer")) for r in tr]
        pref_te = [C.chat_text(p.tok, *C.prompt(r, "answer")) for r in te]
        txt_tr = [s + " " + str(r["want"]) for s, r in zip(pref_tr, tr)]
        txt_te = [s + " " + str(r["want"]) for s, r in zip(pref_te, te)]

        R_tr, _ = forward_capture(p, txt_tr, nd, digit_id)
        R_te, L_te = forward_capture(p, txt_te, nd, digit_id)

        X_tr = torch.cat([R_tr[i] for i in range(nd)], 0)
        y_tr = torch.cat([
            torch.tensor([int(str(r["want"])[i]) for r in tr]) for i in range(nd)
        ], 0)
        X_te = {i: R_te[i] for i in range(nd)}
        y_te = {i: torch.tensor([int(str(r["want"])[i]) for r in te])
                for i in range(nd)}

        # lambda on a validation slice of train
        cut = int(len(X_tr) * 0.8)
        best = (None, -1.0)
        for lam in LAMBDAS:
            Wf = fit_readout(X_tr[:cut], y_tr[:cut], lam)
            sc = X_tr[cut:] @ Wf.T
            acc = (sc.argmax(1) == y_tr[cut:]).float().mean().item()
            if acc > best[1]:
                best = (lam, acc)
        lam = best[0]
        print(f"  lambda {lam} (val digit acc {best[1]:.2f})")

        Wf = fit_readout(X_tr, y_tr, lam)
        g = torch.Generator().manual_seed(13)
        Ws = fit_readout(X_tr, y_tr[torch.randperm(len(y_tr), generator=g)], lam)

        # Least squares puts the digit logits on a 0..1 scale, while real logits
        # run to +-20, so an unscaled fit never wins against the other 151,936
        # tokens. Match each digit row's spread to its original before using it.
        orig_d = orig[digit_id].detach().float().cpu().double()
        lo = X_tr @ orig_d.T
        fi = X_tr @ Wf.T
        per_digit = lo.std(0) / fi.std(0).clamp_min(1e-9)
        Wf = Wf * per_digit[:, None]
        Ws = Ws * per_digit[:, None]
        print(f"  logit spread: original {lo.std():.2f}, "
              f"fitted {float((X_tr @ (Wf / per_digit[:, None]).T).std()):.2f}, "
              f"per-digit multipliers {per_digit.min():.1f} to {per_digit.max():.1f}")

        # per-position accuracy of the fitted readout vs lm_head
        print(f"\n  {'digit':>5s} {'lm_head':>8s} {'fitted':>7s} {'shuffled':>9s}")
        pos_stats = {}
        for i in range(nd):
            base_sc = L_te[i]
            fit_sc = X_te[i] @ Wf.T
            sh_sc = X_te[i] @ Ws.T
            b = (base_sc.argmax(1) == y_te[i]).float().mean().item()
            f = (fit_sc.argmax(1) == y_te[i]).float().mean().item()
            s = (sh_sc.argmax(1) == y_te[i]).float().mean().item()
            pos_stats[i] = {"lm_head": b, "fitted": f, "shuffled": s}
            print(f"  {i:5d} {b:8.2f} {f:7.2f} {s:9.2f}")

        out = {"lambda": lam, "per_digit_scale": per_digit.tolist(),
               "positions": pos_stats}

        # what actually wins at the first digit position
        enc = p.tok(pref_te[0], return_tensors="pt").to(p.device)
        with torch.no_grad():
            W.copy_(orig)
            b0 = p.model(**enc).logits[0, -1]
            W[digit_id] = Wf.to(device=W.device, dtype=W.dtype)
            f0 = p.model(**enc).logits[0, -1]
            W.copy_(orig)
        top_b = [p.tok.decode([int(i)]) for i in b0.topk(5).indices]
        top_f = [p.tok.decode([int(i)]) for i in f0.topk(5).indices]
        print("  first token top5 base  ", top_b)
        print("  first token top5 fitted", top_f)

        # Replacing the rows outright breaks the stop token, so sweep a
        # correction instead: beta 0 is the shipped model, beta 1 is the fit.
        print(f"\n  {'condition':16s} {'exact':>8s}")
        base_row = orig[digit_id].detach().float().cpu().double()
        conds = [("base", None)]
        for beta in [0.25, 0.5, 0.75, 1.0]:
            conds.append((f"fitted beta={beta}", base_row + beta * (Wf - base_row)))
        conds.append(("shuffled beta=1", Ws))
        for label, mat in conds:
            with torch.no_grad():
                W.copy_(orig)
                if mat is not None:
                    W[digit_id] = mat.to(device=W.device, dtype=W.dtype)
            reps = C.generate_batch(p, pref_te, max_new_tokens=MAX_NEW, batch_size=40)
            with torch.no_grad():
                W.copy_(orig)
            ok = sum(C.last_int(r) == r0["want"] for r, r0 in zip(reps, te))
            print(f"  {label:16s} {ok:3d}/{len(te):<3d}  {ok/len(te):.2f}")
            out[label] = {"exact": ok / len(te), "replies": reps}
        results[name] = out

    C.save(results, C.ROOT / "findings" / "e27_readout_refit.json")
    print("\nwrote findings/e27_readout_refit.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
