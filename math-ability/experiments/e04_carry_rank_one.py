"""E04: a conditional rank-one edit for carries.

A constant bias cannot say "if there is a carry". A rank-one change to
down_proj can: it adds d scaled by how strongly the carry feature u is active
in the MLP hidden state.

    W_down <- W_down + alpha * d u^T

u is the mean carry-minus-no-carry direction in the MLP hidden state. d is the
matching direction in the residual after the layer. alpha is calibrated so the
average residual shift is a fixed fraction of the residual norm.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C
import edits as E

SEED = 20261107
N_PAIR = 80
LAYERS = [8, 12, 16, 20]
SHIFTS = [0.05, 0.1, 0.2, 0.4]


@torch.no_grad()
def capture_layer(p, prompt, L):
    ids = p.encode(prompt)
    store = {}

    def pre(module, args):
        store["act"] = args[0][0, -1].detach().float().cpu()

    def post(module, args, out):
        store["res"] = out[0, -1].detach().float().cpu()

    h1 = p.model.model.layers[L].mlp.down_proj.register_forward_pre_hook(pre)
    h2 = p.model.model.layers[L].register_forward_hook(post)
    try:
        p.model(ids)
    finally:
        h1.remove()
        h2.remove()
    return store["act"], store["res"]


def carry_sets(seed, n):
    pool = C.make("add", 3, 3, n * 20, seed)
    carry = [r for r in pool if r["split"] == "train" and 1 <= len(r["carry_pos"]) <= 2 and r["want"] < 1000]
    nocarry = [r for r in pool if r["split"] == "train" and len(r["carry_pos"]) == 0 and r["want"] < 1000]
    return carry[:n], nocarry[:n]


def main() -> int:
    p = C.load()
    carry, nocarry = carry_sets(SEED, N_PAIR)
    print("loaded", p.n_layers, "layers;", len(carry), "carry,", len(nocarry), "no-carry")

    train = E.hard_set(SEED, 20, "train")
    reg = []
    for op, da, db in [("add", 2, 2), ("add", 3, 3), ("mul", 1, 1), ("mul", 2, 1)]:
        pool = C.make(op, da, db, 40, SEED)
        reg += [r for r in pool if r["split"] == "train"][:10]
    print(len(train), "train problems,", len(reg), "regression problems")

    base_train = E.eval_rows(p, train)
    base_reg = E.eval_rows(p, reg)
    print(f"\nbase train {E.accuracy(base_train):.2f}  reg {E.accuracy(base_reg):.2f}")

    results = {"base": {"train": E.accuracy(base_train), "reg": E.accuracy(base_reg)}}
    best = None

    for L in LAYERS:
        ca, cr = [], []
        for r in carry:
            a, h = capture_layer(p, C.chat_text(p.tok, *C.prompt(r, "answer")), L)
            ca.append(a)
            cr.append(h)
        na, nr = [], []
        for r in nocarry:
            a, h = capture_layer(p, C.chat_text(p.tok, *C.prompt(r, "answer")), L)
            na.append(a)
            nr.append(h)

        u = torch.stack(ca).mean(0) - torch.stack(na).mean(0)
        d = torch.stack(cr).mean(0) - torch.stack(nr).mean(0)
        u = u / u.norm()
        d = d / d.norm()
        all_act = torch.stack(ca + na)
        proj = (all_act @ u).abs().mean().item()
        print(f"\nlayer {L}: |u| raw {u.norm():.3f}, mean |u.act| {proj:.3f}")

        W = p.model.model.layers[L].mlp.down_proj.weight
        u_dev = u.to(W.device, W.dtype)
        d_dev = d.to(W.device, W.dtype)
        for shift in SHIFTS:
            alpha = shift / proj
            newW = W.detach() + alpha * torch.outer(d_dev, u_dev)
            with E.applied(p, [(W, newW)]):
                got = E.eval_rows(p, train)
                greg = E.eval_rows(p, reg)
            acc = E.accuracy(got)
            racc = E.accuracy(greg)
            key = f"L{L} shift{shift}"
            results[key] = {"train": acc, "reg": racc, "alpha": alpha}
            print(f"  {key:14s} train {acc:.2f}  reg {racc:.2f}")
            if best is None or acc > best[0]:
                best = (acc, L, shift, alpha, d, u)

    C.save(results, C.ROOT / "findings" / "e04_carry_rank_one.json")
    print(f"\nbest on train: L{best[1]} shift {best[2]} acc {best[0]:.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
