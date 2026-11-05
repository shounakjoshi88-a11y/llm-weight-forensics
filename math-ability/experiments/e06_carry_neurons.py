"""E06: carry neurons.

Rank MLP neurons by how much their last-token activation separates carry from
no-carry problems, then scale the corresponding down_proj rows. Unlike the
rank-one direction in e04, this keeps the neuron basis and only changes gains.

Two tasks: 3-digit addition and 3-digit by 1-digit multiplication.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C
import edits as E

SEED = 20261110
N_PAIR = 80
LAYERS = [8, 12, 16, 20]
KS = [8, 32, 128]
BETAS = [0.5, 1.0]


@torch.no_grad()
def last_act(p, prompt, L):
    ids = p.encode(prompt)
    store = {}

    def pre(module, args):
        store["act"] = args[0][0, -1].detach().float().cpu()

    h = p.model.model.layers[L].mlp.down_proj.register_forward_pre_hook(pre)
    try:
        p.model(ids)
    finally:
        h.remove()
    return store["act"]


def mul_carries(a: int, b: int) -> int:
    c, n = 0, 0
    while a:
        v = (a % 10) * b + c
        if v >= 10:
            n += 1
        c = v // 10
        a //= 10
    return n


def add_sets(seed, n):
    pool = C.make("add", 3, 3, n * 20, seed)
    carry = [r for r in pool if r["split"] == "train" and 1 <= len(r["carry_pos"]) <= 2 and r["want"] < 1000]
    nocarry = [r for r in pool if r["split"] == "train" and len(r["carry_pos"]) == 0 and r["want"] < 1000]
    return carry[:n], nocarry[:n]


def mul_sets(seed, n):
    pool = C.make("mul", 3, 1, n * 20, seed)
    carry = [r for r in pool if r["split"] == "train" and mul_carries(r["a"], r["b"]) >= 1]
    nocarry = [r for r in pool if r["split"] == "train" and mul_carries(r["a"], r["b"]) == 0]
    return carry[:n], nocarry[:n]


def heldout(op, da, db, seeds, n):
    rows = []
    for s in seeds:
        pool = C.make(op, da, db, n, s)
        rows += [r for r in pool if r["split"] == "heldout"]
    return rows[:n]


def rank_neurons(p, carry, nocarry, L):
    ca = torch.stack([last_act(p, C.chat_text(p.tok, *C.prompt(r, "answer")), L) for r in carry])
    na = torch.stack([last_act(p, C.chat_text(p.tok, *C.prompt(r, "answer")), L) for r in nocarry])
    diff = ca.mean(0) - na.mean(0)
    scale = torch.cat([ca, na]).std(0) + 1e-6
    score = diff.abs() / scale
    return torch.argsort(score, descending=True)


def run_task(p, name, carry, nocarry, eval_rows, L, order):
    base = E.accuracy(E.eval_rows(p, eval_rows))
    W = p.model.model.layers[L].mlp.down_proj.weight
    out = {"base": base, "configs": {}}
    for k in KS:
        idx = order[:k].to(W.device)
        for beta in BETAS:
            newW = W.detach().clone()
            newW[:, idx] *= (1.0 + beta)
            with E.applied(p, [(W, newW)]):
                acc = E.accuracy(E.eval_rows(p, eval_rows))
            out["configs"][f"k{k} b{beta}"] = acc
            print(f"  {name} L{L} k{k:3d} beta{beta:.1f}: {base:.2f} -> {acc:.2f}")
    return out


def main() -> int:
    p = C.load()
    print("loaded", p.n_layers, "layers")

    ac, an = add_sets(SEED, N_PAIR)
    mc, mn = mul_sets(SEED, N_PAIR)
    add_eval = heldout("add", 3, 3, [41, 42], 100)
    mul_eval = heldout("mul", 3, 1, [43, 44], 100)
    print(len(ac), "add carry,", len(an), "add no-carry;",
          len(mc), "mul carry,", len(mn), "mul no-carry")

    results = {"add": {}, "mul": {}}
    for L in LAYERS:
        order_a = rank_neurons(p, ac, an, L)
        results["add"][f"L{L}"] = run_task(p, "add", ac, an, add_eval, L, order_a)
        order_m = rank_neurons(p, mc, mn, L)
        results["mul"][f"L{L}"] = run_task(p, "mul", mc, mn, mul_eval, L, order_m)

    C.save(results, C.ROOT / "findings" / "e06_carry_neurons.json")
    print("\nwrote findings/e06_carry_neurons.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
