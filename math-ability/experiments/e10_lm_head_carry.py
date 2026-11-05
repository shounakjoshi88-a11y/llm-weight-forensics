"""E10: carry-conditioned digit bias in lm_head.

The errors are often one digit too low or too high in a carry column. A static
bias cannot condition on the input, but a rank-one edit to lm_head can:

    W_lm <- W_lm + alpha * v d^T

where v is the signed digit-value vector over the vocabulary (digit j gets
j - 4.5, everything else 0) and d is the carry direction in the final normed
residual. The logit of digit j then gets alpha * (j - 4.5) * (d . h), so the
model prefers higher digits when the carry feature is active.

Only the digit rows of lm_head change. No gradients, no new parameters.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C
import edits as E

SEED = 20261111
N_PAIR = 100
LAYERS = [12, 16, 20, 23]
ALPHAS = [0.01, 0.02, 0.05, 0.1]


@torch.no_grad()
def final_normed(p, prompt):
    ids = p.encode(prompt)
    store = {}

    def hook(module, args, out):
        store["h"] = out[0, -1].detach().float().cpu()

    h = p.model.model.norm.register_forward_hook(hook)
    try:
        p.model(ids)
    finally:
        h.remove()
    return store["h"]


def carry_sets(seed, n):
    pool = C.make("add", 3, 3, n * 20, seed)
    carry = [r for r in pool if r["split"] == "train" and 1 <= len(r["carry_pos"]) <= 2 and r["want"] < 1000]
    nocarry = [r for r in pool if r["split"] == "train" and len(r["carry_pos"]) == 0 and r["want"] < 1000]
    return carry[:n], nocarry[:n]


def heldout(op, da, db, seeds, n):
    rows = []
    for s in seeds:
        pool = C.make(op, da, db, n, s)
        rows += [r for r in pool if r["split"] == "heldout"]
    return rows[:n]


def main() -> int:
    p = C.load()
    # separate the tied output table so this edit does not touch the input side
    with torch.no_grad():
        p.model.lm_head.weight = torch.nn.Parameter(
            p.model.model.embed_tokens.weight.detach().clone())

    carry, nocarry = carry_sets(SEED, N_PAIR)
    print("loaded", p.n_layers, "layers;", len(carry), "carry,", len(nocarry), "no-carry")

    # signed digit value vector over the vocabulary
    ids = [p.tok.convert_tokens_to_ids(str(d)) for d in range(10)]
    v = torch.zeros(p.model.lm_head.weight.shape[0], dtype=torch.float32)
    for j, tid in enumerate(ids):
        v[tid] = j - 4.5

    train = heldout("add", 3, 3, [111, 112], 80)
    val = heldout("add", 3, 3, [113, 114], 100)
    mul_val = heldout("mul", 2, 2, [115, 116], 100)
    reg = heldout("mul", 2, 1, [117], 60)

    base = {k: E.accuracy(E.eval_rows(p, rows)) for k, rows in
            [("add_train", train), ("add_val", val), ("mul_val", mul_val), ("reg", reg)]}
    print("base", {k: round(v, 2) for k, v in base.items()})

    # carry direction in the final normed residual
    ch = torch.stack([final_normed(p, C.chat_text(p.tok, *C.prompt(r, "answer"))) for r in carry])
    nh = torch.stack([final_normed(p, C.chat_text(p.tok, *C.prompt(r, "answer"))) for r in nocarry])
    d = ch.mean(0) - nh.mean(0)
    d = d / d.norm()
    print("carry direction norm", round(d.norm().item(), 3))

    W = p.model.lm_head.weight
    results = {}
    best = None
    for alpha in ALPHAS:
        newW = W.detach() + alpha * torch.outer(v.to(W.device, W.dtype), d.to(W.device, W.dtype))
        with E.applied(p, [(W, newW)]):
            got_train = E.accuracy(E.eval_rows(p, train))
            got_val = E.accuracy(E.eval_rows(p, val))
            got_mul = E.accuracy(E.eval_rows(p, mul_val))
            got_reg = E.accuracy(E.eval_rows(p, reg))
        key = f"alpha{alpha}"
        results[key] = {"train": got_train, "val": got_val, "mul": got_mul, "reg": got_reg}
        print(f"  {key:10s} add_train {got_train:.2f}  add_val {got_val:.2f}  mul {got_mul:.2f}  reg {got_reg:.2f}")
        if best is None or got_train > best[0]:
            best = (got_train, alpha)

    print(f"\nbest alpha on train {best[1]} (train {best[0]:.2f})")
    C.save({"base": base, "results": results, "best_alpha": best[1]},
           C.ROOT / "findings" / "e10_lm_head_carry.json")
    print("wrote findings/e10_lm_head_carry.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
