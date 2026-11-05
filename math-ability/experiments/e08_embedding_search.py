"""E08: search the ` is` embedding row.

The prompt ends on one token, ` is`, immediately before the answer. Its
embedding row is 896 numbers. A random perturbation at the row's own norm
moved 3-digit addition and 2x2 multiplication, so this searches that row for a
direction that helps on train, then tests it once on held-out problems.

lm_head is separated from the embedding table first, so this edits the input
side only. No gradients, no new parameters.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C
import edits as E

TRAIN_SEEDS = [101, 102]
VAL_SEEDS = [103, 104]
CELLS = [("add", 3, 3), ("add", 4, 4), ("mul", 2, 2), ("mul", 3, 1)]
N_DIR = 40


def rows_for(seeds, n=50):
    rows = []
    for op, da, db in CELLS:
        for s in seeds:
            pool = C.make(op, da, db, n, s)
            rows += [r for r in pool if r["split"] == "heldout"]
    return rows[: n * len(CELLS)]


def main() -> int:
    p = C.load()
    with torch.no_grad():
        p.model.lm_head.weight = torch.nn.Parameter(
            p.model.model.embed_tokens.weight.detach().clone())

    ids = p.tok("The answer is", add_special_tokens=False).input_ids
    last = ids[-1]
    emb = p.model.model.embed_tokens.weight
    row = emb[last].detach().clone()
    norm = row.norm().item()
    print("last token", p.tok.decode([last]), "row norm", round(norm, 3))

    train = rows_for(TRAIN_SEEDS)
    val = rows_for(VAL_SEEDS)
    print(len(train), "train,", len(val), "held out")

    def evaluate(rows):
        return E.accuracy(E.eval_rows(p, rows))

    base_train = evaluate(train)
    base_val = evaluate(val)
    print(f"base train {base_train:.3f}  held-out {base_val:.3f}")

    torch.manual_seed(7)
    results = []
    for i in range(N_DIR):
        g = torch.randn_like(row)
        g = g / g.norm() * norm
        with torch.no_grad():
            emb[last].copy_(row + g)
        acc = evaluate(train)
        with torch.no_grad():
            emb[last].copy_(row)
        results.append((acc, i, g))
        if (i + 1) % 10 == 0:
            print(f"  {i + 1}/{N_DIR} directions, best train so far {max(r[0] for r in results):.3f}")

    results.sort(key=lambda x: -x[0])
    print("\ntop 5 train directions:")
    for acc, i, _ in results[:5]:
        print(f"  dir {i}: train {acc:.3f}")

    # best single
    best_acc, best_i, best_g = results[0]
    with torch.no_grad():
        emb[last].copy_(row + best_g)
    held_best = evaluate(val)
    with torch.no_grad():
        emb[last].copy_(row)
    print(f"\nbest single: train {best_acc:.3f}  held-out {held_best:.3f}")

    # average of top 5
    avg = torch.stack([g for _, _, g in results[:5]]).mean(0)
    avg = avg / avg.norm() * norm
    with torch.no_grad():
        emb[last].copy_(row + avg)
    held_avg = evaluate(val)
    with torch.no_grad():
        emb[last].copy_(row)
    print(f"top-5 average: held-out {held_avg:.3f}")

    # random control direction not selected on train
    with torch.no_grad():
        emb[last].copy_(row + results[-1][2])
    held_ctrl = evaluate(val)
    with torch.no_grad():
        emb[last].copy_(row)
    print(f"random control: held-out {held_ctrl:.3f}")

    C.save({
        "row_norm": norm,
        "base_train": base_train,
        "base_val": base_val,
        "directions": [{"i": i, "train": a} for a, i, _ in results],
        "best_train": best_acc,
        "best_val": held_best,
        "avg_val": held_avg,
        "control_val": held_ctrl,
        "best_index": best_i,
    }, C.ROOT / "findings" / "e08_embedding_search.json")
    print("\nwrote findings/e08_embedding_search.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
