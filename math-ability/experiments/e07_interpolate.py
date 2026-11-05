"""E07: interpolate toward the base checkpoint, layer by layer.

The base model is better at 4 and 5 digit addition and worse at multiplication.
A full 10 percent blend helps both. This finds which layers carry the addition
gain, then tests the selected blend on held-out problems.

This is a direct weight edit: existing tensors are linearly blended with the
base checkpoint. No training, no new parameters.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C
import edits as E

BASE = str(C.MODEL_WEIGHTS / "Qwen2.5-0.5B")
ALPHA = 0.1
CELLS = [("add", 4, 4), ("add", 5, 5), ("mul", 2, 2), ("mul", 3, 1), ("mul", 5, 1)]
TRAIN_SEEDS = [81, 82]
VAL_SEEDS = [83, 84]


def rows_for(cell, seeds, n=60):
    op, da, db = cell
    rows = []
    for s in seeds:
        pool = C.make(op, da, db, n, s)
        rows += [r for r in pool if r["split"] == "heldout"]
    return rows[:n]


def main() -> int:
    p = C.load()
    base = AutoModelForCausalLM.from_pretrained(BASE, dtype=torch.bfloat16, device_map="cpu")
    bsd = base.state_dict()
    orig = {k: v.detach().clone() for k, v in p.model.state_dict().items()}
    print("loaded both,", len(bsd), "tensors")

    train = {c: rows_for(c, TRAIN_SEEDS) for c in CELLS}
    val = {c: rows_for(c, VAL_SEEDS, 100) for c in CELLS}

    def set_keys(keys, alpha):
        with torch.no_grad():
            for k, v in p.model.state_dict().items():
                if k in keys:
                    v.copy_((1 - alpha) * orig[k] + alpha * bsd[k].to(v.device))

    def evaluate(sets):
        return {f"{c[0]}{c[1]}x{c[2]}": E.accuracy(E.eval_rows(p, rows)) for c, rows in sets.items()}

    base_train = evaluate(train)
    base_val = evaluate(val)
    print("\nbase train:", {k: round(v, 2) for k, v in base_train.items()})
    print("base val:  ", {k: round(v, 2) for k, v in base_val.items()})

    # full blend
    all_keys = set(orig)
    set_keys(all_keys, ALPHA)
    full_train = evaluate(train)
    set_keys(all_keys, 0.0)
    print("\nfull alpha 0.1 train:", {k: round(v, 2) for k, v in full_train.items()})

    # per layer
    layer_delta = {}
    print("\nper-layer deltas on train")
    for L in range(p.n_layers):
        keys = {k for k in orig if f"layers.{L}." in k}
        set_keys(keys, ALPHA)
        res = evaluate(train)
        set_keys(keys, 0.0)
        delta = {k: res[k] - base_train[k] for k in res}
        layer_delta[L] = delta
        print(f"  L{L:2d}  " + "  ".join(f"{k} {delta[k]:+.2f}" for k in delta))

    # select layers that help addition and do not hurt 2x2
    add_keys = ["add4x4", "add5x5"]
    chosen = []
    for L, d in layer_delta.items():
        add_gain = sum(d[k] for k in add_keys)
        if add_gain > 0 and d["mul2x2"] >= -0.02:
            chosen.append(L)
    print("\nchosen layers:", chosen)

    if chosen:
        keys = {k for k in orig if any(f"layers.{L}." in k for L in chosen)}
        set_keys(keys, ALPHA)
        chosen_val = evaluate(val)
        set_keys(keys, 0.0)
    else:
        chosen_val = None

    set_keys(all_keys, ALPHA)
    full_val = evaluate(val)
    set_keys(all_keys, 0.0)

    print("\nval results")
    print("  base  ", {k: round(v, 2) for k, v in base_val.items()})
    if chosen_val:
        print("  chosen", {k: round(v, 2) for k, v in chosen_val.items()})
    print("  full  ", {k: round(v, 2) for k, v in full_val.items()})

    C.save({
        "alpha": ALPHA,
        "base_train": base_train,
        "base_val": base_val,
        "full_train": full_train,
        "full_val": full_val,
        "chosen_layers": chosen,
        "chosen_val": chosen_val,
        "layer_delta": {str(k): v for k, v in layer_delta.items()},
    }, C.ROOT / "findings" / "e07_interpolate.json")
    print("\nwrote findings/e07_interpolate.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
