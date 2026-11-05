"""E05: per-layer gain search.

Each layer gets two scalars: one on o_proj (attention contribution) and one on
down_proj (MLP contribution). Coordinate descent on a training split, then one
held-out evaluation of the selected gains. This is a direct weight edit: the
matrices are multiplied in place, nothing is added.

The point is to find out whether the arithmetic circuit is mis-weighted at the
layer level, which the global sweeps in e03 could not answer.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C
import edits as E

SEED_TRAIN = 10261108
SEED_VAL = 20261109
N = 10
VALUES = [0.85, 1.15]


def build(p, gains):
    edits = []
    for L in range(p.n_layers):
        attn = gains.get(("attn", L), 1.0)
        mlp = gains.get(("mlp", L), 1.0)
        if attn != 1.0:
            w = p.model.model.layers[L].self_attn.o_proj.weight
            edits.append((w, w.detach() * attn))
        if mlp != 1.0:
            w = p.model.model.layers[L].mlp.down_proj.weight
            edits.append((w, w.detach() * mlp))
    return edits


def score(p, rows, gains):
    if not gains:
        return E.accuracy(E.eval_rows(p, rows))
    with E.applied(p, build(p, gains)):
        return E.accuracy(E.eval_rows(p, rows))


def main() -> int:
    p = C.load()
    train = E.hard_set(SEED_TRAIN, N, "train")
    val = E.hard_set(SEED_VAL, N, "train")
    print("loaded", p.n_layers, "layers;", len(train), "train,", len(val), "val")

    base_train = score(p, train, {})
    base_val = score(p, val, {})
    print(f"base train {base_train:.3f}  val {base_val:.3f}")

    gains = {}
    best_train = base_train
    history = []

    for pas in range(2):
        print(f"\npass {pas + 1}")
        for kind in ("attn", "mlp"):
            for L in range(p.n_layers):
                key = (kind, L)
                current = gains.get(key, 1.0)
                best_v, best_s = current, score(p, train, gains)
                for v in VALUES:
                    if v == current:
                        continue
                    trial = dict(gains)
                    trial[key] = v
                    s = score(p, train, trial)
                    if s > best_s + 1e-9:
                        best_v, best_s = v, s
                if best_v != current:
                    gains[key] = best_v
                    print(f"  {kind} L{L:2d} {current:.2f} -> {best_v:.2f}  train {best_s:.3f}")
        best_train = score(p, train, gains)
        print(f"  end of pass {pas + 1}: train {best_train:.3f}")
        history.append({"pass": pas + 1, "train": best_train, "gains": {f"{k[0]}{k[1]}": v for k, v in gains.items()}})

    val_score = score(p, val, gains)
    print(f"\nselected gains: train {best_train:.3f}  val {val_score:.3f}  (base val {base_val:.3f})")
    print("changed gains:", {f"{k[0]}{k[1]}": v for k, v in gains.items()})

    # per-cell held-out detail
    detail = {}
    for op, da, db in E.HARD:
        rows = [r for r in val if r["op"] == op and r["da"] == da and r["db"] == db]
        base = E.accuracy(E.eval_rows(p, rows))
        with E.applied(p, build(p, gains)):
            edited = E.accuracy(E.eval_rows(p, rows))
        detail[f"{op}{da}x{db}"] = {"base": base, "edited": edited}
        print(f"  {op}{da}x{db}: {base:.2f} -> {edited:.2f}")

    C.save({
        "base_train": base_train,
        "base_val": base_val,
        "gains": {f"{k[0]}{k[1]}": v for k, v in gains.items()},
        "train_after": best_train,
        "val_after": val_score,
        "history": history,
        "detail": detail,
    }, C.ROOT / "findings" / "e05_gain_search.json")
    print("\nwrote findings/e05_gain_search.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
