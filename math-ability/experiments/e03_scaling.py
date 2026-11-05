"""E03: do simple volume knobs on the arithmetic circuit move anything?

Scales q, o or down weights by a constant and measures exact accuracy on the
hard training cells. This is a cheap screen, not a final result. Anything that
moves gets a held-out test afterwards.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C
import edits as E

SEED = 20261106
N = 20

CONFIGS = [
    ("base", None, None, None),
    ("q all 0.7", ["self_attn.q_proj"], 0.7, None),
    ("q all 1.3", ["self_attn.q_proj"], 1.3, None),
    ("q all 1.6", ["self_attn.q_proj"], 1.6, None),
    ("q all 2.0", ["self_attn.q_proj"], 2.0, None),
    ("o all 0.7", ["self_attn.o_proj"], 0.7, None),
    ("o all 1.3", ["self_attn.o_proj"], 1.3, None),
    ("o all 1.6", ["self_attn.o_proj"], 1.6, None),
    ("down all 0.7", ["mlp.down_proj"], 0.7, None),
    ("down all 1.3", ["mlp.down_proj"], 1.3, None),
    ("down all 1.6", ["mlp.down_proj"], 1.6, None),
    ("q L8 1.5", ["self_attn.q_proj"], 1.5, [8]),
    ("q L12 1.5", ["self_attn.q_proj"], 1.5, [12]),
    ("q L16 1.5", ["self_attn.q_proj"], 1.5, [16]),
    ("q L20 1.5", ["self_attn.q_proj"], 1.5, [20]),
]


def main() -> int:
    p = C.load()
    rows = E.hard_set(SEED, N, "train")
    print("loaded", p.n_layers, "layers;", len(rows), "train problems")

    print(f"\n{'config':14s} {'total':>6s}  " + "  ".join(f"{o}{a}x{b}" for o, a, b in E.HARD))
    results = {}
    for name, names, beta, layers in CONFIGS:
        if names is None:
            got = E.eval_rows(p, rows)
        else:
            with E.applied(p, E.scale_params(p, names, beta, layers)):
                got = E.eval_rows(p, rows)
        per = {}
        for op, da, db in E.HARD:
            cell = [g for g in got if g["op"] == op and g["da"] == da and g["db"] == db]
            per[f"{op}{da}x{db}"] = E.accuracy(cell)
        results[name] = {"total": E.accuracy(got), "per": per, "rows": got}
        print(f"{name:14s} {E.accuracy(got):6.2f}  " + "  ".join(f"{per[f'{o}{a}x{b}']:.2f}" for o, a, b in E.HARD))

    C.save(results, C.ROOT / "findings" / "e03_scaling.json")
    print("\nwrote findings/e03_scaling.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
