"""E25: does the "Step" row edit help raw arithmetic?

The edit is confirmed on word problems (0.23 to 0.59, e24). This checks the
target the folder started with: bare multi-digit arithmetic, where chain of
thought did not help in e01.

20 problems per cell, base against the edit, same prompts and budget.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C
from e17_word_problems import problems
from e18_math_mode import first_h

MAX_NEW = 384
ALPHA = 0.10
CELLS = [("add", 5, 5), ("mul", 3, 2), ("mul", 5, 3)]
N = 20


def main() -> int:
    p = C.load()
    with torch.no_grad():
        p.model.lm_head.weight = torch.nn.Parameter(
            p.model.model.embed_tokens.weight.detach().clone())
    W = p.model.lm_head.weight
    orig = W.detach().clone()
    step = p.tok("Step", add_special_tokens=False).input_ids[0]

    train = problems(80, 20261120)
    starts, mids = [], []
    for q, _ in train:
        base = C.chat_text(p.tok, q, "")
        starts.append(first_h(p, base))
        mids.append(first_h(p, base + "Step"))
    d = torch.stack(starts).mean(0) - torch.stack(mids).mean(0)
    d = d / d.norm()
    dd = d.to(W.device, W.dtype)

    out = {}
    print(f"\n{'cell':10s} {'base':>8s} {'edit':>8s}")
    for op, da, db in CELLS:
        rows = []
        for s in [20261130, 20261131]:
            pool = C.make(op, da, db, N * 2, s)
            rows += [r for r in pool if r["split"] == "heldout"]
        rows = rows[:N]
        prompts = [C.chat_text(p.tok, f"What is {r['a']} {'+' if op == 'add' else 'x'} {r['b']}? Work through it step by step.", "")
                   for r in rows]
        accs = {}
        for label, a in [("base", 0.0), ("edit", ALPHA)]:
            with torch.no_grad():
                W.copy_(orig)
                if a:
                    W[step] += a * dd
            reps = C.generate_batch(p, prompts, max_new_tokens=MAX_NEW, batch_size=20)
            with torch.no_grad():
                W.copy_(orig)
            accs[label] = sum(C.last_int(r) == r0["want"] for r, r0 in zip(reps, rows)) / len(rows)
        key = f"{op}{da}x{db}"
        out[key] = accs
        print(f"{key:10s} {accs['base']:8.2f} {accs['edit']:8.2f}")

    C.save(out, C.ROOT / "findings" / "e25_edit_raw_arithmetic.json")
    print("\nwrote findings/e25_edit_raw_arithmetic.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
