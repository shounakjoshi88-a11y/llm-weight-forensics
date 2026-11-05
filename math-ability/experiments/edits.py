"""Weight-edit primitives and an evaluation loop.

Every edit here changes existing tensors in place and restores them afterwards.
No parameters are added, no gradients are used.
"""

from __future__ import annotations

import sys
from contextlib import contextmanager
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C


@contextmanager
def applied(p, edits):
    """Apply a list of (parameter, new_tensor) and restore on exit."""
    backup = [(param, param.detach().clone()) for param, _ in edits]
    try:
        with torch.no_grad():
            for param, new in edits:
                param.copy_(new)
        yield
    finally:
        with torch.no_grad():
            for param, old in backup:
                param.copy_(old)


def scale_params(p, names, beta, layers=None):
    """Return edits that multiply selected weight matrices by beta."""
    layers = range(p.n_layers) if layers is None else layers
    edits = []
    for L in layers:
        layer = p.model.model.layers[L]
        for name in names:
            mod = layer
            for part in name.split("."):
                mod = getattr(mod, part)
            edits.append((mod.weight, mod.weight.detach() * beta))
    return edits


def eval_rows(p, rows, fmt="answer", max_new=12, batch_size=60):
    prompts = [C.chat_text(p.tok, *C.prompt(r, fmt)) for r in rows]
    replies = C.generate_batch(p, prompts, max_new_tokens=max_new, batch_size=batch_size)
    return [{**r, "reply": rep, **C.score(rep, r["want"])} for r, rep in zip(rows, replies)]


def accuracy(got):
    return sum(g["ok"] for g in got) / max(1, len(got))


HARD = [
    ("add", 4, 4), ("add", 5, 5), ("sub", 3, 3), ("sub", 4, 4),
    ("mul", 2, 2), ("mul", 3, 2),
]


def hard_set(seed, n=20, split="train"):
    rows = []
    for op, da, db in HARD:
        pool = C.make(op, da, db, n * 3, seed)
        rows += [r for r in pool if r["split"] == split][:n]
    return rows
