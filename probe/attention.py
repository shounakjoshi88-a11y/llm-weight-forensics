"""Attention-pattern utilities shared by the experiments."""

from __future__ import annotations

import numpy as np
import torch


def build_repeat(seed: int = 0, half_len: int = 24, repeat: bool = True) -> torch.Tensor:
    """Two copies of a random token sequence.

    The repeat is what makes induction visible: position i in the second copy
    should attend to i - half_len + 1, the token after the previous occurrence.
    """
    rng = np.random.default_rng(seed)
    toks = rng.integers(1000, 100_000, size=half_len).tolist()
    second = toks if repeat else toks[::-1]
    return torch.tensor([toks + second])


def head_scores(attn: torch.Tensor, half_len: int) -> dict:
    """Per-head scalar scores from one layer's attention tensor [1, H, Q, K]."""
    a = attn[0].numpy()
    H, Q, _ = a.shape
    n = Q

    prev = np.zeros(H)
    for i in range(1, n):
        prev += a[:, i, i - 1]
    prev /= max(n - 1, 1)

    induction = np.zeros(H)
    cnt = 0
    for i in range(half_len, n):
        tgt = i - half_len + 1
        if 0 <= tgt < i:
            induction += a[:, i, tgt]
            cnt += 1
    induction /= max(cnt, 1)

    sink = np.zeros(H)
    for i in range(1, n):
        sink += a[:, i, 0]
    sink /= max(n - 1, 1)

    ent = np.zeros(H)
    for i in range(1, n):
        p = np.clip(a[:, i, : i + 1], 1e-12, None)
        p = p / p.sum(axis=1, keepdims=True)
        ent += -(p * np.log(p)).sum(axis=1) / np.log(i + 1)
    ent /= max(n - 1, 1)

    return {"prev": prev, "induction": induction, "sink": sink, "entropy": ent}
