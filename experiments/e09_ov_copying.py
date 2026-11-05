"""E09: copying heads from weights alone, with no positional dependence.

E08's composition method failed, and the main suspect is RoPE: position is
applied as a rotation to Q and K, so a readout that only touches W_Q and W_K
cannot see "look at the previous token". The OV circuit has no such problem.

For head (l, h), the matrix

    M = W_E @ W_OV_h @ W_U^T

maps an input token to a boost on every output token's logit. A head that copies
tokens produces a matrix with a strong diagonal: writing token a makes token a
more likely. That is a pure weight-space test, no forward pass, no positions.

Scoring: mean diagonal minus mean off-diagonal, over a sample of the vocabulary.

Run: python experiments/e09_ov_copying.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from analyze import Model  # noqa: E402

MODEL = ROOT / "Qwen2.5-0.5B-Instruct"
D_HEAD = 64
VOCAB_SAMPLE = 8192
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def main() -> int:
    m = Model(str(MODEL))
    cfg = json.loads((MODEL / "config.json").read_text(encoding="utf-8"))
    n_heads = cfg["num_attention_heads"]
    n_kv = cfg["num_key_value_heads"]
    n_layers = cfg["num_hidden_layers"]
    d_model = cfg["hidden_size"]

    W_E = torch.tensor(m.read("model.embed_tokens.weight").astype(np.float32),
                       device=DEVICE)
    tied = cfg.get("tie_word_embeddings", False)
    if tied:
        W_U = W_E
    else:
        W_U = torch.tensor(m.read("lm_head.weight").astype(np.float32), device=DEVICE)
    print(f"device {DEVICE} | vocab {W_E.shape[0]:,} | tied embeddings {tied}")

    rng = np.random.default_rng(0)
    vidx = torch.tensor(rng.choice(W_E.shape[0], VOCAB_SAMPLE, replace=False),
                        device=DEVICE)
    E = W_E[vidx]      # [S, d_model]
    U = W_U[vidx]      # [S, d_model]

    kv_per_group = n_heads // n_kv
    scores = np.zeros((n_layers, n_heads), dtype=np.float64)

    for li in range(n_layers):
        p = f"model.layers.{li}.self_attn."
        v = torch.tensor(m.read(p + "v_proj.weight").astype(np.float32), device=DEVICE)
        o = torch.tensor(m.read(p + "o_proj.weight").astype(np.float32), device=DEVICE)
        for h in range(n_heads):
            j = h // kv_per_group
            W_V = v[j * D_HEAD:(j + 1) * D_HEAD, :]          # [d_head, d_model]
            W_O = o[:, h * D_HEAD:(h + 1) * D_HEAD]          # [d_model, d_head]
            W_OV = W_O @ W_V                                  # [d_model, d_model]
            A = E @ W_OV                                      # [S, d_model]
            full = A @ U.T                                    # [S, S]
            diag = full.diagonal()
            off = full.sum() - diag.sum()
            off_mean = off / (VOCAB_SAMPLE * VOCAB_SAMPLE - VOCAB_SAMPLE)
            scores[li, h] = float(diag.mean() - off_mean)
        if li % 6 == 0:
            print(f"  layer {li:2d} done", flush=True)

    flat = [(float(scores[li, h]), li, h)
            for li in range(n_layers) for h in range(n_heads)]
    flat.sort(reverse=True)

    print("\n== top 20 copying heads by OV diagonal (weights only) ==")
    print(f"{'rank':>5}{'layer':>7}{'head':>6}{'copy score':>13}")
    for r, (s, li, h) in enumerate(flat[:20], 1):
        print(f"{r:>5}{li:>7}{h:>6}{s:>13.3f}")

    e01 = json.loads((ROOT / "findings" / "e01_anatomy.json").read_text(encoding="utf-8"))
    act_ind = np.array([row["induction_by_head"] for row in e01["attention"]])
    e05 = -np.array(json.loads(
        (ROOT / "findings" / "e05_induction_importance.json").read_text(encoding="utf-8"))["grid"])

    def corr(a, b):
        a = a.ravel() - a.mean()
        b = b.ravel() - b.mean()
        den = np.sqrt((a * a).sum() * (b * b).sum())
        return float((a * b).sum() / den) if den else float("nan")

    print("\n== agreement with measurement ==")
    print(f"  OV copy score vs activation induction score : {corr(scores, act_ind):+.3f}")
    print(f"  OV copy score vs ablation induction damage  : {corr(scores, e05):+.3f}")

    rank = {(li, h): r for r, (_, li, h) in enumerate(flat, 1)}
    print("\n== ranks of key heads ==")
    for li, h in [(9, 13), (11, 12), (16, 3), (0, 3), (0, 8), (2, 5), (8, 7)]:
        print(f"  L{li}H{h}: copy rank {rank[(li, h)]:>3} of {n_layers*n_heads}, "
              f"score {scores[li, h]:+.3f}, act induction {act_ind[li, h]:.3f}, "
              f"ablation damage {e05[li, h]:+.3f}")

    (ROOT / "findings" / "e09_ov_copying.json").write_text(json.dumps({
        "scores": scores.tolist(),
        "top20": [{"score": s, "layer": li, "head": h} for s, li, h in flat[:20]],
        "correlations": {"vs_activation_induction": corr(scores, act_ind),
                         "vs_ablation_induction": corr(scores, e05)},
        "ranks": {f"L{li}H{h}": rank[(li, h)]
                  for li, h in [(9, 13), (11, 12), (16, 3), (0, 3), (0, 8), (2, 5), (8, 7)]},
    }, indent=2), encoding="utf-8")
    print("\nwrote findings/e09_ov_copying.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
