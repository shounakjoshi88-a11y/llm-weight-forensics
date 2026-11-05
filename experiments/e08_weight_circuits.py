"""E08: read the circuit from weights alone, then check it against ablation.

This is the Elhage et al. method from "A Mathematical Framework for Transformer
Circuits", applied without any forward pass.

The idea: a head's QK circuit is W_Q^T W_K, and its OV circuit is W_V W_O. If
head A's output feeds into head B's query, that is Q-composition. If it feeds
into B's key, that is K-composition. An induction head is defined structurally as
a head with high K-composition from a previous-token head: the previous-token
head writes "what came last" into the residual stream, and the induction head
uses that as its key to find the same prefix.

So the claim under test: from weights alone, can we name the induction heads, and
do those names match what ablation says actually matters?

Ground truth comes from E04 and E05: per-head perplexity damage and per-head
induction damage, both measured by ablation.

Run: python experiments/e08_weight_circuits.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from analyze import Model  # noqa: E402

MODEL = ROOT / "Qwen2.5-0.5B-Instruct"
D_HEAD = 64


def load_weights(m: Model, layer: int):
    p = f"model.layers.{layer}.self_attn."
    q = m.read(p + "q_proj.weight").astype(np.float32)   # [d_head*n_h, d_model]
    k = m.read(p + "k_proj.weight").astype(np.float32)   # [d_head*n_kv, d_model]
    v = m.read(p + "v_proj.weight").astype(np.float32)
    o = m.read(p + "o_proj.weight").astype(np.float32)   # [d_model, d_head*n_h]
    return q, k, v, o


def w_ov(v, o, h, n_kv, n_heads):
    """OV circuit for head h: what it writes into the residual stream."""
    kv_per_group = n_heads // n_kv
    j = h // kv_per_group
    W_V = v[j * D_HEAD:(j + 1) * D_HEAD, :]        # [d_head, d_model]
    W_O = o[:, h * D_HEAD:(h + 1) * D_HEAD]        # [d_model, d_head]
    return W_O @ W_V                                # [d_model, d_model]


def w_q(q, h):
    return q[h * D_HEAD:(h + 1) * D_HEAD, :]        # [d_head, d_model]


def w_k(k, h, n_kv, n_heads):
    kv_per_group = n_heads // n_kv
    j = h // kv_per_group
    return k[j * D_HEAD:(j + 1) * D_HEAD, :]


def k_composition(src_wov, dst_wq):
    """||W_Q W_OV|| / (||W_Q|| ||W_OV||), Elhage's K-composition score."""
    prod = dst_wq @ src_wov
    den = np.linalg.norm(dst_wq) * np.linalg.norm(src_wov)
    return float(np.linalg.norm(prod) / den) if den else 0.0


def main() -> int:
    m = Model(str(MODEL))
    cfg = json.loads((MODEL / "config.json").read_text(encoding="utf-8"))
    n_heads = cfg["num_attention_heads"]
    n_kv = cfg["num_key_value_heads"]
    n_layers = cfg["num_hidden_layers"]
    print(f"{n_layers} layers, {n_heads} q heads, {n_kv} kv heads, d_head {D_HEAD}\n")

    # previous-token heads, from the activation measurement in E06
    PREV_HEADS = [(8, 7), (15, 2), (10, 2), (1, 12)]
    print("previous-token heads used as sources (from E06):", PREV_HEADS)

    # cache OV circuits for the source heads
    src_ov = {}
    for li, h in PREV_HEADS:
        _, k_, v, o = load_weights(m, li)
        src_ov[(li, h)] = w_ov(v, o, h, n_kv, n_heads)

    print("\ncomputing K-composition from each source to every head ...")
    comp = np.zeros((n_layers, n_heads), dtype=np.float64)
    for li in range(n_layers):
        q, k, v, o = load_weights(m, li)
        for h in range(n_heads):
            wq = w_q(q, h)
            best = 0.0
            for (sl, sh), wov in src_ov.items():
                if sl >= li:          # a head can only receive from earlier layers
                    continue
                best = max(best, k_composition(wov, wq))
            comp[li, h] = best

    flat = [(float(comp[li, h]), li, h)
            for li in range(n_layers) for h in range(n_heads)]
    flat.sort(reverse=True)

    print("\n== top 15 heads by K-composition from a previous-token head ==")
    print(f"{'rank':>5}{'layer':>7}{'head':>6}{'composition':>14}")
    for r, (c, li, h) in enumerate(flat[:15], 1):
        print(f"{r:>5}{li:>7}{h:>6}{c:>14.4f}")

    # ---- compare with the activation-measured induction score (E01) ----
    e01 = json.loads((ROOT / "findings" / "e01_anatomy.json").read_text(encoding="utf-8"))
    act_ind = np.array([row["induction_by_head"] for row in e01["attention"]])

    # ---- compare with ablation ground truth (E04, E05) ----
    e04 = np.array(json.loads(
        (ROOT / "findings" / "e04_head_importance.json").read_text(encoding="utf-8"))["importance"])
    e05 = -np.array(json.loads(
        (ROOT / "findings" / "e05_induction_importance.json").read_text(encoding="utf-8"))["grid"])

    def corr(a, b):
        a = a.ravel() - a.mean()
        b = b.ravel() - b.mean()
        den = np.sqrt((a * a).sum() * (b * b).sum())
        return float((a * b).sum() / den) if den else float("nan")

    print("\n== does the weight-space circuit predict anything ==")
    print(f"  composition vs activation induction score : {corr(comp, act_ind):+.3f}")
    print(f"  composition vs ablation induction damage   : {corr(comp, e05):+.3f}")
    print(f"  composition vs ablation ppl damage         : {corr(comp, e04):+.3f}")

    print("\n== the heads E01 flagged as induction heads ==")
    print(f"{'head':<10}{'composition':>14}{'act score':>11}{'ind damage':>12}")
    for li, h in [(9, 13), (11, 12), (16, 3)]:
        print(f"L{li}H{h:<8}{comp[li, h]:>14.4f}{act_ind[li, h]:>11.3f}{e05[li, h]:>+12.3f}")

    print("\n== the heads that ablation says matter for induction ==")
    print(f"{'head':<10}{'composition':>14}{'act score':>11}{'ind damage':>12}")
    for li, h in [(0, 3), (0, 8), (2, 5), (8, 7), (0, 2)]:
        print(f"L{li}H{h:<8}{comp[li, h]:>14.4f}{act_ind[li, h]:>11.3f}{e05[li, h]:>+12.3f}")

    # rank agreement
    order_comp = {(li, h): r for r, (_, li, h) in enumerate(flat, 1)}
    print("\n== rank of ablation-critical heads, by weight-space composition ==")
    for li, h in [(0, 3), (0, 8), (2, 5), (8, 7), (9, 13), (11, 12), (16, 3)]:
        print(f"  L{li}H{h}: composition rank {order_comp[(li, h)]} of {n_layers * n_heads}")

    out = {
        "prev_heads": PREV_HEADS,
        "composition": comp.tolist(),
        "top15": [{"score": c, "layer": li, "head": h} for c, li, h in flat[:15]],
        "correlations": {
            "vs_activation_induction": corr(comp, act_ind),
            "vs_ablation_induction": corr(comp, e05),
            "vs_ablation_ppl": corr(comp, e04),
        },
        "ranks": {f"L{li}H{h}": order_comp[(li, h)]
                  for li, h in [(0, 3), (0, 8), (2, 5), (8, 7), (9, 13), (11, 12), (16, 3)]},
    }
    (ROOT / "findings" / "e08_weight_circuits.json").write_text(
        json.dumps(out, indent=2), encoding="utf-8")
    print("\nwrote findings/e08_weight_circuits.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
