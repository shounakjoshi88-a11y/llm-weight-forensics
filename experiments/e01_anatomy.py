"""E01: anatomical scan of Qwen2.5-0.5B-Instruct.

Measures, in one pass over two controlled inputs:

  1. residual stream norm per layer          (how much the model rewrites x)
  2. attention entropy per head              (focused vs diffuse heads)
  3. attention sink mass on position 0       (the "no-op" sink)
  4. induction head score                    (prefix matching, the in-context
                                              learning circuit)
  5. previous-token head score               (the primitive induction is built on)
  6. MLP activation sparsity                 (how many neurons actually fire)
  7. logit lens trajectory                   (what each layer would emit)

The induction score uses a repeated random sequence. Position i in the second
copy should attend to i - L + 1, the token that followed the previous occurrence
of the same token. A head doing this is doing prefix matching.

Run: python experiments/e01_anatomy.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# Windows console defaults to cp1252; tokenizer strings are arbitrary UTF-8
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from probe import Probe, save_json  # noqa: E402
from probe.attention import build_repeat, head_scores  # noqa: E402

L = 24  # half-length of the repeated sequence


def main() -> int:
    p = Probe(str(ROOT / "Qwen2.5-0.5B-Instruct"))
    print(f"layers={p.n_layers} hidden={p.hidden} heads={p.n_heads} "
          f"kv={p.n_kv} head_dim={p.head_dim} inter={p.inter}\n")

    ids = build_repeat()
    cap = p.capture(ids)
    n = ids.shape[1]
    print(f"input: {n} tokens (two copies of {L} random tokens)")
    print(f"sample: {p.decode(ids[0][:12].tolist())!r}\n")

    # ---- 1. residual stream norm ----
    hs = cap["hidden_states"]
    norms = [float(h[0].norm(dim=-1).mean()) for h in hs]
    print("== residual stream norm per layer ==")
    print(f"{'layer':>5} {'norm':>9} {'growth':>8}")
    for i, v in enumerate(norms):
        g = "" if i == 0 else f"{v / norms[i-1]:.3f}"
        label = "embed" if i == 0 else str(i - 1)
        print(f"{label:>5} {v:>9.3f} {g:>8}")

    # ---- 2-5. attention per layer ----
    print("\n== attention heads per layer ==")
    print(f"{'L':>3} {'entropy':>8} {'max sink':>9} {'sink@':>6} "
          f"{'induction':>10} {'ind@':>6} {'prev':>7}")
    per_layer = []
    for li, attn in enumerate(cap["attentions"]):
        s = head_scores(attn, L)
        hi = int(np.argmax(s["induction"]))
        si = int(np.argmax(s["sink"]))
        per_layer.append({
            "layer": li,
            "entropy_mean": float(s["entropy"].mean()),
            "sink_max": float(s["sink"].max()),
            "sink_head": si,
            "induction_max": float(s["induction"].max()),
            "induction_head": hi,
            "prev_max": float(s["prev"].max()),
            "induction_by_head": s["induction"].tolist(),
            "prev_by_head": s["prev"].tolist(),
            "entropy_by_head": s["entropy"].tolist(),
            "sink_by_head": s["sink"].tolist(),
        })
        print(f"{li:>3} {s['entropy'].mean():>8.3f} {s['sink'].max():>9.3f} "
              f"{si:>6} {s['induction'].max():>10.3f} {hi:>6} {s['prev'].max():>7.3f}")

    # ---- 6. MLP sparsity ----
    print("\n== MLP activation sparsity ==")
    print(f"{'layer':>5} {'max|act|':>9} {'rms':>8} {'frac~0':>8}")
    mlp_stats = []
    for li, act in enumerate(cap["mlp_act"]):
        a = act[0].numpy()
        mx = float(np.abs(a).max())
        rms = float(np.sqrt((a * a).mean()))
        frac0 = float((np.abs(a) < 0.01 * mx).mean())
        mlp_stats.append({"layer": li, "max": mx, "rms": rms, "frac_near_zero": frac0})
        print(f"{li:>5} {mx:>9.3f} {rms:>8.4f} {frac0:>8.3f}")

    # ---- 7. logit lens ----
    lens = p.logit_lens(cap, topk=3)
    print("\n== logit lens at final position ==")
    print(f"{'layer':>5}  top-3")
    lens_rows = []
    for li, per_pos in enumerate(lens):
        toks = per_pos[-1]
        label = "embed" if li == 0 else str(li - 1)
        s = "  ".join(f"{t!r}:{pr:.3f}" for t, pr in toks)
        lens_rows.append({"layer": li, "top": toks})
        print(f"{label:>5}  {s}")

    save_json({
        "model": "Qwen2.5-0.5B-Instruct",
        "config": {
            "layers": p.n_layers, "hidden": p.hidden, "heads": p.n_heads,
            "kv_heads": p.n_kv, "head_dim": p.head_dim, "intermediate": p.inter,
        },
        "input_tokens": int(n),
        "residual_norms": norms,
        "attention": per_layer,
        "mlp": mlp_stats,
        "logit_lens": lens_rows,
    }, ROOT / "findings" / "e01_anatomy.json")

    print("\nwrote findings/e01_anatomy.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
