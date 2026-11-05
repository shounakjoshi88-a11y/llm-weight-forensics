"""E10: neuron atlas. What does each MLP neuron detect?

For every MLP neuron in the model (24 layers x 4864 = 116,736), this finds the
positions in a real corpus where it fires hardest, and keeps the surrounding text.

That is the standard way to characterise a neuron without guessing: a neuron is
described by the inputs that excite it. It is descriptive, not causal. Causal
claims require ablation, which is what E02 and E03 do.

Output:
  findings/e10_neuron_atlas.json    per-neuron top contexts and statistics
  findings/e10_summary.json         aggregate statistics

Run: python experiments/e10_neuron_atlas.py --tokens 30000
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from probe import Probe  # noqa: E402
from probe.corpus import build_corpus  # noqa: E402

TOPK = 8
CHUNK = 512
CTX = 12


class Accumulator:
    """Running top-k per neuron, without storing every activation."""

    def __init__(self, n_neurons: int, k: int = TOPK):
        self.k = k
        self.vals = np.full((n_neurons, k), -np.inf, dtype=np.float32)
        self.pos = np.full((n_neurons, k), -1, dtype=np.int64)
        self.total = 0          # positions seen
        self.nonzero = np.zeros(n_neurons, dtype=np.int64)
        self.sumsq = np.zeros(n_neurons, dtype=np.float64)
        self.absmax = np.zeros(n_neurons, dtype=np.float32)

    def add(self, act: np.ndarray, offset: int):
        """act: [N, n_neurons] float32 for N consecutive positions."""
        n = act.shape[0]
        a = act.T                                  # [n_neurons, N]
        self.total += n
        self.nonzero += (np.abs(a) > 1e-3).sum(axis=1)
        self.sumsq += (a.astype(np.float64) ** 2).sum(axis=1)
        np.maximum(self.absmax, np.abs(a).max(axis=1), out=self.absmax)

        cand_v = np.concatenate([self.vals, a], axis=1)          # [neur, k+N]
        cand_p = np.concatenate(
            [self.pos, np.broadcast_to(
                np.arange(offset, offset + n, dtype=np.int64), (a.shape[0], n))],
            axis=1)
        part = np.argpartition(-cand_v, self.k - 1, axis=1)[:, : self.k]
        rows = np.arange(a.shape[0])[:, None]
        self.vals = cand_v[rows, part]
        self.pos = cand_p[rows, part]

        order = np.argsort(-self.vals, axis=1)
        self.vals = np.take_along_axis(self.vals, order, axis=1)
        self.pos = np.take_along_axis(self.pos, order, axis=1)

    def stats(self):
        mean = self.sumsq / max(self.total, 1)
        rms = np.sqrt(mean)
        freq = self.nonzero / max(self.total, 1)
        with np.errstate(divide="ignore", invalid="ignore"):
            sel = np.where(rms > 0, self.absmax / rms, 0.0)
        return rms, freq, sel


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokens", type=int, default=30000)
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()

    p = Probe(str(ROOT / "Qwen2.5-0.5B-Instruct"), device=a.device)
    print("building corpus ...")
    docs = build_corpus(ROOT / "corpus_cache")
    print()

    accs = [Accumulator(p.inter) for _ in range(p.n_layers)]
    all_tokens: list[int] = []
    consumed = 0

    for di, doc in enumerate(docs):
        if consumed >= a.tokens:
            break
        ids = p.tok(doc, return_tensors="pt", truncation=True,
                    max_length=2048).input_ids
        flat = ids[0].tolist()
        for start in range(0, len(flat), CHUNK):
            if consumed >= a.tokens:
                break
            chunk = flat[start:start + CHUNK]
            if len(chunk) < 8:
                continue
            t = torch.tensor([chunk]).to(p.device)
            cap = p.capture(t)
            for li in range(p.n_layers):
                accs[li].add(cap["mlp_act"][li][0].numpy(), consumed)
            all_tokens.extend(chunk)
            consumed += len(chunk)
        if di % 20 == 0:
            print(f"  doc {di:4d} | {consumed:6d} tokens", flush=True)

    print(f"\nprocessed {consumed} tokens\n")
    toks = np.array(all_tokens)

    def context(pos: int) -> str:
        lo, hi = max(0, pos - CTX), min(len(toks), pos + CTX)
        before = p.tok.decode(toks[lo:pos].tolist())
        at = p.tok.decode([int(toks[pos])])
        after = p.tok.decode(toks[pos + 1:hi].tolist())
        return f"{before}[[{at}]]{after}"

    atlas = {}
    summary = {"tokens": consumed, "layers": {}, "dead": {}, "selective": []}

    for li in range(p.n_layers):
        rms, freq, sel = accs[li].stats()
        vals, pos = accs[li].vals, accs[li].pos

        dead = int((accs[li].absmax < 1e-2).sum())
        summary["dead"][li] = dead

        layer_rows = []
        for n in range(p.inter):
            ctxs = [context(int(pp)) for pp in pos[n] if pp >= 0][:TOPK]
            layer_rows.append({
                "neuron": n,
                "rms": float(rms[n]),
                "max": float(accs[li].absmax[n]),
                "freq": float(freq[n]),
                "selectivity": float(sel[n]),
                "contexts": ctxs,
            })
        atlas[li] = layer_rows

        # most selective neurons in this layer, the ones that look like features
        order = np.argsort(-sel)
        summary["layers"][li] = {
            "dead": dead,
            "median_rms": float(np.median(rms)),
            "median_freq": float(np.median(freq)),
            "max_selectivity": float(sel[order[0]]),
        }
        for n in order[:3]:
            summary["selective"].append({
                "layer": li, "neuron": int(n), "selectivity": float(sel[n]),
                "freq": float(freq[n]), "contexts": atlas[li][n]["contexts"][:3],
            })

    out = ROOT / "findings" / "e10_neuron_atlas.json"
    out.write_text(__import__("json").dumps(atlas), encoding="utf-8")
    (ROOT / "findings" / "e10_summary.json").write_text(
        __import__("json").dumps(summary, indent=2), encoding="utf-8")

    print("== per layer ==")
    print(f"{'layer':>6}{'dead':>8}{'med rms':>10}{'med freq':>10}{'max sel':>9}")
    for li in range(p.n_layers):
        s = summary["layers"][li]
        print(f"{li:>6}{s['dead']:>8}{s['median_rms']:>10.4f}"
              f"{s['median_freq']:>10.3f}{s['max_selectivity']:>9.1f}")

    print("\n== most selective neurons overall ==")
    for row in sorted(summary["selective"], key=lambda r: -r["selectivity"])[:12]:
        print(f"\n  L{row['layer']}N{row['neuron']}  selectivity {row['selectivity']:.1f} "
              f"freq {row['freq']:.3f}")
        for c in row["contexts"][:2]:
            print(f"    {c[:110]}")

    print(f"\nwrote {out.name} and e10_summary.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
