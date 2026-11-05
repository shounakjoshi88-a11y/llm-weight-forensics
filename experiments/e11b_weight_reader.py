"""E11b: does the hand-reading generalise, or was it hindsight?

E11 identified five networks from their weights by eye. Five out of five is
easy to over-read: the reasoning was written after the numbers were visible, so
it is not a generalisation estimate.

This makes the reading explicit and scores it on networks the rule was not
tuned on.

Two readers are compared.

  naive   the statistics that look obvious: how much first-layer energy sits on
          x, how tightly the hidden directions cluster, how large the biases are.
          These are the first three things one would reach for when reading the
          numbers by hand.

  readout the same idea corrected. For a network whose output is monotone in
          some direction, that direction is the output-weighted sum of the
          first-layer weight vectors, v = sum_i w_out_i * w_i, not the average
          of the individual vectors. When the function is not monotone, those
          contributions cancel and v is short, which is itself the signature.

The rule is fixed using seed 7 only (the networks E11 read by eye), then scored
on seeds 11, 23, 41, 97. No forward pass is used to identify anything. The
forward pass is only used to confirm each network learned its task.

    python experiments/e11b_weight_reader.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from e11_read_weights import TASKS, train  # noqa: E402

SEEDS = [7, 11, 23, 41, 97]
TUNE_SEED = 7


def significant(net) -> np.ndarray:
    """Mask of hidden units large enough for their bias to mean anything."""
    norms = np.linalg.norm(net["W1"], axis=1)
    return norms > 0.05 * norms.max()


def naive_features(net) -> dict:
    W1, b1, W2 = net["W1"], net["b1"], net["W2"][0]
    energy = (W1 ** 2).sum()
    norms = np.linalg.norm(W1, axis=1)
    keep = norms > 1e-6
    u = W1[keep] / norms[keep, None]
    return {
        "e_x": float((W1[:, 0] ** 2).sum() / energy) if energy else 0.5,
        "concentration": float(np.linalg.norm(u.sum(axis=0)) / len(u)),
        "bias_ratio": float((np.abs(b1[keep]) / norms[keep]).mean()),
    }


def naive_read(f: dict) -> str:
    if f["e_x"] > 0.90:
        return "vertical"
    if f["e_x"] < 0.10:
        return "horizontal"
    if f["concentration"] > 0.70:
        return "diagonal"
    if f["bias_ratio"] < 0.05:
        return "xor"
    return "circle"


def readout_features(net) -> dict:
    W1, b1, W2 = net["W1"], net["b1"], net["W2"][0]
    norms = np.linalg.norm(W1, axis=1)
    keep = significant(net)

    v = (W2[:, None] * W1).sum(axis=0)
    denom = float((np.abs(W2) * norms).sum())
    alignment = float(np.linalg.norm(v) / denom) if denom else 0.0
    angle = float(np.degrees(np.arctan2(v[1], v[0])) % 180.0)

    d = np.abs(b1[keep]) / norms[keep]
    return {
        "alignment": alignment,
        "angle": angle,
        "boundary_radius": float(d.mean()),
        "v": v.tolist(),
    }


def readout_read(f: dict) -> str:
    """Monotone networks point along one axis. Non-monotone ones cancel."""
    if f["alignment"] >= 0.50:
        a = f["angle"]
        if a < 22.5 or a >= 157.5:
            return "vertical"
        if 67.5 <= a < 112.5:
            return "horizontal"
        return "diagonal"
    return "xor" if f["boundary_radius"] < 0.10 else "circle"


def main() -> int:
    print("training 25 networks (5 tasks x 5 seeds) ...\n")
    nets = {}
    for task in TASKS:
        for seed in SEEDS:
            nets[(task, seed)] = train(task, seed)

    tuned = [(t, TUNE_SEED) for t in TASKS]
    held = [(t, s) for t in TASKS for s in SEEDS if s != TUNE_SEED]

    for name, feat, reader in (
        ("naive", naive_features, naive_read),
        ("readout", readout_features, readout_read),
    ):
        ok_t = sum(reader(feat(nets[k])) == k[0] for k in tuned)
        ok_h = sum(reader(feat(nets[k])) == k[0] for k in held)
        print(f"{name:>8} reader | tuned on seed 7: {ok_t}/5 | "
              f"held out (4 seeds): {ok_h}/{len(held)} = {ok_h/len(held):.0%}")

    print(f"\n{'task':>11}{'seed':>6}{'align':>8}{'angle':>8}{'radius':>9}"
          f"{'read as':>12}{'ok':>5}")
    hits = 0
    for (task, seed), net in nets.items():
        f = readout_features(net)
        guess = readout_read(f)
        ok = guess == task
        hits += ok
        print(f"{task:>11}{seed:>6}{f['alignment']:>8.3f}{f['angle']:>8.1f}"
              f"{f['boundary_radius']:>9.3f}{guess:>12}"
              f"{'yes' if ok else 'NO':>5}")
    print(f"\nreadout reader, all 25: {hits}/25 = {hits/25:.0%}")

    print("\n== where the naive reader went wrong ==")
    for (task, seed), net in nets.items():
        n = naive_features(net)
        if naive_read(n) != task:
            r = readout_features(net)
            print(f"  {task:>10} seed {seed:>2}: naive said {naive_read(n):>10}, "
                  f"because concentration was {n['concentration']:.2f} "
                  f"(threshold 0.70) while alignment is {r['alignment']:.2f}")
    print("  the individual directions are spread even when the function is")
    print("  monotone, because the optimiser spreads the work over many units.")
    print("  Only the output-weighted sum recovers the direction.")

    print("\n== control: permuting hidden units ==")
    rng = np.random.default_rng(0)
    stable = 0
    for net in nets.values():
        perm = rng.permutation(net["W1"].shape[0])
        shuffled = {"W1": net["W1"][perm], "b1": net["b1"][perm],
                    "W2": net["W2"][:, perm], "b2": net["b2"],
                    "acc": net["acc"], "task": net["task"]}
        stable += readout_read(readout_features(shuffled)) == readout_read(readout_features(net))
    print(f"  read unchanged for {stable}/{len(nets)} permuted networks")
    print("  the function is identical under this permutation, so the reading")
    print("  must be identical too. It is, because the reader looks at the set")
    print("  of weight vectors and their output weights, not at neuron indices.")

    print("\n== control: swapping the two inputs ==")
    ok = 0
    for (task, seed), net in nets.items():
        swapped = {"W1": net["W1"][:, ::-1], "b1": net["b1"],
                   "W2": net["W2"], "b2": net["b2"],
                   "acc": net["acc"], "task": net["task"]}
        guess = readout_read(readout_features(swapped))
        expect = {"vertical": "horizontal", "horizontal": "vertical"}.get(task, task)
        ok += guess == expect
    print(f"  correct after the swap for {ok}/{len(nets)} networks")
    print("  vertical and horizontal trade places, the other three are")
    print("  symmetric in x and y and must not change.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
