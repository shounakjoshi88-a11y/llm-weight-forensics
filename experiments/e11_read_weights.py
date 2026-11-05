"""E11: read weights directly, blind.

Five networks with identical architecture (2 -> 32 -> 1) trained on five
different 2D classification tasks. The weights are then dumped with the labels
hidden, so the task has to be identified from the numbers alone.

This is the blog's Experiment 1 shape at a scale where reading is possible: if an
LLM can identify a network from its weights, it should work here, and if the
weights genuinely encode the task, the structure should be visible.

    python experiments/e11_read_weights.py --dump     # anonymised weights
    python experiments/e11_read_weights.py --reveal   # ground truth + accuracy
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

HIDDEN = 32
N = 2000
SEED = 7


def make_data(task: str, n: int = N, seed: int = 0):
    rng = np.random.default_rng(seed)
    X = rng.uniform(-1, 1, size=(n, 2)).astype(np.float32)
    x, y = X[:, 0], X[:, 1]
    if task == "vertical":      # x > 0
        Y = (x > 0).astype(np.float32)
    elif task == "horizontal":  # y > 0
        Y = (y > 0).astype(np.float32)
    elif task == "diagonal":    # x + y > 0
        Y = ((x + y) > 0).astype(np.float32)
    elif task == "xor":
        Y = ((x > 0) ^ (y > 0)).astype(np.float32)
    elif task == "circle":      # inside a radius
        Y = ((x ** 2 + y ** 2) < 0.35).astype(np.float32)
    else:
        raise ValueError(task)
    return torch.tensor(X), torch.tensor(Y).unsqueeze(1)


TASKS = ["vertical", "horizontal", "diagonal", "xor", "circle"]


def train(task: str, seed: int):
    torch.manual_seed(seed)
    net = nn.Sequential(nn.Linear(2, HIDDEN), nn.ReLU(), nn.Linear(HIDDEN, 1))
    X, Y = make_data(task, seed=1)
    opt = torch.optim.Adam(net.parameters(), lr=0.02)
    lossf = nn.BCEWithLogitsLoss()
    for _ in range(3000):
        opt.zero_grad()
        loss = lossf(net(X), Y)
        loss.backward()
        opt.step()
    with torch.no_grad():
        acc = ((net(X) > 0).float() == Y).float().mean().item()
    W1 = net[0].weight.detach().numpy().copy()   # [32, 2]
    b1 = net[0].bias.detach().numpy().copy()     # [32]
    W2 = net[2].weight.detach().numpy().copy()   # [1, 32]
    b2 = float(net[2].bias.detach().numpy()[0])
    return {"W1": W1, "b1": b1, "W2": W2, "b2": b2, "acc": acc, "task": task}


def dump(nets, order):
    for slot, key in zip("ABCDE", order):
        n = nets[key]
        W1, b1, W2 = n["W1"], n["b1"], n["W2"][0]
        norms = np.linalg.norm(W1, axis=1)
        top = np.argsort(-norms)[:8]
        print(f"\n=== NET {slot} ===")
        print(f"{'neuron':>7}{'w_x':>9}{'w_y':>9}{'bias':>9}{'w_out':>9}{'|w|':>8}")
        for i in top:
            print(f"{i:>7}{W1[i,0]:>9.3f}{W1[i,1]:>9.3f}{b1[i]:>9.3f}"
                  f"{W2[i]:>9.3f}{norms[i]:>8.3f}")
        act = (W1[:, 0] ** 2).sum()
        tot = (W1 ** 2).sum()
        print(f"  share of input weight on x: {act / tot:.3f}   "
              f"output bias {n['b2']:+.3f}")
        print(f"  hidden neurons with |w| > 0.5: {int((norms > 0.5).sum())}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump", action="store_true")
    ap.add_argument("--reveal", action="store_true")
    a = ap.parse_args()

    nets = {t: train(t, SEED) for t in TASKS}
    mapping = ROOT / "findings" / "e11_blind_mapping.json"

    if a.dump:
        rng = np.random.default_rng(3)
        order = list(TASKS)
        rng.shuffle(order)
        mapping.write_text(json.dumps({"order": order}, indent=2), encoding="utf-8")
        print("five networks, labels hidden. identify each from the numbers.")
        print("architecture is identical: 2 -> 32 -> 1, ReLU.")
        print("columns: w_x, w_y = first layer weight on each input.")
        print("w_out = second layer weight from that hidden unit to the output.")
        dump(nets, order)
        print("\nmapping saved, do not read it until you have answered.")
        return

    if a.reveal:
        order = json.loads(mapping.read_text(encoding="utf-8"))["order"]
        print(f"{'slot':>5}{'task':>13}{'train acc':>11}")
        for slot, key in zip("ABCDE", order):
            print(f"{slot:>5}{key:>13}{nets[key]['acc']:>11.3f}")
        return

    ap.print_help()


if __name__ == "__main__":
    main()
