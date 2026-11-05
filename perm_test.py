"""Permutation-invariance check: raw cosine is NOT invariant to neuron reordering."""
import numpy as np
from analyze import Model

m = Model("Qwen2.5-0.5B")
rng = np.random.default_rng(0)

targets = [
    "model.layers.10.self_attn.q_proj.weight",
    "model.layers.10.mlp.gate_proj.weight",
    "model.layers.10.mlp.down_proj.weight",
]

print("raw cosine vs a row-permuted copy of the SAME tensor")
print(f"{'tensor':<44}{'true':>8}{'permuted':>10}")
for n in targets:
    w = m.read(n).astype(np.float32)
    p = w[rng.permutation(w.shape[0]), :]
    a, b = w.ravel(), p.ravel()
    cos = float(np.dot(a, b) / (np.sqrt(np.dot(a, a)) * np.sqrt(np.dot(b, b))))
    print(f"{n:<44}{1.0:>8.4f}{cos:>10.4f}")
