"""Causal ablation primitives.

Zeroing, not scaling. Head h is removed by clearing its head_dim columns at the
o_proj input, which is exactly the head's contribution to the residual. A neuron
is removed by clearing its column at the down_proj input.
"""

from __future__ import annotations

from contextlib import contextmanager


@contextmanager
def ablate(p, heads=(), channels=(), head_dim=None):
    """Zero selected head outputs and MLP neuron columns for the duration.

    heads:    iterable of (layer, head)
    channels: iterable of (layer, [neuron indices])
    """
    hd = head_dim or p.head_dim
    handles = []

    for li, h in heads:
        mod = p.model.model.layers[li].self_attn.o_proj

        def fn(module, args, h=h):
            x = args[0].clone()
            x[..., h * hd:(h + 1) * hd] = 0
            return (x,) + args[1:]

        handles.append(mod.register_forward_pre_hook(fn))

    for li, cols in channels:
        mod = p.model.model.layers[li].mlp.down_proj
        cols = list(cols)

        def fn(module, args, cols=cols):
            x = args[0].clone()
            x[..., cols] = 0
            return (x,) + args[1:]

        handles.append(mod.register_forward_pre_hook(fn))

    try:
        yield
    finally:
        for h in handles:
            h.remove()
