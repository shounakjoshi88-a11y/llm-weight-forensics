"""Core instrumentation for reading a transformer's internals.

Design notes:

- eager attention, because we need the attention weights themselves
- everything captured is moved to CPU float32 immediately, so VRAM stays flat
  and a 6 GB card can analyze far larger models than it can train
- hooks are removed after each capture, so nothing leaks between runs
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


class Probe:
    """A loaded model plus the instrumentation to look inside it."""

    def __init__(
        self,
        model_dir: str,
        device: str = "cuda",
        dtype=torch.bfloat16,
    ):
        self.dir = model_dir
        self.tok = AutoTokenizer.from_pretrained(model_dir)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_dir, dtype=dtype, device_map=device, attn_implementation="eager"
        )
        self.model.eval()
        self.device = next(self.model.parameters()).device

        c = self.model.config
        self.cfg = c
        self.n_layers = c.num_hidden_layers
        self.hidden = c.hidden_size
        self.n_heads = c.num_attention_heads
        self.n_kv = c.num_key_value_heads
        self.head_dim = self.hidden // self.n_heads
        self.inter = c.intermediate_size

    # ------------------------------------------------------------------
    def encode(self, text: str) -> torch.Tensor:
        return self.tok(text, return_tensors="pt").input_ids.to(self.device)

    def decode(self, ids) -> str:
        return self.tok.decode(ids, skip_special_tokens=False)

    # ------------------------------------------------------------------
    @torch.no_grad()
    def capture(self, ids: torch.Tensor) -> dict:
        """One forward pass, recording the residual stream and MLP activations.

        Returns CPU float32 tensors keyed by layer index.
        """
        ids = ids.to(self.device)
        resid = []       # hidden state entering each layer
        mlp_act = []     # gated MLP hidden, the (seq, intermediate) activation
        handles = []

        layers = self.model.model.layers

        def pre_hook(idx):
            def fn(module, args):
                resid.append(args[0].detach().float().cpu())
            return fn

        def mlp_hook(idx):
            # input to down_proj is silu(gate(x)) * up(x)
            def fn(module, args, out):
                mlp_act.append(args[0].detach().float().cpu())
            return fn

        for i, layer in enumerate(layers):
            handles.append(layer.register_forward_pre_hook(pre_hook(i)))
            handles.append(layer.mlp.down_proj.register_forward_hook(mlp_hook(i)))

        out = self.model(ids, output_attentions=True, output_hidden_states=True)
        for h in handles:
            h.remove()

        return {
            "ids": ids.detach().cpu(),
            "resid_in": resid,                       # list[L] of [seq, hidden]
            "mlp_act": mlp_act,                      # list[L] of [seq, intermediate]
            "hidden_states": [h.detach().float().cpu() for h in out.hidden_states],
            "attentions": [a.detach().float().cpu() for a in out.attentions],  # [B,H,Q,K]
            "logits": out.logits.detach().float().cpu(),
        }

    # ------------------------------------------------------------------
    @torch.no_grad()
    def logit_lens(self, cap: dict, topk: int = 5) -> list[list[tuple[str, float]]]:
        """Apply the final norm and unembedding to every layer's residual.

        This answers: if the model had to emit a token right here, what would it
        say? It is a crude readout, and its failures are as informative as its
        successes.
        """
        norm = self.model.model.norm
        lm = self.model.lm_head
        results = []
        for h in cap["hidden_states"]:
            x = h.to(self.device).to(torch.bfloat16)
            logits = lm(norm(x))[0]                     # [seq, vocab]
            probs = torch.softmax(logits.float(), dim=-1)
            top = torch.topk(probs, topk, dim=-1)
            per_pos = []
            for pos in range(probs.shape[0]):
                toks = [
                    (self.tok.decode([int(i)]), float(p))
                    for i, p in zip(top.indices[pos], top.values[pos])
                ]
                per_pos.append(toks)
            results.append(per_pos)
        return results

    # ------------------------------------------------------------------
    @torch.no_grad()
    def generate(self, text: str, max_new_tokens: int = 40) -> str:
        ids = self.encode(text)
        out = self.model.generate(
            ids, max_new_tokens=max_new_tokens, do_sample=False,
            attention_mask=torch.ones_like(ids),
        )
        return self.tok.decode(out[0][ids.shape[1]:], skip_special_tokens=True)


# ----------------------------------------------------------------------
def save_json(obj, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    def default(o):
        if isinstance(o, (np.floating, np.integer)):
            return o.item()
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, torch.Tensor):
            return o.tolist()
        raise TypeError(f"{type(o)}")

    path.write_text(json.dumps(obj, indent=2, default=default), encoding="utf-8")
    return path
