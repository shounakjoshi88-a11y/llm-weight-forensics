"""Live neuron visualizer for Qwen2.5-0.5B-Instruct.

Runs the model locally and streams, for every token it reads or writes, the real
activation of a fixed set of MLP neurons in every layer. The browser draws the
network and lights up the neurons that fire.

Which neurons: 48 per layer, chosen once at startup from the weight matrix, by
how strongly each neuron writes into the residual stream (the column norm of
`down_proj`). The set is fixed so a neuron keeps its place on screen across
tokens. The values streamed are the actual post-activation MLP outputs for those
neurons, not a summary or a simulation.

    python viz/server.py
    open http://127.0.0.1:8000
"""

from __future__ import annotations

import argparse
import json
import sys
import threading
import time
from pathlib import Path

import torch
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, StreamingResponse
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

MODEL_DIR = ROOT / "Qwen2.5-0.5B-Instruct"
PER_LAYER = 48
MAX_PROMPT = 256

app = FastAPI(title="neuron visualizer")

_model = None
_tok = None
_dev = "cuda" if torch.cuda.is_available() else "cpu"
_captured: dict[int, torch.Tensor] = {}
_selected: list[torch.Tensor] = []
_lock = threading.Lock()


def load():
    """Load the model and pick the neurons that will be displayed."""
    global _model, _tok, _selected
    if _model is not None:
        return
    _tok = AutoTokenizer.from_pretrained(str(MODEL_DIR))
    _model = AutoModelForCausalLM.from_pretrained(
        str(MODEL_DIR), dtype=torch.bfloat16, device_map=_dev
    )
    _model.eval()

    for i, layer in enumerate(_model.model.layers):
        # down_proj is [hidden, intermediate]. Column norm = how much this
        # neuron can write into the residual stream.
        col = layer.mlp.down_proj.weight.float().norm(dim=0)
        order = torch.argsort(col, descending=True)
        half = col.numel() // 2
        # spread the picks over the stronger half so the view is not only the
        # handful of always-on neurons
        picks = torch.linspace(0, half - 1, PER_LAYER).round().long().unique()
        sel = order[picks].sort().values.cpu()
        _selected.append(sel)

        def hook(module, args, out, i=i, sel=sel):
            x = args[0][0]                       # [seq, intermediate]
            _captured[i] = (
                x.index_select(1, sel.to(x.device)).detach().float().cpu()
            )

        layer.mlp.down_proj.register_forward_hook(hook)


def sse(obj: dict) -> str:
    return "data: " + json.dumps(obj) + "\n\n"


def payload(phase: str, step: int, token: str, token_id: int,
            pos: int, row: int) -> dict:
    """One frame: the activation of every displayed neuron while reading `token`."""
    values, maxima = [], []
    for li in range(len(_selected)):
        cap = _captured.get(li)
        if cap is None:
            values.append([0.0] * PER_LAYER)
            maxima.append(0.0)
            continue
        v = cap[row].tolist()
        values.append([round(x, 3) for x in v])
        maxima.append(round(max(abs(x) for x in v), 3))
    return {
        "phase": phase,
        "step": step,
        "pos": pos,
        "token": token,
        "id": token_id,
        "v": values,
        "mx": maxima,
    }


def stream(prompt: str, max_tokens: int):
    load()
    ids = _tok(prompt, return_tensors="pt", truncation=True,
               max_length=MAX_PROMPT).input_ids.to(_dev)
    toks = ids[0].tolist()
    if not toks:
        toks = [_tok.eos_token_id]
        ids = torch.tensor([toks], device=_dev)

    with _lock:
        with torch.no_grad():
            out = _model(ids, use_cache=True)
        past = out.past_key_values

        # phase 1: the model reading the question, one position at a time.
        # Paced so the read is watchable rather than a single blur.
        for p, tid in enumerate(toks):
            yield sse(payload("read", p, _tok.decode([tid]), tid, p, p))
            time.sleep(0.07)

        # phase 2: generation, one token at a time
        nxt = int(out.logits[0, -1].argmax())
        for step in range(max_tokens):
            cur = torch.tensor([[nxt]], device=_dev)
            with torch.no_grad():
                out = _model(cur, past_key_values=past, use_cache=True)
            past = out.past_key_values
            yield sse(payload("write", step, _tok.decode([nxt]), nxt,
                              len(toks) + step, 0))
            if nxt == _tok.eos_token_id:
                break
            nxt = int(out.logits[0, -1].argmax())

    yield 'data: {"done": true}\n\n'


@app.get("/health")
def health():
    return {"ok": True, "device": _dev, "loaded": _model is not None}


@app.get("/meta")
def meta():
    return {
        "model": "Qwen2.5-0.5B-Instruct",
        "layers": len(_selected),
        "per_layer": PER_LAYER,
        "neurons": [s.tolist() for s in _selected],
        "device": _dev,
    }


@app.get("/stream")
def stream_ep(prompt: str = "The capital of France is", max_tokens: int = 40):
    return StreamingResponse(stream(prompt, max_tokens),
                             media_type="text/event-stream")


@app.get("/", response_class=HTMLResponse)
def index():
    return (Path(__file__).parent / "static" / "index.html").read_text(
        encoding="utf-8")


def main():
    import uvicorn
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--host", default="127.0.0.1")
    a = ap.parse_args()
    print(f"loading model on {_dev} ...")
    load()
    print(f"open http://{a.host}:{a.port}")
    uvicorn.run(app, host=a.host, port=a.port, log_level="warning")


if __name__ == "__main__":
    main()
