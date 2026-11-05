"""Live neuron visualizer for Qwen2.5-0.5B-Instruct.

Runs the model locally and streams, for every generated token, the activation of
every MLP neuron in every layer. The browser renders that as a heatmap that
flashes as the model thinks.

    python viz/server.py
    open http://127.0.0.1:8000

No build step, no dependencies beyond what the repo already uses.
"""

from __future__ import annotations

import argparse
import json
import sys
import threading
from pathlib import Path

import torch
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, StreamingResponse
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

MODEL_DIR = ROOT / "Qwen2.5-0.5B-Instruct"
TOPK = 64

app = FastAPI(title="neuron visualizer")

_model = None
_tok = None
_dev = "cuda" if torch.cuda.is_available() else "cpu"
_captured: dict[int, torch.Tensor] = {}
_lock = threading.Lock()


def load():
    global _model, _tok
    if _model is not None:
        return
    _tok = AutoTokenizer.from_pretrained(str(MODEL_DIR))
    _model = AutoModelForCausalLM.from_pretrained(
        str(MODEL_DIR), dtype=torch.bfloat16, device_map=_dev
    )
    _model.eval()

    for i, layer in enumerate(_model.model.layers):
        def hook(module, args, out, i=i):
            _captured[i] = args[0][0, -1, :].detach().float().cpu()
        layer.mlp.down_proj.register_forward_hook(hook)


def payload_for(token: str, token_id: int, step: int) -> dict:
    layers = []
    for li in range(_model.config.num_hidden_layers):
        act = _captured.get(li)
        if act is None:
            continue
        mag = act.abs()
        k = min(TOPK, mag.numel())
        vals, idx = torch.topk(mag, k)
        top = [
            {"n": int(idx[j]), "v": float(act[idx[j]])}
            for j in range(k)
        ]
        layers.append({
            "layer": li,
            "max": float(vals[0]) if k else 0.0,
            "top": top,
        })
    return {"step": step, "token": token, "token_id": token_id, "layers": layers}


def stream(prompt: str, max_tokens: int):
    load()
    ids = _tok(prompt, return_tensors="pt").input_ids.to(_dev)
    past = None
    cur = ids

    with _lock:
        for step in range(max_tokens):
            with torch.no_grad():
                out = _model(cur, past_key_values=past, use_cache=True)
            past = out.past_key_values
            nxt = int(out.logits[0, -1].argmax())
            cur = torch.tensor([[nxt]], device=_dev)
            text = _tok.decode([nxt])
            yield "data: " + json.dumps(payload_for(text, nxt, step)) + "\n\n"
            if nxt == _tok.eos_token_id:
                break
    yield "data: {\"done\": true}\n\n"


@app.get("/health")
def health():
    return {"ok": True, "device": _dev, "loaded": _model is not None}


@app.get("/stream")
def stream_ep(prompt: str = "The capital of France is", max_tokens: int = 40):
    return StreamingResponse(stream(prompt, max_tokens), media_type="text/event-stream")


@app.get("/", response_class=HTMLResponse)
def index():
    return (Path(__file__).parent / "static" / "index.html").read_text(encoding="utf-8")


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
