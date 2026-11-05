import traceback
import torch
from transformer_lens import TransformerBridge

try:
    bridge = TransformerBridge.boot_transformers(
        "Qwen2.5-0.5B-Instruct", device="cuda", dtype=torch.bfloat16
    )
    print("BOOTED")
    print("  n_layers", bridge.cfg.n_layers)
    print("  d_model ", bridge.cfg.d_model)
    print("  n_heads ", bridge.cfg.n_heads)
    print("  d_mlp   ", bridge.cfg.d_mlp)
    print("  hooks   ", [h for h in dir(bridge) if "hook" in h.lower()][:10])
    print("  run_with_cache:", hasattr(bridge, "run_with_cache"))
    print("  all methods:", [n for n in dir(bridge) if not n.startswith("_")][:60])
except Exception:
    traceback.print_exc()
