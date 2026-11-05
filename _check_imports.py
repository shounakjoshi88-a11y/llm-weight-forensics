import importlib

for m in ["transformer_lens", "circuitsvis", "sae_lens", "nnsight",
          "einops", "plotly", "fastapi", "uvicorn", "flask"]:
    try:
        mod = importlib.import_module(m)
        print("  {:<18} OK  {}".format(m, getattr(mod, "__version__", "")))
    except Exception as e:
        print("  {:<18} MISSING ({})".format(m, type(e).__name__))
