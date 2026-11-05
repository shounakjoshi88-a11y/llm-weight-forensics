#!/usr/bin/env python
"""Download the checkpoints this repo analyzes.

Weights are not committed (GitHub rejects files over 100 MB), so run this once
after cloning. Total download is about 2.15 GB.

    python fetch_models.py
"""

import sys

MODELS = [
    ("Qwen/Qwen2.5-0.5B", "Qwen2.5-0.5B", "base model, 494M params"),
    ("Qwen/Qwen2.5-0.5B-Instruct", "Qwen2.5-0.5B-Instruct", "SFT of the base, 494M params"),
    ("HuggingFaceTB/SmolLM2-135M", "SmolLM2-135M", "negative control, 135M params"),
]


def main() -> int:
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        print("missing dependency. install with:\n  pip install huggingface_hub")
        return 1

    for repo_id, local_dir, note in MODELS:
        print(f"=== {repo_id}  ({note})", flush=True)
        path = snapshot_download(repo_id=repo_id, local_dir=local_dir)
        print(f"    -> {path}", flush=True)

    print("\ndone. run: python analyze.py Qwen2.5-0.5B-Instruct")
    return 0


if __name__ == "__main__":
    sys.exit(main())
