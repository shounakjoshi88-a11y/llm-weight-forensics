"""Shared harness for the arithmetic editing experiments.

Design notes:

- problems are generated deterministically and split by hash, so a problem is
  either train or held out for its whole life
- every number is a sequence of single digit tokens in this tokenizer, which
  makes digit-level scoring and digit-level edits clean
- generation is batched with left padding; scoring is exact final-integer match
"""

from __future__ import annotations

import hashlib
import json
import math
import random
import re
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
MODEL_WEIGHTS = ROOT.parent
sys.path.insert(0, str(MODEL_WEIGHTS))

from probe.core import Probe  # noqa: E402

MODEL = str(MODEL_WEIGHTS / "Qwen2.5-0.5B-Instruct")

SYSTEM = (
    "You are a careful calculator. When a final answer is requested, write it "
    "as a plain integer with no commas and no other text."
)


class FastModel:
    """Probe-compatible wrapper that uses SDPA attention for generation.

    Probe uses eager attention because the reading experiments need the
    attention weights. Generation does not, and SDPA is several times faster.
    """

    def __init__(self, model_dir: str, device: str = "cuda"):
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.dir = model_dir
        self.tok = AutoTokenizer.from_pretrained(model_dir)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_dir, dtype=torch.bfloat16, device_map=device,
            attn_implementation="sdpa",
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

    def encode(self, text: str) -> torch.Tensor:
        return self.tok(text, return_tensors="pt").input_ids.to(self.device)

    def decode(self, ids) -> str:
        return self.tok.decode(ids, skip_special_tokens=False)


# ----------------------------------------------------------------------
# model
# ----------------------------------------------------------------------
def load(device: str = "cuda", fast: bool = True):
    if fast:
        p = FastModel(MODEL, device=device)
    else:
        p = Probe(MODEL, device=device, dtype=torch.bfloat16)
    if p.tok.pad_token is None:
        p.tok.pad_token = p.tok.eos_token
    p.tok.padding_side = "left"
    return p


def chat_text(tok, user: str, prefill: str = "") -> str:
    msgs = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": user},
    ]
    text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    return text + prefill


@torch.no_grad()
def generate_batch(
    p: Probe,
    prompts: list[str],
    max_new_tokens: int = 32,
    batch_size: int = 60,
    min_new_tokens: int = 0,
) -> list[str]:
    """Greedy generation for a list of already-templated prompts.

    min_new_tokens suppresses EOS for that many tokens. It is how the harness
    tests whether a model is stopping before it has finished reasoning.
    """
    out: list[str] = []
    for i in range(0, len(prompts), batch_size):
        chunk = prompts[i : i + batch_size]
        enc = p.tok(chunk, return_tensors="pt", padding=True).to(p.device)
        ids = p.model.generate(
            **enc,
            max_new_tokens=max_new_tokens,
            min_new_tokens=min_new_tokens,
            do_sample=False,
            pad_token_id=p.tok.pad_token_id,
        )
        prompt_len = enc["input_ids"].shape[1]
        for row in ids:
            out.append(p.tok.decode(row[prompt_len:], skip_special_tokens=True))
    return out


# ----------------------------------------------------------------------
# scoring
# ----------------------------------------------------------------------
_INT = re.compile(r"-?\d[\d,]*")


def integers(text: str) -> list[int]:
    vals = []
    for m in _INT.findall(text.replace(" ", "")):
        try:
            vals.append(int(m.replace(",", "")))
        except ValueError:
            pass
    return vals


def last_int(text: str) -> int | None:
    vals = integers(text)
    return vals[-1] if vals else None


def first_int(text: str) -> int | None:
    vals = integers(text)
    return vals[0] if vals else None


def score(reply: str, want: int) -> dict:
    got = last_int(reply)
    return {
        "got": got,
        "ok": got == want,
        "close": got is not None and abs(got - want) <= max(2, abs(want) // 100),
        "n_ints": len(integers(reply)),
    }


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (0.0, 0.0)
    ph = k / n
    d = 1 + z * z / n
    c = ph + z * z / (2 * n)
    r = z * math.sqrt(ph * (1 - ph) / n + z * z / (4 * n * n))
    return ((c - r) / d, (c + r) / d)


# ----------------------------------------------------------------------
# problems
# ----------------------------------------------------------------------
def split_of(op: str, a: int, b: int) -> str:
    h = hashlib.sha1(f"{op}:{a}:{b}".encode()).hexdigest()
    return "train" if int(h, 16) % 100 < 50 else "heldout"


def _rand_with_digits(rng: random.Random, d: int) -> int:
    lo = 10 ** (d - 1) if d > 1 else 0
    hi = 10**d - 1
    return rng.randint(lo, hi)


def carries(a: int, b: int) -> list[int]:
    """Column positions, from the ones column, where a carry is produced."""
    pos, c = [], 0
    while a or b:
        s = a % 10 + b % 10 + c
        if s >= 10:
            pos.append(len(pos))
        c = 1 if s >= 10 else 0
        a //= 10
        b //= 10
    return pos


def borrows(a: int, b: int) -> list[int]:
    """Column positions, from the ones column, where a borrow is taken."""
    pos = []
    while a or b:
        da, db = a % 10, b % 10
        if da < db:
            pos.append(len(pos))
            # propagate the borrow left
            a -= 10
        a //= 10
        b //= 10
    return pos


def make(op: str, da: int, db: int, n: int, seed: int) -> list[dict]:
    rng = random.Random(seed)
    seen: set[tuple[int, int]] = set()
    rows: list[dict] = []
    guard = 0
    while len(rows) < n and guard < n * 200:
        guard += 1
        if op in ("add", "sub"):
            a = _rand_with_digits(rng, da)
            b = _rand_with_digits(rng, db)
            if da == db and rng.random() < 0.5 and a < b:
                a, b = b, a
            if op == "sub" and a < b:
                a, b = b, a
            if (a, b) in seen:
                continue
            seen.add((a, b))
            want = a + b if op == "add" else a - b
            lab = carries(a, b) if op == "add" else borrows(a, b)
            rows.append({
                "op": op, "a": a, "b": b, "want": want,
                "da": da, "db": db, "cols": len(lab),
                "carry_pos": lab,
            })
        elif op == "mul":
            a = _rand_with_digits(rng, da)
            b = _rand_with_digits(rng, db)
            if (a, b) in seen:
                continue
            seen.add((a, b))
            rows.append({
                "op": op, "a": a, "b": b, "want": a * b,
                "da": da, "db": db, "cols": 0, "carry_pos": [],
            })
        elif op == "div":
            q = _rand_with_digits(rng, max(1, da - db))
            b = _rand_with_digits(rng, db)
            a = q * b
            if len(str(a)) != da or (a, b) in seen:
                continue
            seen.add((a, b))
            rows.append({
                "op": op, "a": a, "b": b, "want": a // b,
                "da": da, "db": db, "cols": 0, "carry_pos": [],
            })
    for r in rows:
        r["split"] = split_of(op, r["a"], r["b"])
    return rows


def prompt(row: dict, fmt: str) -> tuple[str, str]:
    a, b = row["a"], row["b"]
    op = {"add": "+", "sub": "-", "mul": "x", "div": "/"}[row["op"]]
    if fmt == "bare":
        return f"What is {a} {op} {b}?", ""
    if fmt == "answer":
        return f"What is {a} {op} {b}?", "The answer is"
    if fmt == "steps":
        return f"What is {a} {op} {b}? Work through it step by step, then give the final answer.", ""
    if fmt == "column":
        return f"Compute {a} {op} {b} in column form. Write each column, then the total.", ""
    raise ValueError(fmt)


def save(obj, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2), encoding="utf-8")
    return path
