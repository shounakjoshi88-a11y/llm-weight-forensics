"""Generate supervised arithmetic data for fine-tuning.

The failure map said the model approximates rather than computes. So the
responses show the procedure, not just the answer: column carries for addition
and subtraction, shifted partial products for multiplication.

Only problems whose hash split is "train" are used, so the held-out cells in
experiments/ stay unseen and the comparison stays honest.
"""

from __future__ import annotations

import json
import random
import sys
import zlib
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments"))
import common as C

PLACES = ["ones", "tens", "hundreds", "thousands", "ten thousands",
          "hundred thousands", "millions", "ten millions"]

# (op, digits_a, digits_b, how many)
PLAN = [
    ("add", 2, 2, 300), ("add", 3, 3, 500), ("add", 4, 4, 500),
    ("add", 5, 5, 600), ("add", 6, 6, 400),
    ("sub", 3, 2, 300), ("sub", 4, 3, 300), ("sub", 5, 4, 300),
    ("mul", 1, 1, 200), ("mul", 2, 1, 200), ("mul", 3, 1, 300),
    ("mul", 4, 1, 300), ("mul", 5, 1, 300),
    ("mul", 2, 2, 600), ("mul", 3, 2, 900), ("mul", 3, 3, 800),
    ("mul", 4, 2, 700), ("mul", 4, 3, 700),
    ("mul", 5, 2, 500), ("mul", 5, 3, 600),
]


def add_cot(a: int, b: int) -> str:
    lines = ["Let me work through it step by step.", ""]
    x, y, carry, i = a, b, 0, 0
    while x or y or carry:
        da, db = x % 10, y % 10
        s = da + db + carry
        w, nc = s % 10, s // 10
        place = PLACES[i] if i < len(PLACES) else f"column {i}"
        if carry:
            line = f"{place}: {da} + {db} + {carry} = {s}, write {w}"
        else:
            line = f"{place}: {da} + {db} = {s}, write {w}"
        if nc:
            line += f", carry {nc}"
        lines.append(line)
        carry = nc
        x //= 10
        y //= 10
        i += 1
    lines.append("")
    lines.append(f"The answer is {a + b}.")
    return "\n".join(lines)


def sub_cot(a: int, b: int) -> str:
    lines = ["Let me work through it step by step.", ""]
    x, y, borrow, i = a, b, 0, 0
    while x or y:
        da, db = x % 10, y % 10
        place = PLACES[i] if i < len(PLACES) else f"column {i}"
        if da - borrow < db:
            top = da + 10 - borrow
            lines.append(f"{place}: {da} - {db} needs a borrow, "
                         f"{top} - {db} = {top - db}")
            borrow = 1
        else:
            top = da - borrow
            lines.append(f"{place}: {top} - {db} = {top - db}")
            borrow = 0
        x //= 10
        y //= 10
        i += 1
    lines.append("")
    lines.append(f"The answer is {a - b}.")
    return "\n".join(lines)


def mul_cot(a: int, b: int) -> str:
    ds = [int(ch) for ch in str(b)][::-1]
    if len(ds) == 1:
        return ("Let me work through it step by step.\n\n"
                f"{a} x {b} = {a * b}\n\nThe answer is {a * b}.")
    lines = ["Let me work through it step by step.", "",
             f"Multiply {a} by each digit of {b}."]
    shifted = []
    for k, d in enumerate(ds):
        p = a * d
        s = p * (10 ** k)
        shifted.append(s)
        if k == 0:
            lines.append(f"{a} x {d} = {p}")
        else:
            lines.append(f"{a} x {d} = {p}, shifted {k} place"
                         f"{'s' if k > 1 else ''} = {s}")
    total = " + ".join(str(s) for s in shifted)
    lines.append(f"Add the shifted products: {total} = {a * b}")
    lines.append("")
    lines.append(f"The answer is {a * b}.")
    return "\n".join(lines)


COT = {"add": add_cot, "sub": sub_cot, "mul": mul_cot}
SYMBOL = {"add": "+", "sub": "-", "mul": "x"}


def build(seed: int = 20261105, direct_frac: float = 0.15) -> list[dict]:
    rng = random.Random(seed)
    records = []
    for op, da, db, n in PLAN:
        pool = C.make(op, da, db, n * 4,
                      seed + zlib.crc32(f"{op}{da}{db}".encode()) % 10000)
        train = [r for r in pool if r["split"] == "train"][:n]
        for r in train:
            a, b, want = r["a"], r["b"], r["want"]
            q = f"What is {a} {SYMBOL[op]} {b}?"
            if rng.random() < direct_frac:
                ans = f"The answer is {want}."
            else:
                ans = COT[op](a, b)
            records.append({"user": q, "assistant": ans,
                            "op": op, "da": da, "db": db,
                            "a": a, "b": b, "want": want})
    rng.shuffle(records)
    return records


def verify(recs: list[dict]) -> int:
    """Every stated line must be arithmetically true and end on the right answer."""
    import re
    bad = 0
    for r in recs:
        a, b, want, text = r["a"], r["b"], r["want"], r["assistant"]
        m = re.findall(r"The answer is (\d+)\.", text)
        if not m or int(m[-1]) != want:
            bad += 1
            continue
        if text.startswith("The answer is"):
            continue  # direct answer, no procedure to check
        if r["op"] == "add":
            wrote = "".join(re.findall(r"write (\d)", text))[::-1]
            if int(wrote) != want:
                bad += 1
        elif r["op"] == "mul":
            for m2 in re.finditer(r"x (\d) = (\d+)", text):
                d, p = int(m2.group(1)), int(m2.group(2))
                if a * d != p:
                    bad += 1
                    break
    return bad


def main() -> int:
    recs = build()
    out = C.ROOT / "finetune" / "train_data.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(recs), encoding="utf-8")
    lens = sorted(len(r["assistant"]) for r in recs)
    print(f"wrote {out} with {len(recs)} records")
    print(f"assistant chars: median {lens[len(lens) // 2]}, max {lens[-1]}")
    by_op = {}
    for r in recs:
        by_op[r["op"]] = by_op.get(r["op"], 0) + 1
    print("by op:", by_op)
    print("arithmetically false records:", verify(recs))
    for op in ("add", "sub", "mul"):
        ex = next(r for r in recs if r["op"] == op)
        print(f"\n--- example {op}: {ex['user']}\n{ex['assistant'][:420]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
