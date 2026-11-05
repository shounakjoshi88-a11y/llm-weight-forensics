"""E02: what shape are the errors?

Offline analysis of the e01 transcripts. No model calls. For addition and
subtraction it checks whether an error is exactly the no-carry or no-borrow
value, which is the signature of a dropped carry. For multiplication it checks
whether the error matches a single partial product.
"""

from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C


def nocarry(a: int, b: int) -> int:
    v, p = 0, 1
    while a or b:
        v += (a % 10 + b % 10) * p
        a //= 10
        b //= 10
        p *= 10
    return v


def noborrow(a: int, b: int) -> int:
    v, p = 0, 1
    while a or b:
        v += (a % 10 - b % 10) * p
        a //= 10
        b //= 10
        p *= 10
    return v


def main() -> int:
    d = json.load(open(C.ROOT / "findings" / "e01_failure_map.json", encoding="utf-8"))
    rows = []
    for cell, got in d["grid"].items():
        if cell.endswith(" bare"):
            continue
        for g in got:
            rows.append({**g, "cell": cell})

    print(f"{len(rows)} problems with answer prefill\n")

    print("addition: accuracy by carry count")
    by = defaultdict(lambda: [0, 0])
    for r in rows:
        if r["op"] != "add":
            continue
        n = len(r["carry_pos"])
        by[n][1] += 1
        by[n][0] += r["ok"]
    for n in sorted(by):
        k, m = by[n]
        print(f"  {n} carries: {k}/{m} = {k/m:.2f}")

    print("\naddition errors that equal the no-carry value")
    add = [r for r in rows if r["op"] == "add" and not r["ok"] and r["got"] is not None]
    hit = sum(r["got"] == nocarry(r["a"], r["b"]) for r in add)
    print(f"  {hit}/{len(add)} = {hit/max(1,len(add)):.2f}")

    print("\naddition errors that are the correct answer plus or minus a power of ten")
    pow10 = 0
    for r in add:
        diff = abs(r["got"] - r["want"])
        if diff and (diff & (diff - 1) == 0 or str(diff).strip("0") == "1"):
            pow10 += 1
    print(f"  {pow10}/{len(add)} = {pow10/max(1,len(add)):.2f}")

    print("\nsubtraction: accuracy by borrow count")
    by = defaultdict(lambda: [0, 0])
    for r in rows:
        if r["op"] != "sub":
            continue
        n = len(r["carry_pos"])
        by[n][1] += 1
        by[n][0] += r["ok"]
    for n in sorted(by):
        k, m = by[n]
        print(f"  {n} borrows: {k}/{m} = {k/m:.2f}")

    print("\nsubtraction errors that equal the no-borrow value")
    sub = [r for r in rows if r["op"] == "sub" and not r["ok"] and r["got"] is not None]
    hit = sum(r["got"] == noborrow(r["a"], r["b"]) for r in sub)
    print(f"  {hit}/{len(sub)} = {hit/max(1,len(sub)):.2f}")

    print("\nsubtraction errors with the correct ones digit")
    ones = sum((r["got"] % 10) == (r["want"] % 10) for r in sub)
    print(f"  {ones}/{len(sub)} = {ones/max(1,len(sub)):.2f}")

    print("\nmultiplication: accuracy by digit pair")
    by = defaultdict(lambda: [0, 0])
    for r in rows:
        if r["op"] != "mul":
            continue
        key = f"{r['da']}x{r['db']}"
        by[key][1] += 1
        by[key][0] += r["ok"]
    for key in sorted(by):
        k, m = by[key]
        print(f"  {key}: {k}/{m} = {k/m:.2f}")

    print("\nmultiplication errors that match a partial product a*b_hi or a*b_lo")
    mul = [r for r in rows if r["op"] == "mul" and not r["ok"] and r["got"] is not None]
    hit = 0
    for r in mul:
        a, b = r["a"], r["b"]
        parts = {a * (b // 10), a * (b % 10), b * (a // 10), b * (a % 10)}
        if r["got"] in parts:
            hit += 1
    print(f"  {hit}/{len(mul)} = {hit/max(1,len(mul)):.2f}")

    print("\nmultiplication errors within 10 percent of the answer")
    near = sum(abs(r["got"] - r["want"]) <= max(2, abs(r["want"]) // 10) for r in mul)
    print(f"  {near}/{len(mul)} = {near/max(1,len(mul)):.2f}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
