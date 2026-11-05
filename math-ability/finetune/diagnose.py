"""Where the fine-tuned model fails inside its own worked steps.

Exact match only says the answer is wrong. This reads the stored replies and
checks them against the canonical worked solution for the same problem, so the
failure is localised rather than just counted.

Three distinct things are measured per cell:

  exact        final answer equals the true answer
  locally true every stated column line is arithmetically true on its own
  internal     the reply agrees with itself: the final answer matches the
               digits it just wrote (addition), or the borrow chain is valid
               against the canonical solution (subtraction)

A reply can be locally true and still internally wrong, which is exactly what
happens when a borrow is dropped between columns or the final answer is
transcribed from the wrong digits.

Usage: python finetune/diagnose.py [findings/e31_finetune_eval.json]
"""

from __future__ import annotations

import importlib.util
import json
import re
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))
import common as C

_spec = importlib.util.spec_from_file_location(
    "ftdata", Path(__file__).resolve().parent / "data.py")
D = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(D)

MUL_LINE = re.compile(r"(\d+) x (\d) = (\d+)")
ADD_SUM = re.compile(r"Add the (?:results|shifted products):\s*([\d\s+]+)=\s*(\d+)")
FINAL = re.compile(r"The answer is (\d+)")
COL = re.compile(r"^[a-z][a-z ]*: \d")


def check_mul(text: str) -> dict:
    partial_ok = partial_n = 0
    for a, d, p in MUL_LINE.findall(text):
        partial_n += 1
        if int(a) * int(d) == int(p):
            partial_ok += 1
    m = ADD_SUM.search(text)
    sum_ok = sum(int(x) for x in re.findall(r"\d+", m.group(1))) == int(m.group(2)) \
        if m else None
    return {"partial_ok": partial_ok, "partial_n": partial_n,
            "sum_ok": sum_ok, "sum_present": m is not None}


def check_add(text: str) -> dict:
    ok = n = 0
    for m in re.finditer(r": (\d+) ([+-]) (\d+)(?: \+ (\d+))? = (\d+)", text):
        a, op, b, c, s = m.groups()
        val = int(a) + int(b) + (int(c) if c else 0) if op == "+" else int(a) - int(b)
        n += 1
        ok += val == int(s)
    return {"line_ok": ok, "line_n": n}


def add_internal(text: str) -> bool | None:
    """Does the final answer match the digits the reply wrote down?"""
    wrote = "".join(re.findall(r"write (\d)", text))[::-1]
    m = FINAL.search(text)
    if not wrote or not m:
        return None
    return int(wrote) == int(m.group(1))


def sub_chain(text: str, a: int, b: int) -> bool:
    """Do the column lines match the canonical borrow chain for a - b?"""
    want = [ln.strip() for ln in D.sub_cot(a, b).splitlines() if COL.match(ln.strip())]
    got = [ln.strip() for ln in text.splitlines() if COL.match(ln.strip())]
    return got == want


def pct(xs) -> str:
    xs = [x for x in xs if x is not None]
    return f"{sum(xs) / len(xs):.2f}" if xs else "n/a"


def main() -> int:
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else \
        ROOT / "findings" / "e31_finetune_eval.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    tuned = data.get("tuned")
    if not tuned:
        print("no tuned results in", path)
        return 1

    print(f"{'cell':10s} {'exact':>6s} {'steps':>6s} {'locally true':>13s} "
          f"{'internal':>9s} {'sum right':>10s} {'exact|partials':>15s}")
    examples = {}
    for cell, v in tuned.items():
        reps = v["natural_replies"]
        op, da, db = cell[:3], int(cell[3]), int(cell[5])
        rows = [r for r in C.make(op, da, db, 200, 20261215)
                if r["split"] == "heldout"][:len(reps)]
        wants = [r["want"] for r in rows]

        if op == "mul":
            st = [check_mul(r) for r in reps]
            with_steps = [s for s in st if s["partial_n"] > 0]
            with_sum = [s for s in with_steps if s["sum_present"]]
            clean = [i for i, s in enumerate(st)
                     if s["partial_n"] > 0 and s["partial_ok"] == s["partial_n"]]
            print(f"{cell:10s} {v['natural']:6.2f} {len(with_steps):6d} "
                  f"{pct([s['partial_ok'] == s['partial_n'] for s in with_steps]):>13s} "
                  f"{'':>9s} "
                  f"{pct([s['sum_ok'] for s in with_sum]):>10s} "
                  f"{pct([C.last_int(reps[i]) == wants[i] for i in clean]):>15s}")
        elif op == "add":
            st = [check_add(r) for r in reps]
            with_steps = [s for s in st if s["line_n"] > 0]
            internal = [add_internal(r) for r in reps]
            print(f"{cell:10s} {v['natural']:6.2f} {len(with_steps):6d} "
                  f"{pct([s['line_ok'] == s['line_n'] for s in with_steps]):>13s} "
                  f"{pct(internal):>9s} {'':>10s} {'':>15s}")
        else:
            st = [check_add(r) for r in reps]
            with_steps = [s for s in st if s["line_n"] > 0]
            chain = [sub_chain(r, rw["a"], rw["b"]) for r, rw in zip(reps, rows)]
            print(f"{cell:10s} {v['natural']:6.2f} {len(with_steps):6d} "
                  f"{pct([s['line_ok'] == s['line_n'] for s in with_steps]):>13s} "
                  f"{pct(chain):>9s} {'':>10s} {'':>15s}")

        for i, (r, w) in enumerate(zip(reps, wants)):
            if C.last_int(r) != w and cell not in examples:
                examples[cell] = (w, r)

    for cell in ("add5x5", "sub5x4", "mul5x3"):
        if cell in examples:
            want, reply = examples[cell]
            print(f"\nfailing {cell}, want {want}:")
            print("  " + reply.strip()[:600].replace("\n", " | "))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
