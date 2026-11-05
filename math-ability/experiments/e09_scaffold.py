"""E09: a base-100 chunked scaffold.

The failure map says the atoms are reliable at two digits and unreliable
beyond that. This scaffold only ever asks the model for:

  - a 2-digit by 1-digit product
  - a 2-digit plus 2-digit plus carry

and does the place-value bookkeeping outside the model. Each atom is asked
three ways and the majority is taken. The model does every arithmetic step; the
harness only sequences and shifts chunks.
"""

from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C

ADD_T = ["What is {a} + {b}?", "Compute {a} plus {b}.", "Add {a} and {b}."]
MUL_T = ["What is {a} x {b}?", "Compute {a} times {b}.", "Multiply {a} by {b}."]


class Engine:
    def __init__(self, p):
        self.p = p
        self.calls = 0

    def ask(self, templates, a, b):
        prompts = [C.chat_text(self.p.tok, t.format(a=a, b=b), "The answer is") for t in templates]
        reps = C.generate_batch(self.p, prompts, max_new_tokens=12, batch_size=60)
        vals = [C.last_int(r) for r in reps]
        self.calls += 1
        vals = [v for v in vals if v is not None]
        if not vals:
            return None
        c = Counter(vals).most_common()
        if len(c) > 1 and c[0][1] == c[1][1]:
            return vals[0]
        return c[0][0]

    def mul(self, a, b):
        return self.ask(MUL_T, a, b)

    def add(self, a, b):
        return self.ask(ADD_T, a, b)


def chunks(x: int, base: int = 100) -> list[int]:
    out = []
    while x:
        out.append(x % base)
        x //= base
    return out or [0]


def assemble(cs: list[int], base: int = 100) -> int:
    v = 0
    for c in reversed(cs):
        v = v * base + c
    return v


def chunk_add(engine: Engine, x: int, y: int, trace=None) -> int | None:
    a, b = chunks(x), chunks(y)
    carry, out = 0, []
    for i in range(max(len(a), len(b))):
        u = a[i] if i < len(a) else 0
        v = b[i] if i < len(b) else 0
        s = engine.add(u, v)
        if s is None:
            return None
        s = s + carry
        out.append(s % 100)
        carry = s // 100
        if trace is not None:
            trace.append((u, v, carry, s))
    if carry:
        out.append(carry)
    return assemble(out)


def chunk_mul_digit(engine: Engine, x: int, d: int, trace=None) -> int | None:
    a = chunks(x)
    carry, out = 0, []
    for i, c in enumerate(a):
        p = engine.mul(c, d)
        if p is None:
            return None
        p = p + carry
        out.append(p % 100)
        carry = p // 100
        if trace is not None:
            trace.append((c, d, carry, p))
    while carry:
        out.append(carry % 100)
        carry //= 100
    return assemble(out)


def chunk_mul(engine: Engine, a: int, b: int, trace=None) -> int | None:
    bd = chunks(b)
    total = 0
    for i, d in enumerate(bd):
        if d == 0:
            continue
        part = chunk_mul_digit(engine, a, d)
        if part is None:
            return None
        total = chunk_add(engine, total, part * (100 ** i))
        if total is None:
            return None
        if trace is not None:
            trace.append(("partial", i, d, part, total))
    return total


def main() -> int:
    p = C.load()
    eng = Engine(p)
    print("loaded", p.n_layers, "layers")

    cases = [(57892, 512), (51960, 293), (86057, 244), (20324, 139),
             (50028, 132), (22740, 563), (97323, 519), (82760, 925)]
    ok = 0
    out = []
    for a, b in cases:
        tr = []
        v = chunk_mul(eng, a, b, tr)
        good = v == a * b
        ok += good
        print(f"{a} x {b} = {a*b}  got {v}  {'OK' if good else 'X'}  calls {eng.calls}")
        out.append({"a": a, "b": b, "want": a * b, "got": v, "ok": good, "trace": tr})

    print(f"\nscaffold {ok}/{len(cases)}")
    C.save({"cases": out, "ok": ok, "n": len(cases)}, C.ROOT / "findings" / "e09_scaffold.json")
    print("wrote findings/e09_scaffold.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
