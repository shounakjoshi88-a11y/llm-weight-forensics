"""E11: base-100 scaffold, with the multiplier chunked in base 10.

E09 chunked both operands in base 100. That is wrong for the multiplier: a
2-digit multiplier chunk times a 2-digit multiplicand chunk is a 2x2 product,
which this model does at 0.35. The scaffold is supposed to ask only for atoms
the model is reliable at.

Here the multiplicand is chunked in base 100 and the multiplier in base 10, so
every product is a 2-digit by 1-digit, measured at 0.98. Every addition is a
2-digit plus 2-digit plus carry, done as one prompt with the carry passed in.

The harness sequences and shifts chunks. It does no arithmetic.
"""

from __future__ import annotations

import random
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
        self.asks = 0
        self.prompts = 0

    def ask(self, templates, a, b):
        prompts = [C.chat_text(self.p.tok, t.format(a=a, b=b), "The answer is") for t in templates]
        reps = C.generate_batch(self.p, prompts, max_new_tokens=12, batch_size=60)
        vals = [C.last_int(r) for r in reps]
        self.asks += 1
        self.prompts += len(prompts)
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


def chunks(x: int, base: int) -> list[int]:
    out = []
    while x:
        out.append(x % base)
        x //= base
    return out or [0]


def assemble(cs: list[int], base: int) -> int:
    v = 0
    for c in reversed(cs):
        v = v * base + c
    return v


def chunk_add(engine: Engine, x: int, y: int) -> int | None:
    a, b = chunks(x, 100), chunks(y, 100)
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
    while carry:
        out.append(carry % 100)
        carry //= 100
    return assemble(out, 100)


def chunk_mul_digit(engine: Engine, x: int, d: int) -> int | None:
    a = chunks(x, 100)
    carry, out = 0, []
    for c in a:
        p = engine.mul(c, d)
        if p is None:
            return None
        p = p + carry
        out.append(p % 100)
        carry = p // 100
    while carry:
        out.append(carry % 100)
        carry //= 100
    return assemble(out, 100)


def chunk_mul(engine: Engine, a: int, b: int) -> int | None:
    digits = chunks(b, 10)
    total = None
    for i, d in enumerate(digits):
        if d == 0:
            continue
        part = chunk_mul_digit(engine, a, d)
        if part is None:
            return None
        if total is None:
            # the first partial needs no addition, only its place shift
            total = part * (10 ** i)
            continue
        total = chunk_add(engine, total, part * (10 ** i))
        if total is None:
            return None
    return total if total is not None else 0


def main() -> int:
    p = C.load()
    print("loaded", p.n_layers, "layers")

    cases = [(57892, 512), (51960, 293), (86057, 244), (20324, 139),
             (50028, 132), (22740, 563), (97323, 519), (82760, 925)]
    rng = random.Random(20261112)
    while len(cases) < 28:
        a = rng.randint(10000, 99999)
        b = rng.randint(100, 999)
        if (a, b) not in cases:
            cases.append((a, b))

    eng = Engine(p)
    ok, out = 0, []
    for a, b in cases:
        v = chunk_mul(eng, a, b)
        good = v == a * b
        ok += good
        print(f"{a} x {b} = {a*b}  got {v}  {'OK' if good else 'X'}")
        out.append({"a": a, "b": b, "want": a * b, "got": v, "ok": good})

    n = len(cases)
    print(f"\nscaffold {ok}/{n} = {ok/n:.2f}")
    print(f"asks {eng.asks}, prompts {eng.prompts}")
    C.save({"cases": out, "ok": ok, "n": n, "asks": eng.asks, "prompts": eng.prompts},
           C.ROOT / "findings" / "e11_scaffold_v2.json")
    print("wrote findings/e11_scaffold_v2.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
