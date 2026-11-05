"""Evaluate base against the fine-tuned adapter on held-out cells.

Same held-out split as every other experiment, so the comparison is against
problems neither model has seen. Two prompt modes: natural, where the model
answers however it wants, and the `The answer is` prefill used by the old
failure map.

Run with no argument to evaluate the base model only, which is the reference
row in the table.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments"))
import common as C

ADAPTER = C.ROOT / "finetune" / "adapter"
CELLS = [("add", 1, 1), ("add", 2, 2), ("add", 3, 3), ("add", 4, 4),
         ("add", 5, 5), ("add", 6, 6),
         ("sub", 3, 2), ("sub", 5, 4),
         ("mul", 1, 1), ("mul", 2, 2), ("mul", 3, 2), ("mul", 3, 3),
         ("mul", 4, 3), ("mul", 5, 3)]
N = 40
MAX_NEW = 256
SYMBOL = {"add": "+", "sub": "-", "mul": "x"}


def evaluate(p, cells, tag: str, use_adapter: bool) -> dict:
    if use_adapter:
        from peft import PeftModel
        p.model = PeftModel.from_pretrained(p.model, str(ADAPTER))
        p.model.eval()
        for prm in p.model.parameters():
            prm.requires_grad_(False)

    out = {}
    print(f"\n[{tag}]")
    print(f"  {'cell':8s} {'natural':>8s} {'prefill':>8s} {'chars':>6s}")
    for op, da, db in cells:
        rows = [r for r in C.make(op, da, db, N * 5, 20261215)
                if r["split"] == "heldout"][:N]
        qs = [f"What is {r['a']} {SYMBOL[op]} {r['b']}?" for r in rows]
        wants = [r["want"] for r in rows]
        nat = C.generate_batch(p, [C.chat_text(p.tok, q, "") for q in qs],
                               max_new_tokens=MAX_NEW, batch_size=32)
        pre = C.generate_batch(p, [C.chat_text(p.tok, q, "The answer is")
                                   for q in qs],
                               max_new_tokens=MAX_NEW, batch_size=32)
        a = sum(C.last_int(r) == w for r, w in zip(nat, wants)) / len(wants)
        b = sum(C.last_int(r) == w for r, w in zip(pre, wants)) / len(wants)
        med = sorted(len(r) for r in nat)[len(nat) // 2]
        print(f"  {op}{da}x{db:<5d} {a:8.2f} {b:8.2f} {med:6d}")
        out[f"{op}{da}x{db}"] = {"natural": a, "prefill": b,
                                 "median_chars": med,
                                 "natural_replies": nat,
                                 "prefill_replies": pre}
    return out


def target(p, tag: str) -> None:
    q = "What is 57892 x 512?"
    r = C.generate_batch(p, [C.chat_text(p.tok, q, "")], max_new_tokens=MAX_NEW)
    print(f"\n[{tag}] 57892 x 512 -> {C.last_int(r[0])}")
    print("  " + r[0].strip()[:300].replace("\n", " | "))


def word_problems(p, tag: str) -> dict:
    """Check for damage outside bare arithmetic."""
    from e17_word_problems import problems
    rows = problems(60, 20261121)
    qs = [q for q, _ in rows]
    wants = [w for _, w in rows]
    nat = C.generate_batch(p, [C.chat_text(p.tok, q, "") for q in qs],
                           max_new_tokens=384, batch_size=20)
    pre = C.generate_batch(p, [C.chat_text(p.tok, q, "Step 1: ") for q in qs],
                           max_new_tokens=384, batch_size=20)
    a = sum(C.last_int(r) == w for r, w in zip(nat, wants)) / len(wants)
    b = sum(C.last_int(r) == w for r, w in zip(pre, wants)) / len(wants)
    print(f"\n[{tag}] word problems: natural {a:.2f}, `Step 1: ` prefill {b:.2f}")
    return {"natural": a, "prefill": b, "natural_replies": nat}


def main() -> int:
    p = C.load()
    use = "--base-only" not in sys.argv
    base = evaluate(p, CELLS, "base", use_adapter=False)
    target(p, "base")
    out = {"base": base, "base_word": word_problems(p, "base")}
    if use and ADAPTER.exists():
        tuned = evaluate(p, CELLS, "tuned", use_adapter=True)
        target(p, "tuned")
        out["tuned"] = tuned
        out["tuned_word"] = word_problems(p, "tuned")
        C.save(out, C.ROOT / "findings" / "e31_finetune_eval.json")
        print("\nwrote findings/e31_finetune_eval.json")
        print(f"\n  {'cell':8s} {'base':>6s} {'tuned':>6s} {'delta':>7s}")
        for k in base:
            d = tuned[k]["natural"] - base[k]["natural"]
            print(f"  {k:8s} {base[k]['natural']:6.2f} "
                  f"{tuned[k]['natural']:6.2f} {d:+7.2f}")
    else:
        C.save(out, C.ROOT / "findings" / "e31_finetune_eval.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
