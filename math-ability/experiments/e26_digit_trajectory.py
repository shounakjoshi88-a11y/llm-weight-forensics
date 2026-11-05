"""E26: does the model compute the digits and then drift, or never compute them?

Every edit in this folder aimed at the readout. That only makes sense if the
residual already holds the right digit at each step. This measures that
directly, with no editing.

For each problem the correct answer is appended to the prompt, so the model is
never allowed to make an early mistake. At every digit position we ask:

    does the model's own argmax equal the true digit?     (teacher forced)
    is the true digit linearly decodable from the residual?  (ridge probe)
    how big is the margin on the true digit?

Three outcomes, three different stories:

    high teacher-forced accuracy, low free-running
        the model computes correctly and a single early error cascades.
        Edits should target the first digit and error recovery.
    low teacher-forced accuracy from the first digit
        the computation is not there, and no readout edit can add it.
    probe far below the model's own readout
        the residual does not carry the digit, so lm_head is already doing
        all a linear readout can do. This is the explanation for the nulls.

Also reports the free-running first-error position, which is where a cascading
failure actually starts.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C

N_TRAIN = 400
N_TEST = 200
LAMBDAS = [0.1, 1.0, 10.0, 100.0]
BATCH = 8


def fixed_length(op: str, da: int, db: int, ndigits: int,
                 need: int, seed: int) -> list[dict]:
    out, s = [], seed
    while len(out) < need:
        for r in C.make(op, da, db, 2000, s):
            if len(str(r["want"])) == ndigits:
                out.append(r)
                if len(out) >= need:
                    break
        s += 1
    return out


@torch.no_grad()
def forward_capture(p, texts: list[str], nd: int,
                    digit_ids: list[int]) -> tuple[dict, dict]:
    """Normed residual and digit logits at the nd predictor positions.

    The full vocab logits for a batch are far too large for a 6 GB card, so
    only the ten digit columns at the six positions of interest are kept.
    Left padding means every row ends at the same index, so the position
    arithmetic is identical for the whole batch.
    """
    R = {i: [] for i in range(nd)}
    LG = {i: [] for i in range(nd)}
    store: list[torch.Tensor] = []
    handle = p.model.model.norm.register_forward_hook(
        lambda m, a, o: store.append(o.detach().float().cpu()))
    try:
        for i in range(0, len(texts), BATCH):
            store.clear()
            enc = p.tok(texts[i:i + BATCH], return_tensors="pt",
                        padding=True).to(p.device)
            out = p.model(**enc)
            res = store[0]
            for k in range(nd):
                idx = -(nd - k) - 1
                R[k].append(res[:, idx, :])
                LG[k].append(out.logits[:, idx, :][:, digit_ids].float().cpu())
            del out
    finally:
        handle.remove()
    return ({k: torch.cat(v, 0).double() for k, v in R.items()},
            {k: torch.cat(v, 0) for k, v in LG.items()})


def standardize(X):
    mu = X.mean(0, keepdim=True)
    sd = X.std(0, keepdim=True).clamp_min(1e-6)
    return (X - mu) / sd, mu, sd


def ridge_fit(X, Y, lam):
    d = X.shape[1]
    A = X.T @ X + lam * torch.eye(d, dtype=X.dtype)
    return torch.linalg.solve(A, X.T @ Y)


def onehot(y, k=10):
    Y = torch.zeros(y.shape[0], k, dtype=torch.float64)
    Y[torch.arange(y.shape[0]), y] = 1.0
    return Y


def probe(Xtr, ytr, Xte, lam):
    Xtr, mu, sd = standardize(Xtr)
    Xte = (Xte - mu) / sd
    if int(ytr.min()) == int(ytr.max()):
        return torch.full((Xte.shape[0],), int(ytr[0]))
    Y = onehot(ytr) - onehot(ytr).mean(0, keepdim=True)
    W = ridge_fit(Xtr, Y, lam)
    return (Xte @ W).argmax(1)


def main() -> int:
    p = C.load()
    print("loaded", p.n_layers, "layers")
    digit_id = {d: p.tok(str(d), add_special_tokens=False).input_ids[0]
                for d in range(10)}

    tasks = [("add5x5", "add", 5, 5, 6), ("mul3x2", "mul", 3, 2, 5)]
    results = {}

    for name, op, da, db, nd in tasks:
        rows = fixed_length(op, da, db, nd, N_TRAIN + N_TEST, 20261201)
        tr = [r for r in rows if r["split"] == "train"][:N_TRAIN]
        te = [r for r in rows if r["split"] == "heldout"][:N_TEST]
        print(f"\n=== {name}: {len(tr)} train, {len(te)} held out, "
              f"{nd}-digit answers ===")
        if len(tr) < 200 or len(te) < 100:
            print("  not enough problems, skipping")
            continue

        pref = [C.chat_text(p.tok, *C.prompt(r, "answer")) for r in tr + te]
        # Generation emits a space before the first digit (" 141522"), so the
        # teacher-forced text must contain it or every position is off by one.
        texts = [s + " " + str(r["want"]) for s, r in zip(pref, tr + te)]

        n_tok = [len(p.tok(t, add_special_tokens=False).input_ids) for t in texts]
        ok = all(n_tok[i] - len(p.tok(pref[i], add_special_tokens=False).input_ids) == nd + 1
                 for i in range(len(texts)))
        print("  answers tokenise to one token per digit:", ok)
        if not ok:
            print("  aborting, tokenisation assumption broken")
            continue

        R, L = forward_capture(p, texts, nd, [digit_id[d] for d in range(10)])
        n_tr = len(tr)

        # ---- teacher forced, per digit position
        tf_acc, tf_margin = [], []
        for i in range(nd):
            logits = L[i]
            want = torch.tensor([int(str(r["want"])[i]) for r in tr + te])
            got = logits.argmax(1)
            tf_acc.append((got == want).float().mean().item())
            true_lg = logits.gather(1, want.unsqueeze(1)).squeeze(1)
            tf_margin.append((true_lg - logits.max(1).values).mean().item())

        # ---- free running, first error position
        reps = C.generate_batch(p, pref[n_tr:], max_new_tokens=12, batch_size=40)
        free_ok = sum(C.last_int(r) == r0["want"] for r, r0 in zip(reps, te))
        first_err = []
        for r, r0 in zip(reps, te):
            got = C.last_int(r)
            s_got = str(got) if got is not None else ""
            s_want = str(r0["want"])
            if s_got == s_want:
                continue
            k = 0
            while k < min(len(s_got), len(s_want)) and s_got[k] == s_want[k]:
                k += 1
            first_err.append(k)

        print(f"  free running exact {free_ok}/{len(te)} = "
              f"{free_ok/len(te):.2f}")
        if first_err:
            hist = {}
            for k in first_err:
                hist[k] = hist.get(k, 0) + 1
            print("  first error at digit " +
                  ", ".join(f"{k}:{v}" for k, v in sorted(hist.items())))

        # ---- probe, per digit position
        print(f"\n  {'digit':>5s} {'teacher':>8s} {'probe':>7s} "
              f"{'margin':>7s} {'chance':>7s} {'shuffled':>9s}")
        row = {"n_train": len(tr), "n_test": len(te),
               "free_exact": free_ok / len(te),
               "first_error": {str(k): first_err.count(k)
                               for k in sorted(set(first_err))},
               "positions": {}}
        for i in range(nd):
            X = R[i]
            Xtr, Xte = X[:n_tr], X[n_tr:]
            ytr = torch.tensor([int(str(r["want"])[i]) for r in tr])
            yte = torch.tensor([int(str(r["want"])[i]) for r in te])
            cut = int(len(tr) * 0.8)
            best = (None, -1.0)
            for lam in LAMBDAS:
                v = (probe(Xtr[:cut], ytr[:cut], Xtr[cut:], lam) == ytr[cut:])
                v = v.float().mean().item()
                if v > best[1]:
                    best = (lam, v)
            pr = probe(Xtr, ytr, Xte, best[0])
            pacc = (pr == yte).float().mean().item()
            g = torch.Generator().manual_seed(11)
            shuf = ytr[torch.randperm(len(ytr), generator=g)]
            sp = probe(Xtr, shuf, Xte, best[0])
            sacc = (sp == yte).float().mean().item()
            chance = max((yte == d).float().mean().item() for d in range(10))
            print(f"  {i:5d} {tf_acc[i]:8.2f} {pacc:7.2f} "
                  f"{tf_margin[i]:7.2f} {chance:7.2f} {sacc:9.2f}")
            row["positions"][i] = {"teacher_forced": tf_acc[i],
                                   "probe": pacc, "margin": tf_margin[i],
                                   "chance": chance, "shuffled": sacc,
                                   "lambda": best[0]}
        results[name] = row

    C.save(results, C.ROOT / "findings" / "e26_digit_trajectory.json")
    print("\nwrote findings/e26_digit_trajectory.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
