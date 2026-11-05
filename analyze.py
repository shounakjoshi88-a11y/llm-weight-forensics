#!/usr/bin/env python
"""
Weight-space forensics for small LLMs.

No torch required. Reads safetensors headers directly and streams tensors one at
a time, so a 0.5B model costs ~2 MB of RAM per tensor instead of 2 GB total.

Usage:
    python analyze.py <model_dir>                     # fingerprint a single model
    python analyze.py --compare <base_dir> <ft_dir>   # lineage / delta analysis
    python analyze.py --gguf <file.gguf>              # quantization signature
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import struct
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

# --------------------------------------------------------------------------
# safetensors reader (bf16 / fp16 / fp32, framework-free)
# --------------------------------------------------------------------------

DTYPE_MAP = {
    "F32": np.float32,
    "F16": np.float16,
    "F64": np.float64,
    "I64": np.int64,
    "I32": np.int32,
    "I16": np.int16,
    "I8": np.int8,
    "U8": np.uint8,
    "BOOL": np.bool_,
}
DTYPE_SIZE = {"F64": 8, "F32": 4, "F16": 2, "BF16": 2, "I64": 8, "I32": 4,
              "I16": 2, "I8": 1, "U8": 1, "BOOL": 1}


class SafeTensors:
    """Minimal streaming reader for .safetensors files."""

    def __init__(self, path: str | os.PathLike):
        self.path = str(path)
        with open(self.path, "rb") as f:
            raw = f.read(8)
            if len(raw) < 8:
                raise ValueError(f"{path}: too short to be safetensors")
            (n,) = struct.unpack("<Q", raw)
            if n > 100_000_000:
                raise ValueError(f"{path}: implausible header length {n}")
            self.header = json.loads(f.read(n).decode("utf-8"))
        self.data_start = 8 + n
        self.tensors = {k: v for k, v in self.header.items() if k != "__metadata__"}
        self.metadata = self.header.get("__metadata__", {}) or {}

    def names(self):
        return list(self.tensors)

    def info(self, name):
        return self.tensors[name]

    def read(self, name) -> np.ndarray:
        t = self.tensors[name]
        dtype = t["dtype"]
        shape = tuple(t["shape"])
        start, end = t["data_offsets"]
        nbytes = end - start
        with open(self.path, "rb") as f:
            f.seek(self.data_start + start)
            buf = f.read(nbytes)

        if dtype == "BF16":
            u16 = np.frombuffer(buf, dtype=np.uint16)
            # bf16 -> f32 is just a 16-bit left shift into the exponent space
            out = (u16.astype(np.uint32) << 16).view(np.float32)
        elif dtype in DTYPE_MAP:
            out = np.frombuffer(buf, dtype=DTYPE_MAP[dtype]).astype(
                np.float32 if DTYPE_MAP[dtype] not in (np.int64, np.bool_) else DTYPE_MAP[dtype]
            )
        else:
            raise NotImplementedError(f"dtype {dtype}")

        return out.reshape(shape) if shape else out

    def __len__(self):
        return len(self.tensors)


def find_weight_files(model_dir: Path) -> list[Path]:
    single = model_dir / "model.safetensors"
    if single.exists():
        return [single]
    shards = sorted(model_dir.glob("model-*-of-*.safetensors"))
    if shards:
        return shards
    any_st = sorted(model_dir.glob("*.safetensors"))
    if any_st:
        return any_st
    raise FileNotFoundError(f"no .safetensors in {model_dir}")


class Model:
    """Union view over one or more safetensors shards."""

    def __init__(self, model_dir: str | os.PathLike):
        self.dir = Path(model_dir)
        self.shards = [SafeTensors(p) for p in find_weight_files(self.dir)]
        self.index = {}
        for sh in self.shards:
            for n in sh.names():
                self.index[n] = sh
        self.config = {}
        cfg = self.dir / "config.json"
        if cfg.exists():
            self.config = json.loads(cfg.read_text(encoding="utf-8"))

    def names(self):
        return list(self.index)

    def read(self, name):
        return self.index[name].read(name)

    def __len__(self):
        return len(self.index)


# --------------------------------------------------------------------------
# fingerprinting
# --------------------------------------------------------------------------

def human(n: float) -> str:
    for unit in ("", "K", "M", "B", "T"):
        if abs(n) < 1000:
            return f"{n:.2f}{unit}"
        n /= 1000
    return f"{n:.2f}P"


def arch_fingerprint(m: Model) -> dict:
    cfg = m.config
    return {
        "architectures": cfg.get("architectures"),
        "model_type": cfg.get("model_type"),
        "hidden_size": cfg.get("hidden_size"),
        "num_layers": cfg.get("num_hidden_layers"),
        "num_heads": cfg.get("num_attention_heads"),
        "num_kv_heads": cfg.get("num_key_value_heads"),
        "intermediate_size": cfg.get("intermediate_size"),
        "vocab_size": cfg.get("vocab_size"),
        "tie_word_embeddings": cfg.get("tie_word_embeddings"),
        "rope_theta": cfg.get("rope_theta"),
        "max_position": cfg.get("max_position_embeddings"),
    }


def tokenizer_fingerprint(model_dir: Path) -> dict:
    """Hash the tokenizer artifacts. This is the strongest family fingerprint."""
    out = {}
    for fn in ("tokenizer.json", "vocab.json", "merges.txt", "tokenizer.model"):
        p = model_dir / fn
        if not p.exists():
            continue
        h = hashlib.sha256()
        size = 0
        with open(p, "rb") as f:
            while chunk := f.read(1 << 20):
                h.update(chunk)
                size += len(chunk)
        out[fn] = {"sha256": h.hexdigest()[:32], "bytes": size}
    return out


def tensor_inventory(m: Model) -> list[tuple[str, str, tuple, int]]:
    rows = []
    for n in m.names():
        t = m.index[n].info(n)
        rows.append((n, t["dtype"], tuple(t["shape"]), int(np.prod(t["shape"]))))
    return rows


def param_count(m: Model) -> int:
    return sum(int(np.prod(m.index[n].info(n)["shape"])) for n in m.names())


def per_layer_stats(m: Model, sample: int | None = None) -> dict:
    """Norms and outlier structure per tensor."""
    stats = {}
    names = m.names()
    if sample:
        names = names[:: max(1, len(names) // sample)]
    for n in names:
        w = m.read(n).astype(np.float32, copy=False)
        if w.size == 0:
            continue
        absmax = float(np.abs(w).max())
        rms = float(np.sqrt(np.mean(w * w)))
        # "outlier channel" = a column whose norm dwarfs its peers
        if w.ndim >= 2:
            flat = w.reshape(-1, w.shape[-1])
            col = np.linalg.norm(flat, axis=0)
            med = float(np.median(col)) or 1e-12
            outlier_ratio = float(col.max() / med)
        else:
            outlier_ratio = 1.0
        stats[n] = {
            "shape": list(w.shape),
            "absmax": absmax,
            "rms": rms,
            "mean": float(w.mean()),
            "outlier_ratio": outlier_ratio,
            "kurtosis": float(_kurtosis(w)),
        }
    return stats


def _kurtosis(x: np.ndarray) -> float:
    x = x.ravel().astype(np.float64)
    mu = x.mean()
    d = x - mu
    var = (d * d).mean()
    if var <= 0:
        return 0.0
    return float((d ** 4).mean() / (var * var) - 3.0)


# --------------------------------------------------------------------------
# lineage / delta analysis
# --------------------------------------------------------------------------

def compare(base_dir: str, ft_dir: str) -> dict:
    a, b = Model(base_dir), Model(ft_dir)
    shared = sorted(set(a.names()) & set(b.names()))
    only_a = sorted(set(a.names()) - set(b.names()))
    only_b = sorted(set(b.names()) - set(a.names()))

    rows = []
    for n in shared:
        # float32, not float64: the embedding matrix is 136M params and float64
        # would spike to ~4 GB for the pair. numpy's pairwise summation keeps
        # cosine accurate well past 1e-7 in float32.
        wa = a.read(n).astype(np.float32, copy=False)
        wb = b.read(n).astype(np.float32, copy=False)
        if wa.shape != wb.shape:
            rows.append({"name": n, "shape_mismatch": [list(wa.shape), list(wb.shape)]})
            continue
        fa, fb = wa.ravel(), wb.ravel()
        na = float(np.sqrt(np.dot(fa, fa)))
        nb = float(np.sqrt(np.dot(fb, fb)))
        cos = float(np.dot(fa, fb) / (na * nb)) if na and nb else float("nan")
        d = fa - fb
        rows.append({
            "name": n,
            "cosine": cos,
            "rel_delta": float(np.sqrt(np.dot(d, d)) / na) if na else float("nan"),
            "delta_rms": float(np.sqrt(np.mean(d * d))),
            "n": int(fa.size),
        })
        del wa, wb, fa, fb, d

    valid = [r for r in rows if "cosine" in r and np.isfinite(r["cosine"])]
    total = sum(r["n"] for r in valid)
    wcos = sum(r["cosine"] * r["n"] for r in valid) / total if total else float("nan")

    # did anyone permute neurons? a strong reorder signal = high cosine drop
    # that vanishes once rows are sorted by norm
    return {
        "shared_tensors": len(shared),
        "only_in_a": only_a,
        "only_in_b": only_b,
        "param_weighted_cosine": wcos,
        "mean_rel_delta": float(np.mean([r["rel_delta"] for r in valid])) if valid else float("nan"),
        "rows": rows,
    }


def print_comparison(res: dict, top: int = 25):
    print(f"shared tensors      : {res['shared_tensors']}")
    print(f"only in A (base)    : {len(res['only_in_a'])} {res['only_in_a'][:5]}")
    print(f"only in B (finetune): {len(res['only_in_b'])} {res['only_in_b'][:5]}")
    mism = [r for r in res["rows"] if "shape_mismatch" in r]
    print(f"shape mismatches    : {len(mism)}")
    print(f"param-weighted cosine: {res['param_weighted_cosine']:.6f}")
    print(f"mean relative delta : {res['mean_rel_delta']:.6f}")

    valid = [r for r in res["rows"] if "cosine" in r]
    print("\n-- most changed tensors (highest rel_delta) --")
    for r in sorted(valid, key=lambda r: -r["rel_delta"])[:top]:
        print(f"  {r['rel_delta']:.5f}  cos={r['cosine']:+.5f}  {r['name']}")

    print("\n-- least changed tensors --")
    for r in sorted(valid, key=lambda r: r["rel_delta"])[:10]:
        print(f"  {r['rel_delta']:.5f}  cos={r['cosine']:+.5f}  {r['name']}")

    # group by module type
    groups = defaultdict(list)
    for r in valid:
        key = r["name"].split(".", 1)[0]
        for tag in ("layers", "embed", "norm", "lm_head"):
            if tag in r["name"]:
                key = tag
                break
        groups[key].append(r["rel_delta"])
    print("\n-- rel_delta by module group --")
    for k, v in sorted(groups.items(), key=lambda kv: -np.mean(kv[1])):
        print(f"  {k:<14} mean={np.mean(v):.5f}  n={len(v)}")


# --------------------------------------------------------------------------
# GGUF quantization signature
# --------------------------------------------------------------------------

GGML_TYPES = {
    0: "F32", 1: "F16", 2: "Q4_0", 3: "Q4_1", 6: "Q5_0", 7: "Q5_1",
    8: "Q8_0", 9: "Q8_1", 10: "Q2_K", 11: "Q3_K", 12: "Q4_K", 13: "Q5_K",
    14: "Q6_K", 15: "Q8_K", 16: "IQ2_XXS", 17: "IQ2_XS", 18: "IQ3_XXS",
    19: "IQ1_S", 20: "IQ4_NL", 21: "IQ3_S", 22: "IQ2_S", 23: "IQ4_XS",
    24: "I8", 25: "I16", 26: "I32", 27: "I64", 28: "F64", 29: "IQ1_M",
    30: "BF16",
}
GGUF_SCALAR = {0: "B", 1: "b", 2: "H", 3: "h", 4: "I", 5: "i", 6: "f",
               7: "?", 10: "Q", 11: "q", 12: "d"}


def read_gguf_header(path: str) -> dict:
    with open(path, "rb") as f:
        magic = f.read(4)
        if magic != b"GGUF":
            raise ValueError(f"{path}: not a GGUF file (magic={magic!r})")
        version = struct.unpack("<I", f.read(4))[0]
        n_tensors = struct.unpack("<Q", f.read(8))[0]
        n_kv = struct.unpack("<Q", f.read(8))[0]

        def rd_str():
            (ln,) = struct.unpack("<Q", f.read(8))
            return f.read(ln).decode("utf-8", "replace")

        def rd_val(t):
            if t == 8:
                return rd_str()
            if t == 9:  # array
                (et,) = struct.unpack("<I", f.read(4))
                (cnt,) = struct.unpack("<Q", f.read(8))
                return [rd_val(et) for _ in range(cnt)]
            fmt = GGUF_SCALAR[t]
            size = struct.calcsize("<" + fmt)
            return struct.unpack("<" + fmt, f.read(size))[0]

        kv = {}
        for _ in range(n_kv):
            key = rd_str()
            (vt,) = struct.unpack("<I", f.read(4))
            try:
                kv[key] = rd_val(vt)
            except Exception:
                kv[key] = f"<unparsed type {vt}>"
                break

        tensors = []
        for _ in range(n_tensors):
            name = rd_str()
            n_dims = struct.unpack("<I", f.read(4))[0]
            dims = [struct.unpack("<Q", f.read(8))[0] for _ in range(n_dims)]
            (tt,) = struct.unpack("<I", f.read(4))
            (off,) = struct.unpack("<Q", f.read(8))
            tensors.append({"name": name, "dims": dims,
                            "type": GGML_TYPES.get(tt, str(tt)), "offset": off})

    return {"version": version, "n_tensors": n_tensors, "kv": kv, "tensors": tensors}


def print_gguf(path: str):
    h = read_gguf_header(path)
    print(f"GGUF version {h['version']} | {h['n_tensors']} tensors")
    print("\n-- metadata --")
    for k, v in h["kv"].items():
        s = str(v)
        if len(s) > 70:
            s = s[:67] + "..."
        print(f"  {k} = {s}")

    hist = defaultdict(int)
    hist_bytes = defaultdict(int)
    for t in h["tensors"]:
        hist[t["type"]] += 1
        hist_bytes[t["type"]] += int(np.prod(t["dims"]))
    print("\n-- quant type histogram --")
    for k, v in sorted(hist.items(), key=lambda kv: -kv[1]):
        print(f"  {k:<10} {v:4d} tensors  {human(hist_bytes[k])} weights")

    print("\n-- tensor layout (first 12) --")
    for t in h["tensors"][:12]:
        print(f"  {t['type']:<8} {str(t['dims']):<22} {t['name']}")


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def fingerprint(model_dir: str):
    m = Model(model_dir)
    print(f"=== {m.dir} ===")
    print(f"shards: {len(m.shards)} | tensors: {len(m)} | params: {human(param_count(m))}")
    if m.shards[0].metadata:
        print(f"safetensors metadata: {m.shards[0].metadata}")

    print("\n-- architecture --")
    for k, v in arch_fingerprint(m).items():
        print(f"  {k:<22} {v}")

    print("\n-- tokenizer --")
    for k, v in tokenizer_fingerprint(m.dir).items():
        print(f"  {k:<16} sha256={v['sha256']} bytes={v['bytes']}")

    print("\n-- dtype mix --")
    dtypes = defaultdict(int)
    for n in m.names():
        dtypes[m.index[n].info(n)["dtype"]] += int(np.prod(m.index[n].info(n)["shape"]))
    for k, v in sorted(dtypes.items(), key=lambda kv: -kv[1]):
        print(f"  {k:<6} {human(v)} weights")

    print("\n-- per-tensor stats (sample) --")
    st = per_layer_stats(m, sample=24)
    for n, s in list(st.items())[:24]:
        print(f"  {n:<46} rms={s['rms']:.5f} absmax={s['absmax']:.3f} "
              f"outlier={s['outlier_ratio']:.1f} kurt={s['kurtosis']:+.2f}")


def main():
    # Windows consoles default to cp1252; tokenizer dumps contain raw UTF-8 bytes
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser()
    ap.add_argument("model_dir", nargs="?")
    ap.add_argument("--compare", nargs=2, metavar=("BASE", "FINETUNE"))
    ap.add_argument("--gguf")
    ap.add_argument("--json", action="store_true", help="emit machine-readable JSON")
    a = ap.parse_args()

    if a.gguf:
        print_gguf(a.gguf)
        return
    if a.compare:
        res = compare(*a.compare)
        if a.json:
            print(json.dumps(res, indent=2))
        else:
            print_comparison(res)
        return
    if a.model_dir:
        fingerprint(a.model_dir)
        return
    ap.print_help()


if __name__ == "__main__":
    main()
