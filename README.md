# Weight-space forensics: local testbed

Can you look at an LLM's weights and tell what it is? This directory is a working
answer for small models, with the controls needed to keep the claim honest.

## What's here

```
analyze.py        fingerprint / lineage / GGUF tooling (no torch needed)
perm_test.py      permutation-invariance check on cosine similarity
Qwen2.5-0.5B/          494M params, bf16, 942 MB   (base)
Qwen2.5-0.5B-Instruct/ 494M params, bf16, 942 MB   (SFT of the base)
SmolLM2-135M/          135M params, bf16, 270 MB   (negative control)
```

Model weights are not committed to this repo (GitHub rejects files over 100 MB).
`fetch_models.py` downloads them; they land in the paths above.

`qwen2.5:0.5b` is also already in Ollama, a Q4_K_M GGUF giving a third checkpoint
of the same model at a different precision.

Test hardware: i5-13420H / 15.6 GB RAM / RTX 4050 Laptop 6 GB.

## Running

```bash
python fetch_models.py                              # download the three checkpoints
python analyze.py Qwen2.5-0.5B-Instruct             # architecture + tokenizer + tensor stats
python analyze.py --compare A B                     # lineage / delta analysis
python analyze.py --gguf <blob>                     # quantization signature
python analyze.py --compare A B --json              # machine-readable
```

The safetensors reader is framework-free and streams tensor-by-tensor, so peak
memory is one tensor, not the whole model. bf16 is decoded by shifting bits into
float32 rather than going through torch.

## Results

### 1. Identification is easy

From `config.json` plus tensor names alone: `Qwen2ForCausalLM`, 24 layers, hidden
896, 14 heads and 2 KV heads (GQA), intermediate 4864, vocab 151936, tied
embeddings, rope_theta 1e6, 32k context. That is enough to name the family.

The tokenizer hash is the stronger fingerprint. `tokenizer.json` sha256
`c0382117ea329cdf...` identifies the Qwen2 BPE exactly, independent of weights.

### 2. Quantization leaves a readable signature

`qwen2.5:0.5b` is not uniform Q4:

| Quant | Tensors | Weights |
|---|---|---|
| Q5_0 | 132 | 251.9M |
| Q8_0 | 13 | 137.5M (embedding) |
| Q6_K | 12 | 52.3M |
| Q4_K | 12 | 52.3M (`ffn_down`) |
| F32 | 121 | norms and biases |

That is the Q4_K_M recipe: mixed precision chosen by layer sensitivity. The
allocation itself is a fingerprint of which tool and preset produced the file.

### 3. Lineage is recoverable, and this is the real result

Base vs Instruct, 290/290 tensors shared, zero shape mismatches:

```
param-weighted cosine : 0.996226
mean relative delta   : 0.044292
```

SFT moved the weights about 4.4% in relative norm. Where it moved them is
structured, not random:

- **Embedding moved most** (rel_delta 0.112), chat formatting and special tokens.
- **`q_proj` in late layers** (0.09 to 0.105), query/key geometry is what
  instruction following retunes.
- **`mlp.gate_proj` mid-layers** (0.088 to 0.092), the routing half of the MLP.
- **Biases barely moved** (0.0001 to 0.001). SFT is a low-rank update concentrated
  in matrices, not a shift.

So "is this a fine-tune of X, and what kind of fine-tune" is answerable from
tensors with no metadata at all.

### 4. Controls

**Negative control.** Qwen2.5-0.5B vs SmolLM2-135M: 218 shared tensor *names*,
**218/218 shape mismatches**. A different architecture does not even line up for
comparison. Discrimination is trivial at this level.

**Permutation test.** Cosine against a row-permuted copy of the identical tensor:

| tensor | true | permuted |
|---|---|---|
| `layers.10.self_attn.q_proj` | 1.0000 | 0.0030 |
| `layers.10.mlp.gate_proj` | 1.0000 | 0.0091 |
| `layers.10.mlp.down_proj` | 1.0000 | -0.0001 |

**This is the important caveat.** Raw cosine cannot distinguish "different model"
from "same model with reordered neurons", since both score about 0. Any real
provenance claim needs permutation-invariant matching (Git Re-Basin or weight
matching) before the cosine means anything.

## What this does and does not show

**Does:** architecture, tokenizer family, quantization recipe, and fine-tune
lineage are all recoverable from weights alone. Metadata is unnecessary.

**Doesn't:** recover training data, predict capabilities, or identify an unknown
model from scratch. Two models independently trained on similar data with similar
recipes are not separable by these statistics. Nothing in the weights encodes a
unique identity unless someone embedded one deliberately.

## Note on GGUF metadata

The Ollama blob self-declares its lineage:

```
general.base_model.0.repo_url = https://huggingface.co/Qwen/Qwen2.5-0.5B
```

That is a claim, not evidence, and trivially forgeable. The comparison in section
3 is what you would use when that field is missing or lies.
