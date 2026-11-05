# Weight-space forensics: local testbed

Can you look at an LLM's weights and tell what it is? This directory is a working
answer for small models, with the controls needed to keep the claim honest.

## What's here

Two layers of work: weight-level forensics (what the file is) and mechanistic
interpretability (what the network does).

```
analyze.py        fingerprint / lineage / GGUF tooling (no torch needed)
perm_test.py      permutation-invariance check on cosine similarity
fetch_models.py   download the checkpoints

probe/            instrumentation
  core.py         model loading, forward-pass capture, logit lens
  ablate.py       zeroing heads and neurons
  attention.py    induction, sink, previous-token and entropy scores
  corpus.py       public domain text download for neuron analysis

experiments/      one script per experiment, e01 to e12
findings/         one writeup per finding, raw JSON next to each
viz/              live neuron visualizer (FastAPI server + Three.js frontend)
math-ability/     the arithmetic editing program, e01 to e17 (own README)

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

## Live neuron visualizer

Watches the model think. The browser draws the network in 3D: 24 rings, one per
layer, 48 neurons on each ring, stacked into a tube you can orbit. Real
activations stream in as the model reads the prompt and then writes, and the
neurons that fire light up.

```bash
python viz/server.py
# open http://127.0.0.1:8000
```

- 48 of 4,864 MLP neurons per layer, chosen once from `down_proj` by how strongly
  each neuron writes into the residual stream. The set is fixed, so a neuron
  keeps its place on screen across tokens.
- Grey is idle, clay is a positive activation, blue-grey is negative, and both
  grow and brighten with magnitude. Edges connect the strongest neurons between
  adjacent layers, so the view reads as a network rather than a grid.
- The camera holds still. The opening shot is solved from the network's projected
  size, so the tube fills the stage at any window shape. Auto focus is off by
  default; switch it on for a slow pan toward the firing layer. Scroll to zoom,
  drag to orbit, "Reset view" to return home.
- Layer numbers sit under the tube and the hottest layer is marked in clay.

The prompt pass is paced at about 90 ms per token so the reading phase is
visible instead of a blur. The checkpoint is an instruct model, so the server
wraps the prompt in its chat template before the forward pass; without that it
continues the string instead of answering. Three.js, OrbitControls and GSAP are
vendored under `viz/static/vendor/`, so there is no build step and no network
access needed.

Note: this shows a fixed 48-neuron sample per layer, not the whole MLP. The
selection is weight-derived, not activation-derived, so it is not cherry-picked
per token.

## Inside the network

A separate line of work, documented in [`findings/`](findings/README.md). Five
results that survived controls:

- **Residual norm is not monotonic.** 90x growth across the first four layers,
  then a flat plateau, then a 4x spike at the final layer.
- **Massive activations are fixed biases.** Channels 813, 2247 and 567 are the
  maximum in 5 of 5 unrelated inputs, at about 1536 regardless of content.
  Removing all three costs 18x perplexity. Removing any one costs about 1.1x.
- **Attention pattern does not predict importance.** The three heads with the
  highest induction scores rank 200, 254 and 336 out of 336 when ablated.
- **Language and induction use disjoint heads.** Correlation between the two
  per-head importance maps is -0.016.
- **Classical weight-space readouts fail on this architecture.** K-composition
  from previous-token heads correlates -0.232 with measured importance, and an
  OV copying score correlates -0.022. GQA and RoPE move the circuit out of the
  matrices that the readout touches.

## Reading weights directly

The blog's opening experiment, at a scale where reading is possible. Five
networks of **129 parameters** each (`2 -> 32 -> 1`), trained on five different 2D
tasks, then dumped with the labels replaced by the slots A to E.

Identified blind, from the numbers, before the mapping was revealed:

| slot | identified as | true task |
|---|---|---|
| A | circle | circle |
| B | diagonal | diagonal |
| C | horizontal | horizontal |
| D | xor | xor |
| E | vertical | vertical |

5 of 5. Then the reading was turned into an explicit rule and repeated on 25
networks (5 tasks x 5 seeds) to check it was not hindsight:

| reader | set on 1 seed | held out, 4 seeds |
|---|---|---|
| obvious statistics (energy on x, direction clustering, bias size) | 3/5 | 11/20 |
| output-weighted direction, `sum_i w_out_i * w_i` | 5/5 | **20/20** |

The obvious statistics fail because the optimiser spreads a linear function over
units pointing in many directions; only the output-weighted sum recovers the
discriminant. The working reader recovers the function class, the discriminant
angle to within 1.3 degrees, and sometimes the circle radius as a number.

Full writeup: [`findings/06-weight-reading.md`](findings/06-weight-reading.md).

## Editing weights directly

The blog also edits a model with no gradients. Here the target is sycophancy:
Qwen2.5-0.5B-Instruct knows 76 simple facts, but agrees with a wrong answer when
the user states one.

The direction is `d = mean h(suggested) - mean h(neutral)` at the answer position
after layer 18. Two ways to remove it: project it out of the residual (the
oracle), or make every matrix above layer 18 blind to it (`W <- W (I - d d^T)` on
q, k, v, gate, up and `lm_head`). The second is a weight edit.

| condition | suggested margin | flips | neutral margin |
|---|---|---|---|
| base | -4.74 | 0% | 4.59 |
| activation steering | -1.44 | 31% | 4.49 |
| read-side weight edit | -2.66 | 18% | 4.53 |
| random direction | -4.69 | 0% | 4.57 |
| write-side weight edit | -2.73 | 8% | 4.17 |

51 held-out facts. The read-side edit flips about one fact in five and leaves the
neutral margin nearly untouched. The write-side edit cascades and costs knowledge.

The important caveat is what the direction is. A wrong suggestion and a correct
suggestion move the residual in almost the same direction (cosine 0.985). The
edit removes the user's assertion, not a judgement that the answer is false. It
also makes the model ignore correct corrections.

Full writeup: [`findings/07-weight-editing.md`](findings/07-weight-editing.md).

## Arithmetic: nine edits, one that moves held-out accuracy

A separate program in [`math-ability/`](math-ability/README.md), run under the same
rule: change existing tensors, no LoRA, no adapters, no gradients. The target was
`57892 x 512`.

**Where it breaks.** 1x1 to 2x2 addition is perfect, 4x4 and 5x5 are 0.60, 7x7 is
0.15. Multiplication collapses at the second factor digit: 2x1 is 1.00, 2x2 is
0.35, 3x2 is 0.05.

The errors are not dropped carries. 0 of 38 addition errors equal the no-carry
value, 55 percent are off by a power of ten, and 93 percent of multiplication
errors are within 10 percent of the answer with none matching a partial product.
The model is rounding an approximation, not running the wrong algorithm.

**Seven arithmetic edits, all null on held-out problems.**

| edit | what changes | held-out |
|---|---|---|
| global gain | all of q, o or down | 0.633 vs 0.625 |
| carry rank-one | one `down_proj` | null |
| layer gain search | one scalar per block | 0.661 vs 0.679, overfit |
| carry neurons | top-k `down_proj` rows | 0.91 vs 0.88, selected on eval |
| base blend | every tensor toward `Qwen2.5-0.5B` | 0.82 vs 0.81 |
| ` is` row | 896 numbers in `embed_tokens` | 0.732 vs 0.732 |
| digit bias | 10 rows of `lm_head` | 0.89 vs 0.90 |
| EOS row | 896 numbers in `lm_head` | 0.27 vs 0.27, rambles |

Every gap is inside one standard error at n = 100. Edits big enough to move
behaviour move it everywhere, in directions that disagree across operations.

**Why they fail, measured.** At each digit position, with the correct prefix fed
in, a closed-form probe from the residual matches the model's own accuracy almost
exactly: 0.97, 0.90, 0.90, 0.87, 0.89, 0.98 for the six digits of a 5x5 sum,
against a probe at 1.00, 0.88, 0.90, 0.85, 0.92, 0.98 and a shuffled control at
chance. There is no hidden correct digit for a better readout to find. Exact
match multiplies six per-digit rates, and six rates near 0.9 multiply to the 0.55
the model actually scores.

**The one arithmetic edit that works.** The readout is still a couple of points
worse than a least-squares fit to those same states. Replacing a quarter of the
ten digit rows of `lm_head` with that fit, 8,960 numbers solved in closed form,
raises held-out 5x5 addition from 0.60 to 0.65 (18 fixed, 3 broken, p = 0.001) and
costs 3x2 multiplication, 0.15 to 0.11. It holds on five-digit and six-digit sums
alike, so it is not the constant leading digit. It is two points per digit, which
is five points of exact answers: the size of the effect available at the output
layer, and the reason the larger edits failed.

**The `Step` row is not an arithmetic edit.** Boosting it by a start-of-reply
direction, 896 numbers, raises held-out word-problem accuracy from 0.23 to 0.59
(41 fixed, 5 broken, p = 4.4e-8) by making the model start reasoning instead of
blurting a number. It costs 5x5 addition, 0.55 to 0.05, and it is worse than the
`Step 1: ` prefill, which gets 0.95. It is a format prior, not a capability.

**What works without touching weights.** Two things.

A base-100 chunked scaffold that only asks for 2-digit by 1-digit products and
2-digit sums solves `57892 x 512 = 29640704` exactly. Across 30 held-out problems
per cell: 2x2 0.35 to 1.00, 3x2 0.05 to 1.00, 5x3 0.05 to 0.81.

A one-sentence chain-of-thought prefill on 60 generated word problems: 0.23 to
0.78 exact, and `Step 1: ` reaches 0.92. The `The answer is` prefill used for
every arithmetic number above scores 0.13, the worst of the three, because it
forces an answer before the model has reasoned.

**Why the large edits fail, and why this is the Astra control.** The Astra post reports
MATH-500 going 7.9 to 48.1 on Qwen3.5-0.8B by subtracting a loop direction from
two MLP layers. That baseline was crippled by a loop: 93 percent of its answers
hit the 32,768 token limit. Their own no-loop control, Gemma 3 270M, did not
improve, and they say so: editing weights can remove an obstacle that hid an
ability, but it cannot add one.

This model does not loop (0 of 20 prompts). It stops: median reply 2 to 4
characters. Suppressing EOS does not make it reason, it makes it ramble. So there
is no obstacle of that kind to remove.

Full writeups: [`math-ability/findings/`](math-ability/findings/).

## What this does and does not show

**Does:** architecture, tokenizer family, quantization recipe, and fine-tune
lineage are all recoverable from weights alone. Metadata is unnecessary. In a
small network, the task itself is recoverable too: the label shows up as a
low-dimensional geometric summary of all the weights, not as any single number.
A behavioural obstacle can also be partially removed without gradients: a
read-side weight edit flips 18% of held-out sycophancy facts while leaving the
neutral margin nearly unchanged.

**Doesn't:** recover training data, predict capabilities, or identify an unknown
model from scratch. The 5/5 reading result is at 129 parameters; the same
statistics return noise at 494M, where GQA, RoPE and superposition break the
clean geometry. Two models independently trained on similar data with similar
recipes are not separable by these statistics. Nothing in the weights encodes a
unique identity unless someone embedded one deliberately. The edit does not
distinguish a wrong assertion from a correct one, it is partial, and it cannot
add an ability the model does not have.

## Note on GGUF metadata

The Ollama blob self-declares its lineage:

```
general.base_model.0.repo_url = https://huggingface.co/Qwen/Qwen2.5-0.5B
```

That is a claim, not evidence, and trivially forgeable. The comparison in section
3 is what you would use when that field is missing or lies.
