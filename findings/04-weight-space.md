# Finding 04: classical weight-space readouts fail on this model

Motivated by "Sparks of Weight-Space Programming" (Takehi, Teoh, Zhang, Khattab,
2026), which reports an LLM naming induction heads from weights alone using the
composition method of Elhage et al. This tests the classical version of that
claim on a model where I already have ablation ground truth for all 336 heads.

Raw numbers: [`e08_weight_circuits.json`](e08_weight_circuits.json),
[`e09_ov_copying.json`](e09_ov_copying.json).

## Test 1: K-composition (Elhage et al.)

The structural definition of an induction head: a head whose key reads what a
previous-token head wrote. Formally, high K-composition from a previous-token
head, where composition is `||W_Q W_OV|| / (||W_Q|| ||W_OV||)`.

Sources were the four previous-token heads measured in E06: L8H7, L15H2, L10H2,
L1H12.

Result:

| comparison | correlation |
|---|---|
| composition vs activation-measured induction score | +0.126 |
| composition vs ablation induction damage | **-0.232** |
| composition vs ablation perplexity damage | -0.145 |

The heads E01 flagged as induction heads by their attention pattern:

| head | composition | activation score | ablation damage |
|---|---|---|---|
| L9H13 | 0.0337 | 0.981 | +0.038 |
| L11H12 | 0.0421 | 0.974 | -0.482 |
| L16H3 | 0.0406 | 0.989 | +0.467 |

Composition ranks them 259, 29 and 44 of 336. The correlation with actual
functional importance is negative.

### Why it fails here

**GQA breaks per-head attribution.** This model has 14 query heads and 2 KV
heads. Seven query heads share each key and value projection. The composition
score attributes a key-side effect to one query head when six others are reading
the same key. Elhage's framework assumes one key per head.

**RoPE hides position from the weights.** The entire notion of "previous token"
lives in the rotary rotation applied to Q and K at inference time. A readout that
only touches `W_Q` and `W_K` cannot see it. The circuit is real, the measurement
is blind to it.

**Layer 0 is excluded by construction.** L0H3 and L0H8 score exactly 0.0000
because a layer-0 head has no earlier layer to compose from. They are the two
most important heads for induction in the model, at +8.85 and +9.05 ablation
damage, and the method structurally cannot see them.

## Test 2: OV copying score (position-free)

The OV circuit has no positional dependence, so this should dodge the RoPE
problem. For each head, compute `M = W_E @ W_OV @ W_U^T` over a sample of 8192
tokens. A copying head should have a strong diagonal: writing token `a` makes
token `a` more likely.

Result: flat.

| head | copy rank | copy score | ablation damage |
|---|---|---|---|
| L9H13 | 190 | +0.000 | +0.038 |
| L11H12 | 154 | +0.000 | -0.482 |
| L16H3 | 245 | -0.000 | +0.467 |
| L0H3 | 197 | +0.000 | +8.852 |
| L8H7 | 247 | -0.000 | +4.092 |

Every head scores between -0.000 and +0.004. There is no discrimination at all.
Correlation with ablation damage is -0.022.

### Why it fails here

**RMSNorm normalizes away magnitude.** The OV diagonal measures a raw logit
boost, but the final norm rescales the residual stream before unembedding. A head
can write a large copy signal that the norm then flattens.

**The residual stream carries a massive constant bias.** Finding 02 showed fixed
activations around 1536 in three layers. Any small directional signal is swamped
by the constant offset that later layers read.

**Tied embeddings.** `W_U = W_E` here, so the copy test is asking whether `W_OV`
approximates the identity in embedding space. It does not, and it does not need
to: the model can copy through other routes.

## The honest boundary

This does not refute the blog. That work uses a frontier model *reasoning over*
the weights, not evaluating fixed matrix expressions. The distinction matters:

- **Classical circuit algebra** (what this experiment does) is a fixed formula.
  It has no way to notice that GQA invalidates its assumptions or that RoPE moves
  position out of the matrices. It returns confident numbers that are wrong.
- **An LLM reasoning over weights** can notice those things, which is presumably
  why the reported results are on reasoning rather than on matrix norms.

So the useful conclusion is narrower and more practical: **the standard
weight-only readouts, applied naively, do not recover function in a GQA + RoPE
model.** Any weight-space tool for this architecture needs to handle the KV
grouping and the rotary encoding explicitly, or it will report noise.

## What would actually work

1. **Fold RoPE into the readout.** Compute the QK circuit per relative offset by
   rotating the query and key matrices, rather than ignoring position.
2. **Respect GQA.** Attribute key-side effects to the KV group, not to a single
   query head.
3. **Use ablation as the ground truth.** Every weight-space readout should be
   scored against measured importance, which is exactly what this experiment does
   and what most weight-space work does not.

## Reproduce

```bash
python experiments/e08_weight_circuits.py
python experiments/e09_ov_copying.py
```
