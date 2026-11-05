# Finding 01: anatomy of Qwen2.5-0.5B-Instruct

First pass over the model. Everything here is measured, not assumed. Raw numbers
in [`e01_anatomy.json`](e01_anatomy.json).

## Setup

24 layers, hidden 896, 14 query heads, 2 KV heads (GQA), head dim 64,
intermediate 4864, 494M params.

Input: 48 tokens, made by concatenating two copies of 24 random tokens. The
repeat is deliberate. It is what makes induction heads visible: position `i` in
the second copy should attend to `i - 24 + 1`, the token that followed the
previous occurrence of the same token.

Model ran in bf16 on a 6 GB laptop GPU. Activations captured to CPU float32.

## Result 1: the residual stream does not grow smoothly

Norm of the residual stream, averaged over positions:

| layer | norm | growth |
|---|---|---|
| embed | 0.48 | |
| 0 | 7.11 | 14.8x |
| 1 | 8.89 | 1.25x |
| 2 | 25.29 | 2.85x |
| 3 | 43.82 | 1.73x |
| 4 to 15 | 44 to 54 | ~1.02x per layer |
| 16 to 20 | 59.9 to 81.0 | 1.04 to 1.10x |
| 21 | 65.64 | **0.81x** |
| 22 | 70.44 | 1.07x |
| 23 | 282.23 | **4.01x** |

Three things stand out.

1. **Almost all growth happens in the first four layers.** From 0.48 to 43.8, a
   90x increase. Layers 4 to 15 barely change it at all (about 2% per layer).
   The early layers are writing the bulk of the state, and the middle of the
   network is mostly refining it.
2. **Layer 21 shrinks the stream**, the only layer that does. A drop of 19% is
   too large to be rounding.
3. **Layer 23 multiplies it by 4.** The last layer before unembedding does not
   refine, it amplifies. This is the layer where the model commits to a token,
   and it does so by scaling up.

If you assume "deeper means more processed" you would predict a monotonic curve.
The model does not do that.

## Result 2: three layers are dominated by massive activations

Max absolute MLP activation per layer:

| layer | max abs activation | fraction near zero |
|---|---|---|
| 2 | **1536.0** | 1.000 |
| 3 | **996.0** | 1.000 |
| 21 | **1552.0** | 1.000 |
| 5 | 51.75 | 0.995 |
| 23 | 59.75 | 0.903 |
| 20 | 27.0 | 0.836 |
| 18 | 19.4 | 0.864 |
| everything else | 2.9 to 11.1 | 0.40 to 0.72 |

A typical layer peaks around 4 to 6. Layers 2, 3 and 21 peak at roughly 1000 to
1550, which is **250 to 350 times larger**. The "fraction near zero" column is
the tell: in those layers 99.9% of activations sit below 1% of the maximum, so a
single neuron is carrying more magnitude than the entire rest of the layer.

This is the massive activation phenomenon reported for larger models (Sun et al.,
2024). It is present here in a 0.5B model, at specific layers, not spread
uniformly. It is not a bug in my measurement: the same three layers also showed
the largest residual norm jumps in Result 1.

## Result 3: three clean induction heads

Induction score is the mean attention paid, over the second copy of the sequence,
to the token that followed the previous occurrence of the current token. 1.0
means perfect prefix matching.

| layer | head | induction score |
|---|---|---|
| 16 | 3 | **0.989** |
| 9 | 13 | **0.981** |
| 11 | 12 | **0.974** |
| 23 | 2 | 0.617 |
| 19 | 6 | 0.549 |
| 13 | 9 | 0.436 |

Three heads are essentially perfect. They sit in the middle of the network
(layers 9 to 16), which is exactly where the induction circuit is expected to
form: early layers build the previous-token signal, middle layers match prefixes,
later layers use the match.

The previous-token heads that feed them are also visible:

| layer | head | previous-token score |
|---|---|---|
| 8 | 11 | 0.906 |
| 15 | 6 | 0.786 |
| 1 | 12 | 0.758 |

That pairing, a previous-token head feeding an induction head, is the two-stage
circuit described by Olsson et al. (2022). Here it is in a 0.5B model.

## Result 4: two heads are perfect attention sinks

Attention mass on position 0:

| layer | head | sink mass |
|---|---|---|
| 9 | 7 | **1.000** |
| 11 | 7 | **1.000** |
| 20 | 11 | 0.957 |
| 19 | 4 | 0.944 |
| 3 | 13 | 0.919 |

Heads 7 in layers 9 and 11 attend to the first token and nothing else, always.
They are not reading content. They are dumping attention somewhere harmless so
that the softmax has somewhere to put its mass when a head has nothing to do.
The first token acts as a no-op register.

Note that these sit in the same layers as the induction heads. Layer 9 has both a
perfect sink (head 7) and a near-perfect induction head (head 13). The layer is
doing two unrelated jobs in parallel, in different heads.

## Result 5: MLP sparsity is about 50%, except where one neuron dominates

Outside the three massive-activation layers, roughly 40 to 72% of MLP
activations sit below 1% of the layer maximum. So the MLP is not sparse in the
"1% of neurons fire" sense that some architecture claims suggest. It is closer to
half the neurons mattering at any given position.

## Method validity note: the logit lens fails on this input

The logit lens applies the final norm and unembedding to each layer's residual
stream, asking "what would the model emit here". On random tokens it produces
garbage at every layer: Cyrillic characters, Chinese fragments, code identifiers
like `_modules` and `.getBounds`.

That is not a defect in the lens. Random tokens are out of distribution, and the
model has no sensible next-token prediction to make. The lesson is that a
readout tool must be validated on inputs the model was trained for, or it reports
noise confidently. The next experiment uses natural text for lens work.

## What this does not show

- Nothing here proves the induction heads are *used*. Correlation of attention
  pattern with a score is not causation. That requires ablation, which is
  experiment 02.
- The massive activations are identified, not explained. Whether they act as
  biases, as attention-sink anchors, or as something else is open.
- One input, one random seed. Head rankings may shift across inputs.

## Reproduce

```bash
python experiments/e01_anatomy.py
```
