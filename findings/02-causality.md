# Finding 02: what actually matters, measured by ablation

E01 found structures by looking at attention patterns. That is correlation. This
removes them and measures the damage.

Raw numbers: [`e02_ablation.json`](e02_ablation.json), [`e03_sensitivity.json`](e03_sensitivity.json).

## Method

Ablation means zeroing, not scaling:

- a head is removed by clearing its 64 columns at the `o_proj` input, which is
  exactly that head's contribution to the residual stream
- a neuron is removed by clearing its column at the `down_proj` input

Every experiment runs a control: removing randomly chosen heads or neurons of the
same count. Without that control, "removal changed the output" proves nothing,
because removal always changes the output.

Two measurements:

- **perplexity** on a natural passage, the general "is it still working" number
- **induction margin**, the mean gap between the logit for the copied token and
  the best competing logit, over 64 positions of a repeated random sequence. A
  margin keeps moving after accuracy saturates, so it detects damage that argmax
  hides. On a shuffled sequence the same measurement scores -10.4, which is the
  control proving the number means something.

## Result 1: the hooks work

Before trusting any null result, prove the intervention does something:

| condition | perplexity | ratio |
|---|---|---|
| baseline | 28.79 | 1.00x |
| all heads in layer 9 | 29.62 | 1.03x |
| all heads in all layers | 6,444,082 | 223,861x |
| all MLP neurons in layer 9 | 30.45 | 1.06x |

Zeroing every head destroys the model. So the plumbing is fine, and the small
numbers elsewhere are real.

That table also says something on its own: removing all 14 heads of a single
layer costs 3% perplexity, and removing all 4864 MLP neurons of a layer costs 6%.
A 0.5B model is enormously redundant at the single-layer level.

## Result 2: massive activations are fixed biases, and they are load-bearing

E01 found layers 2, 3 and 21 with activations 250 to 350 times larger than
typical. Two questions: are they input-dependent, and do they matter.

**They are input-independent.** Across five completely different passages:

| layer | top channel | 5/5 inputs | peak values observed |
|---|---|---|---|
| 2 | 813 | yes | 1536, 1480, 1536, 1784, 1536 |
| 3 | 2247 | yes | 996, 1000, 996, 1008, 996 |
| 21 | 567 | yes | 1536, 1512, 1536, 1704, 1536 |

The same channel is the maximum every time, and the magnitude barely moves:
around 1536 regardless of what the text is about. That is not content. It is a
learned constant, a bias term written into a single neuron and then read back out
by the rest of the network.

**They are load-bearing, and super-additively so:**

| ablated | perplexity | ratio |
|---|---|---|
| layer 2 channels only | 34.87 | 1.31x |
| layer 3 channels only | 28.55 | 1.07x |
| layer 21 channels only | 28.14 | 1.06x |
| all three layers | **486.25** | **18.30x** |
| random neurons, same count | 26.87 | 1.01x |

Each layer alone is nearly harmless. All three together is catastrophic, an 18x
perplexity increase. The random-neuron control at 1.01x confirms this is specific
to those channels and not an artifact of zeroing 15 columns.

This matches the reported behaviour of massive activations in larger models
(Sun et al., 2024), where they act as explicit attention biases and anchor the
attention sink. Here the same structure exists in a 0.5B model, at three specific
layers, in three specific channels.

## Result 3: the induction heads are not load-bearing

E01 found three near-perfect induction heads. Removing them:

| ablated | induction margin | change |
|---|---|---|
| baseline | +5.508 | |
| L9H13 alone | +5.470 | -0.038 |
| L11H12 alone | +5.990 | +0.482 |
| L16H3 alone | +5.041 | -0.467 |
| all three | +5.019 | -0.489 |
| six random heads | -0.740 | **-6.248** |

Removing a head that looks like a perfect induction head changes the margin by
less than 0.5. Removing six arbitrary heads costs 6.2 and drops accuracy from
0.938 to 0.547.

**L11H12 scores 0.954 on the induction metric and removing it makes induction
slightly better.** Attention pattern and function are not the same thing.

## Result 4: the network compensates, non-monotonically

Ablating the top induction-critical heads in order:

| K heads removed | margin | change |
|---|---|---|
| 1 | -3.543 | -9.050 |
| 2 | +4.558 | -0.950 |
| 3 | +2.951 | -2.557 |
| 5 | -1.294 | -6.802 |
| 8 | -5.701 | -11.209 |
| 12 | -7.442 | -12.950 |

Removing one head costs 9.05. Removing two costs 0.95. Adding the second head
**recovers most of the damage the first one caused.**

That is not a clean circuit. In a clean circuit, damage accumulates. Here the
heads are partly redundant and partly competing, so the ordering of ablation
changes the result. Any claim of the form "head X implements behaviour Y" has to
survive this test, and most of them will not.

## What this does not show

- Perplexity on one passage is a narrow probe. Induction heads may matter for
  tasks this measurement does not reach.
- Zeroing is the bluntest intervention. A head could matter through a pathway
  that zeroing hides, because downstream layers can compensate when they see a
  zeroed signal but not when they see a corrupted one.
- The 18x collapse from removing massive activations is measured, not explained.
  Whether they act as biases, as sink anchors, or as both is still open.

## Reproduce

```bash
python experiments/e02_ablation.py
python experiments/e03_sensitivity.py
```
