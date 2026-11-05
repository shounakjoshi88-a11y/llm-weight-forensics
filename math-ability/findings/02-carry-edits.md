# Finding 02: six ways to edit the weights for carry, all null

Every method here changes existing tensors. None adds a parameter. None uses
gradient descent. None is a LoRA or an adapter. This is the constraint the
whole folder was built under: change the content of the book, not paste a note
on the page.

The target was fixed before fitting: 100 held-out additions with at least two
carries, operands of 3 to 5 digits, exact final-integer match. The SPEC calls
for a paired bootstrap interval excluding zero and a sign test at p < 0.05.
Nothing below comes close.

## Summary

| # | edit | what it changes | best result | held-out | verdict |
|---|---|---|---:|---:|---|
| e03 | global gain | all of q, o or down in every layer | 0.633 vs 0.625 | same set | null |
| e04 | carry rank-one | one `down_proj`, `alpha * d v^T` | 0.625 vs 0.617 train | not reached | null |
| e05 | layer gain search | one scalar per attention or MLP block | 0.650 train | 0.661 vs 0.679 | overfit |
| e06 | carry neurons | top-k `down_proj` rows, scaled | 0.91 vs 0.88 | same set | null |
| e07 | base blend | every tensor, `(1-a)W + aW_base` | 0.62 vs 0.60 | 0.82 vs 0.81 | null |
| e08 | ` is` row | 896 numbers in `embed_tokens` | 0.745 train | 0.732 vs 0.732 | null |
| e10 | digit bias | 10 rows of `lm_head`, rank-one | 0.84 train | 0.89 vs 0.90 | null |

With 100 problems, the standard error at p = 0.8 is about 0.04. Every "best"
in the table is inside one standard error of its baseline, and several are
inside a single problem.

## e03: global gain

Multiply every `q_proj`, `o_proj` or `down_proj` by a constant. This is the
crudest possible edit and it is included as the floor.

| config | total | add4x4 | add5x5 | mul2x2 |
|---|---:|---:|---:|---:|
| base | 0.625 | 0.80 | 0.55 | 0.60 |
| q all 1.3 | 0.333 | 0.10 | 0.00 | 0.50 |
| q all 0.7 | 0.050 | 0.10 | 0.00 | 0.05 |
| o all 1.3 | 0.375 | 0.40 | 0.25 | 0.40 |
| down all 1.3 | 0.358 | 0.70 | 0.40 | 0.65 |
| q L8 1.5 | 0.633 | 0.75 | 0.60 | 0.65 |

Scaling one matrix type across all 24 layers destroys the model. Scaling a
single layer at 1.5 is survivable and buys one problem out of 120. The network
is not sitting on a global arithmetic gain knob.

The failure mode is informative: q at 1.3 keeps subtraction at 0.80 and
collapses addition to 0.10. The same global change moves different operations
in opposite directions, so there is no single scale that helps arithmetic as a
class.

## e04: carry rank-one in the MLP

Build the carry direction `d` at the answer position, then write it into one
`down_proj` as `W <- W + alpha * d v^T`. This is the ROME-shaped edit applied to
a state rather than a fact.

| layer | shift | train | regression |
|---|---|---:|---:|
| base | | 0.617 | 0.95 |
| L8 | 0.2 | 0.625 | 0.95 |
| L8 | 0.4 | 0.625 | 0.95 |
| L12 | 0.1 | 0.617 | 0.95 |
| L16 | 0.2 | 0.617 | 0.925 |
| L20 | 0.4 | 0.617 | 0.95 |

The best cell is one problem. The regression set (2x1 multiplication) never
moves, which means the edit is inert rather than harmful: it is not changing
the computation, it is being washed out.

## e05: per-layer gain search

One scalar gain per attention block and per MLP block, chosen greedily on 60
train problems.

```
base            train 0.567  val 0.679
selected        train 0.650  val 0.661
gains           attn9 1.15, attn11 0.85, attn15 1.15, mlp0 1.15
```

Train goes up 8 points, held-out goes down 2, and 2x2 multiplication drops from
0.50 to 0.33. This is the cleanest overfit in the set, and it is exactly what
the SPEC predicted for a search against a small training objective. Four
scalars fitted on 60 problems do not generalise.

## e06: carry neurons

Rank every MLP neuron by the standardised difference in mean activation
between carry and no-carry problems, then scale the top k `down_proj` rows by
`1 + beta`. Four layers, three values of k, two values of beta.

The best cell is L8, k = 128, beta = 1.0, which takes addition from 0.88 to
0.91. That is 3 problems out of 100, and it was selected on the same 100
problems it is scored on. Multiplication is flat at 0.85 to 0.86 everywhere.
Scaling the top 128 of 4864 neurons in one layer is a 2.6 percent perturbation
of that layer's MLP output, and the result is consistent with noise.

The bottom-k and random-k controls were not needed: the effect is not there to
control for.

## e07: blending toward the base checkpoint

`Qwen2.5-0.5B` sits next to the instruct checkpoint. Scored the same way, on
the same held-out problems, it is not a general donor.

| cell | base | instruct | delta |
|---|---:|---:|---:|
| add 3x3 | 0.92 | 0.94 | -0.02 |
| add 4x4 | 0.84 | 0.76 | +0.08 |
| add 5x5 | 0.68 | 0.60 | +0.08 |
| mul 2x1 | 0.98 | 0.95 | +0.03 |
| mul 3x1 | 0.73 | 0.84 | -0.11 |
| mul 2x2 | 0.48 | 0.55 | -0.07 |
| mul 3x2 | 0.10 | 0.12 | -0.02 |
| mul 5x1 | 0.52 | 0.60 | -0.08 |

The base model is better at long addition and worse at multiplication. That
split is real and it is the only cross-checkpoint signal in the folder.

A full 10 percent blend looked promising on a 60-problem screen: 4x4 addition
0.81 to 0.85, 5x5 0.52 to 0.56, 2x2 0.43 to 0.45. A layer-wise sweep then
found 13 layers whose individual blends helped addition, and the selected blend
was tested on 100 fresh problems per cell.

| cell | base | full 0.1 | selected layers |
|---|---:|---:|---:|
| add 4x4 | 0.81 | 0.82 | 0.82 |
| add 5x5 | 0.60 | 0.62 | 0.59 |
| mul 2x2 | 0.53 | 0.53 | 0.51 |
| mul 3x1 | 0.848 | 0.848 | 0.838 |
| mul 5x1 | 0.64 | 0.63 | 0.61 |

The screen was noise. The layer selection made 5x5 worse than the full blend,
which is the signature of fitting layer-level deltas that do not exist. This is
the most useful null in the set because the donor checkpoint is genuinely
better at addition, and blending still does not transfer it.

## e08: the ` is` embedding row

The prompt ends on a single token, ` is` (id 374), immediately before the
answer. Its embedding row is 896 numbers with norm 0.389. A random perturbation
at that norm moved 3-digit addition from 0.94 to 0.97 and 2x2 from 0.37 to
0.40, while a perturbation at twice the norm destroyed the model (0.00). The
row is load-bearing, so it was worth a proper search.

40 random directions at the row's own norm, selected on 200 train problems,
tested on 194 held-out.

```
base train 0.725   held-out 0.732
best single train 0.745   held-out 0.732
top-5 average              held-out 0.758
random control             held-out 0.737
```

The best direction on train does nothing on held-out. The top-5 average is 2.6
points over base, but the random control is 0.5 points over base, so the
selectable signal is about 2 points on 194 problems. That is one standard
error. Not a result.

The related geometry check is worth recording: the ten digit embeddings are
already ordered by value, Spearman(value, PC1) = +0.903. There is nothing to
repair in the digit geometry, which removes one plausible edit from the list
before it was tried.

## e10: carry-conditioned digit bias in `lm_head`

This was the last idea and the most targeted. The errors are often one digit
off in a carry column. A static bias cannot condition on the input, but a
rank-one edit can:

```
W_lm <- W_lm + alpha * v d^T
```

where `v[digit j] = j - 4.5` and `d` is the carry direction in the final normed
residual. The logit of digit `j` then moves by `alpha * (j - 4.5) * (d . h)`,
so the model prefers higher digits when the carry feature is active. Only ten
rows of `lm_head` change.

| alpha | add train | add held-out | mul 2x2 | regression |
|---:|---:|---:|---:|---:|
| base | 0.84 | 0.90 | 0.45 | 0.90 |
| 0.01 | 0.82 | 0.88 | 0.48 | 0.93 |
| 0.05 | 0.84 | 0.89 | 0.48 | 0.93 |
| 0.1 | 0.66 | 0.83 | 0.43 | 0.87 |

At the useful scale it does nothing to addition. At 0.1 it damages everything.
The small multiplication and regression gains at 0.01 and 0.05 are 3 problems
each and appear at every alpha, so they are not a dose response.

## What the failures have in common

Every edit that is large enough to change behaviour changes it everywhere at
once, and the changes do not agree in sign across operations. Global q scaling
helps subtraction and destroys addition. Base blending helps addition and hurts
multiplication. The ` is` row at twice its norm destroys the model.

That is what an entangled representation looks like from the outside. The
carry feature is not a separable direction, a separable neuron set, or a
separable scale. It is distributed through the same weights that do everything
else, so a local edit trades one behaviour for another.

The diagnosis in [`01-failure-map.md`](01-failure-map.md) explains why the
target was wrong to begin with. The errors are magnitude approximations, not
dropped carries: 0 of 38 addition errors equal the no-carry value, 55 percent
are off by a power of ten, and 93 percent of multiplication errors are within
10 percent of the answer with none matching a partial product. There is no
carry flag to strengthen because the model is not computing a carry at all.

## What would actually be needed

Adding a composition the model does not perform is a capability change, not a
state change. The sycophancy result in
[`../findings/07-weight-editing.md`](../findings/07-weight-editing.md)
is the contrast: there, the ability was present and a direction was hiding it,
and a gradient-free edit recovered it. Here the ability is absent, and six
gradient-free edits did not install it.

Within the constraint of no gradients and no new parameters, the honest options
are:

1. Accept the atom-level result and move the composition outside the model.
   That is [`03-scaffolds.md`](03-scaffolds.md), and it works.
2. Relax the constraint and train. That is a different experiment and it is not
   what this folder set out to test.

## What this does not show

- It does not show that no weight edit could ever add this ability. It shows
  that global scalings, rank-one directions, neuron gains, linear checkpoint
  blends, a single embedding row search, and a ten-row output bias do not.
- The carry direction in e04 and e10 was built from 100 train problems with
  imperfect labels. A better estimate of `d` might behave differently, though
  e08's search over 40 random directions at the same norm found nothing, which
  argues against it.
- e06 selected on the evaluation set, which flatters it. Even so it is null.
- These are exact-match scores on greedy decoding. A sampling-based method
  could in principle find correct answers inside the distribution. That is a
  different claim and it is tested in [`03-scaffolds.md`](03-scaffolds.md).
