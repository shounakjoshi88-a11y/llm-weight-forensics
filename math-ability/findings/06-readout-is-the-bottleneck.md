# Finding 06: the residual holds what the model reads out, and the readout is slightly wrong

Every edit in [finding 02](02-carry-edits.md) assumed the answer was somewhere
in the residual and the readout was failing to find it. That was never tested.
It is now, and the answer is sharper than either a win or a null: the residual
holds almost exactly what the model reads out, and the readout is a couple of
points worse than a closed-form fit to the same states.

That is enough to move raw arithmetic for the first time in this folder.

Raw measurements: [`e26_digit_trajectory.json`](e26_digit_trajectory.json),
[`e27_readout_refit.json`](e27_readout_refit.json),
[`e28_confirm_readout.json`](e28_confirm_readout.json),
[`e29_unrestricted.json`](e29_unrestricted.json),
[`e30_pooled_readout.json`](e30_pooled_readout.json).

## Is the answer in the residual?

Feed the model the prompt plus the correct answer, so it never makes an early
mistake, and at each digit position ask two questions: does the model's own
argmax give the right digit, and is the right digit linearly decodable from the
normed residual at that position (closed-form ridge, fit on a training split,
scored on held-out)?

If the probe is well above the model, the information is there and the readout
is the bug. If the probe matches the model, there is nothing to recover.

| 5x5 addition digit | 1 | 2 | 3 | 4 | 5 | 6 |
|---|---:|---:|---:|---:|---:|---:|
| model | 0.97 | 0.90 | 0.90 | 0.87 | 0.89 | 0.98 |
| probe | 1.00 | 0.88 | 0.90 | 0.85 | 0.92 | 0.98 |
| shuffled probe | 1.00 | 0.12 | 0.09 | 0.08 | 0.10 | 0.09 |

| 3x2 multiplication digit | 1 | 2 | 3 | 4 | 5 |
|---|---:|---:|---:|---:|---:|
| model | 0.94 | 0.66 | 0.26 | 0.43 | 0.96 |
| probe | 0.92 | 0.61 | 0.17 | 0.28 | 1.00 |
| shuffled probe | 0.21 | 0.09 | 0.09 | 0.11 | 0.08 |

The probe tracks the model, and the shuffled control sits at chance. There is no
hidden correct answer waiting for a better readout.

The exact-answer rate follows from the per-digit rates almost arithmetically.
For addition, `0.97 * 0.90 * 0.90 * 0.87 * 0.89 * 0.98 = 0.59`, against a
measured 0.55. The model is not drifting after an early error; it is about 10
percent wrong on every digit, independently, and the product of six of those is
what exact match measures.

For multiplication the bottleneck is one digit. The third digit is right 26
percent of the time, which no readout can repair.

## The readout is a little worse than it could be

Pool the residuals across all digit positions, fit one closed-form linear map to
the correct next digit, and compare it against `lm_head` on the same states:

| 5x5 addition digit | 1 | 2 | 3 | 4 | 5 | 6 |
|---|---:|---:|---:|---:|---:|---:|
| `lm_head` | 0.97 | 0.87 | 0.90 | 0.85 | 0.89 | 0.98 |
| closed-form fit | 1.00 | 0.89 | 0.92 | 0.87 | 0.92 | 0.99 |

Every digit is between one and three points better. The gain comes from data:
pooling six positions gives six times as many examples for one shared readout,
which is better regularised than the shipped rows.

## Installing it

Replace the ten digit rows of `lm_head` with a quarter of the correction:

```
W[d] <- W[d] + 0.25 * (W_fit[d] - W[d])     for the ten digit rows
```

That is 8,960 numbers, solved in closed form, no gradients and no new
parameters. `lm_head` is separated from the embedding table first so the input
side does not move.

5x5 addition, unrestricted, 300 held-out problems from a seed the fit never saw:

| condition | exact | fixed | broken | p |
|---|---:|---:|---:|---:|
| base | 180/300 = 0.60 | | | |
| correction | 195/300 = 0.65 | 18 | 3 | 0.001 |
| shuffled labels | 166/300 = 0.55 | 8 | 22 | 0.016 |

Split by answer length, to check the gain is not the constant leading digit of
a six-digit sum: five-digit sums 81 to 88, six-digit sums 99 to 107. It holds on
both, so it is a real readout improvement and not an artifact of the restriction.

A first confirmation on 200 fresh problems restricted to six-digit sums gave
116 to 128, `+14 -2`, p = 0.004, with the shuffled control at the same strength
scoring 99, `+7 -24`, p = 0.003.

## The price

The ten digit rows are shared, and the fit was made for one operation.

3x2 multiplication, 300 held-out problems:

| condition | exact | fixed | broken | p |
|---|---:|---:|---:|---:|
| base | 45/300 = 0.15 | | | |
| correction | 34/300 = 0.11 | 4 | 15 | 0.019 |

Fitting on both operations pooled does not remove the trade. At the strength
that helps addition it still costs multiplication:

| fit on | 5x5 addition | 3x2 multiplication |
|---|---:|---:|
| base | 0.60 | 0.15 |
| addition only, 0.25 | 0.64 | 0.04 |
| pooled, 0.30 | 0.65 | 0.11 |

So the edit buys about five points of exact addition and pays about four points
of exact multiplication. It is a better calibration of one shared readout, and a
single readout cannot be right for two operations that fail in different places.

## What this means

This is the first weight edit in this folder that moves raw arithmetic accuracy
on held-out problems. It is also small, and it is worth being precise about why.

It does not add arithmetic the model cannot do. It raises each digit's accuracy
by two points, and exact match multiplies six digits together, so two points per
digit becomes five points of exact answers. The same arithmetic explains why the
larger edits failed: a change big enough to move a digit by ten points moves the
other digits too, and moves the stop token, and the reply stops parsing.

The Astra contrast from [finding 04](04-astra-and-the-real-failure.md) still
holds. Astra removed an obstacle and the score jumped because the baseline was
capped. Here the baseline is capped by a per-digit error rate, and a readout can
move that by two points, not by the factor of ten that exact multiplication
needs.

## What this does not show

- It does not generalise across operations. Fitting on addition costs
  multiplication, and pooling does not fix it.
- The gain is five points on a template-generated problem set. It is well
  outside the paired interval, but it is not a benchmark number.
- Beta was chosen from a small sweep. The direction is monotone and the gain
  reproduces at three independent fits (0.125, 0.25, 0.375 all beat base), but
  the exact best strength is not pinned down.
- It says nothing about whether a readout fit on a mixture with learned weights
  per task would work. Only that one shared closed-form readout trades.
