# Finding 05: the one edit that works is a format prior, and it has a price

Seven edits aimed at arithmetic did nothing. This one moves held-out accuracy,
and it does it by changing the model's default format rather than its
arithmetic.

Raw measurements: [`e18_math_mode.json`](e18_math_mode.json),
[`e19_start_direction.json`](e19_start_direction.json),
[`e20_openers.json`](e20_openers.json),
[`e21_step_row.json`](e21_step_row.json),
[`e23_step_small_alpha.json`](e23_step_small_alpha.json),
[`e24_confirm.json`](e24_confirm.json),
[`e25_edit_raw_arithmetic.json`](e25_edit_raw_arithmetic.json).

## The result

100 held-out word problems, seed `20261122`, greedy, 384 tokens.

| condition | exact | 95% CI |
|---|---:|---|
| base | 23/100 = 0.23 | [0.16, 0.32] |
| one-row edit | 59/100 = 0.59 | [0.49, 0.68] |
| `Step 1: ` prefill | 95/100 = 0.95 | [0.89, 0.98] |

Paired on the same problems: 41 fixed, 5 broken, exact McNemar p = 4.4e-8.

The edit changes 896 numbers: one row of `lm_head`, the row for the token
`Step`. Nothing else in the model moves.

## How it is built

1. At the position that picks the first reply token, capture the final normed
   residual `h_start` from the bare prompt.
2. Capture `h_mid` from the same prompt with `Step` appended.
3. Take `d = mean(h_start) - mean(h_mid)`, normalised. This direction is large
   at the start of a reply and near zero inside one, so a boost along it does
   not keep firing after the first token.
4. Add `alpha * d` to the `Step` row of `lm_head`, with `alpha = 0.10`.

No gradients. No new parameters. The embedding table is separated from `lm_head`
first, so the input side is untouched.

The alpha matters more than anything else. At 1.0 and above the model starts
with `Step` and then repeats it, `Step Step Step`, because the boost still wins
at later positions. At 0.05 it wins only 15 percent of first tokens. The window
between those is narrow:

| alpha | exact (60 problems) |
|---:|---:|
| 0.00 | 0.17 |
| 0.05 | 0.37 |
| 0.10 | 0.60 |
| 0.20 | 0.53 |

## The price

The same edit, on raw arithmetic, 20 problems per cell, 384 tokens.

| cell | base | edit |
|---|---:|---:|
| add 5x5 | 0.55 | 0.05 |
| mul 3x2 | 0.15 | 0.00 |
| mul 5x3 | 0.00 | 0.00 |

The edit makes the model write `Step 1:` for arithmetic prompts too. Step by
step was already the worst format for raw arithmetic in
[`01-failure-map.md`](01-failure-map.md), and forcing it costs half the 5x5
addition accuracy.

So this is a trade, not a free gain: +0.36 on word problems, -0.50 on 5x5
addition.

## It is a prompt, not a capability

The prefill gets 0.95 and the edit gets 0.59. Both work by making the model
start with `Step`, and the prefill does it better because it puts the words in
the context, where every later layer can read them. The edit only tilts one
logit at one position, and the reasoning that follows is visibly worse:

```
Step-by-step reasoning:

Step 1: Calculate the total number of apples.
Step 2: Subtract the number of rotten apples from the total.

Step 1:
Step 2:
Step 3:
Step 4:
```

That is the model doing the shape of a solution and losing the content, which
is the same failure the format comparison found in the first place.

The regression is also real. The edit has no notion of task, so some non-math
prompts get step-formatted:

```
What does HTTP stand for?
  -> Step 1: Identify the term "Step 1" in this context.
     Step 2: Recognize that "Step 1" refers ...
```

Factual prompts that have a one-word answer survive (`Paris`, `Red, Yellow,
Blue`, `こんにちは`), because the first token is already confident and the boost
does not reach it.

## What this says about weight editing

It is the same lesson as Astra, from the other side.

Astra removed a direction to stop a loop, and the score jumped because the
baseline was capped by the loop. Here, adding a direction to start reasoning
raises word-problem accuracy because the baseline was capped by the model
answering immediately. Both are policy edits. Neither adds arithmetic the model
cannot do.

And in both cases the policy edit is bounded by what the model can already
compute. The edit gets 0.59, the prefill gets 0.95, and the gap is the
reasoning quality the model does not have. The edit can start the process. It
cannot make the process good.

The arithmetic edits in [`02-carry-edits.md`](02-carry-edits.md) failed for the
complementary reason: there was no policy to flip, only a computation the model
approximates.

## What this does not show

- It does not make the model better at `57892 x 512`. It makes it worse at raw
  arithmetic.
- It is not better than prompting. It is worse than prompting, and it is
  reported because it is a weight edit that moves a held-out number, which
  nothing else in this folder did.
- The word-problem set is template-generated. The effect is far outside its
  interval, but the exact size would need a real benchmark.
- One row and one alpha were selected on 60 problems and confirmed on 100. The
  selection surface is small, but it is not zero.
