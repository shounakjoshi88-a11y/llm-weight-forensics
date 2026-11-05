# Finding 07: gradients work, and the remaining errors are three different bugs

Findings 01 to 06 were constrained to editing existing weights, and every edit
either did nothing or bought five points by trading against another operation.
That constraint is gone. This is the first run with new parameters and gradient
descent: LoRA on all attention and MLP projections, trained on generated worked
solutions.

The answer to "did it get smarter" is yes, plainly, and by a lot on held-out
problems it never saw. It is not solved. The leftover errors are now specific
enough to name, and they are not the same bug in each operation.

Raw measurements: [`e31_finetune_eval.json`](e31_finetune_eval.json). Code:
[`../finetune/`](../finetune/).

## What was trained

9,147 generated problems, all verified arithmetically true before training. The
responses show the procedure rather than the answer: column carries for addition
and subtraction, shifted partial products for multiplication. 15 percent are
direct answers so the `The answer is` prefill still has something to condition
on.

The corpus is weighted toward the cells the failure map said were worst
(multiplication above 2x2). Only problems whose hash split is `train` are used,
so the held-out cells are unseen by construction. Checked explicitly: of the 560
held-out eval problems, zero appear in the training corpus.

LoRA r=32, alpha=64, on `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`,
`up_proj`, `down_proj`. 17.6M trainable parameters, 3.4 percent of the model.
Two epochs, 572 optimizer steps, effective batch 32, bf16, gradient
checkpointing. Final train loss 0.0105. Runtime 1,812 s on the 6 GB card.

## Held-out results

40 held-out problems per cell, natural prompting, no prefill. Base is the
shipped checkpoint, tuned is base plus the adapter.

| cell | base | tuned | delta |
|---|---:|---:|---:|
| add 1x1 | 1.00 | 0.97 | -0.03 |
| add 2x2 | 0.97 | 1.00 | +0.03 |
| add 3x3 | 0.80 | 1.00 | +0.20 |
| add 4x4 | 0.82 | 1.00 | +0.18 |
| add 5x5 | 0.57 | 0.97 | +0.40 |
| add 6x6 | 0.35 | 0.97 | +0.62 |
| sub 3x2 | 0.80 | 0.88 | +0.07 |
| sub 5x4 | 0.50 | 0.75 | +0.25 |
| mul 1x1 | 0.97 | 1.00 | +0.03 |
| mul 2x2 | 0.50 | 1.00 | +0.50 |
| mul 3x2 | 0.17 | 0.90 | +0.73 |
| mul 3x3 | 0.00 | 0.72 | +0.72 |
| mul 4x3 | 0.00 | 0.50 | +0.50 |
| mul 5x3 | 0.00 | 0.42 | +0.42 |

Word problems from [`e17_word_problems.py`](../experiments/e17_word_problems.py),
60 problems, natural prompting: base 0.17, tuned 0.90. The `Step 1: ` prefill
goes 0.92 to 0.88, so the gain is in the model producing the steps itself, not
in the prompt doing the work.

This is not memorisation. The eval problems are disjoint from training, the two
1x1 cells were barely or not at all in the corpus and stayed flat, and the
improvement grows with problem size instead of collapsing on unseen lengths.

One detail matters for the interpretation. The `The answer is` prefill skips the
written procedure, so it measures the model's one-shot arithmetic:

| cell | base prefill | tuned prefill |
|---|---:|---:|
| add 5x5 | 0.65 | 0.93 |
| add 6x6 | 0.45 | 0.88 |
| mul 3x3 | 0.00 | 0.00 |
| mul 5x3 | 0.00 | 0.00 |

Addition improved even without visible steps, so the weights themselves got
better at addition. Multiplication did not improve at all in one shot, which
says the multiplication gain lives entirely in the generated procedure. The
model learned to write the algorithm, not to skip it.

## The target problem

`57892 x 512`. Base gives 29046304. Tuned gives 30130624. True answer is
29640704, so it is still wrong, and it is wrong in an informative way:

```
Multiply 57892 by each digit of 512.
57892 x 2 = 115784
57892 x 1 = 57892, shifted 1 place = 578920
57892 x 5 = 289460, shifted 2 places = 28946000
Add the shifted products: 115784 + 578920 + 28946000 = 30130624
The answer is 30130624.
```

Every partial product is correct. The three shifted values are correct. The sum
of those three values is not: `115784 + 578920 + 28946000 = 29640704`, and the
model wrote 30130624. This is the 15-step pilot pattern surviving full training,
exactly as predicted: partial products self-consistent, final sum wrong.

## Where the remaining errors are

The stored replies are checked against the canonical worked solution for the
same problem. Three numbers per cell: exact match, whether every stated column
line is arithmetically true on its own, and whether the reply agrees with itself
(the answer matches the digits it wrote, or the borrow chain is valid).

| cell | exact | locally true | internal | sum right |
|---|---:|---:|---:|---:|
| add 5x5 | 0.97 | 1.00 | 0.95 | |
| add 6x6 | 0.97 | 1.00 | 0.97 | |
| sub 3x2 | 0.88 | 1.00 | 0.85 | |
| sub 5x4 | 0.75 | 1.00 | 0.65 | |
| mul 3x2 | 0.90 | 0.93 | | 0.97 |
| mul 3x3 | 0.72 | 0.97 | | 0.75 |
| mul 4x3 | 0.50 | 0.85 | | 0.57 |
| mul 5x3 | 0.42 | 0.82 | | 0.50 |

Three distinct failures, not one:

**Addition: transcription.** Every column is right, and then the final answer
disagrees with the digits just written. One failing 5x5 reply writes 0, 3, 4, 4,
1, 1 column by column, which is 114430, and then says `The answer is 114420`.
The arithmetic is done; the answer is copied out wrong. This is the readout
problem from [finding 06](06-readout-is-the-bottleneck.md) reappearing in a much
smaller form: 3 to 5 percent of replies, down from the per-digit error rate that
used to dominate.

**Subtraction: a dropped borrow.** Every line is locally true but the chain is
broken. One failing 5x4 reply takes a borrow at the thousands column and then
leaves the ten-thousands column at `6 - 0 = 6` instead of decrementing it to
`5 - 0 = 5`. The result is off by exactly one in that column, 66652 instead of
67652. Locally true, internally wrong. This is the subtraction analogue of a
carry not propagating, and it is why sub 5x4 sits at 0.75 while addition at the
same width is at 0.97.

**Multiplication: the final sum.** When every partial product is correct, the
answer is right 97 percent of the time at 3x2, 74 percent at 3x3, 59 percent at
4x3, 52 percent at 5x3. The partial products are not the bottleneck any more.
Adding three large shifted products in a single line is. The failing 5x3 example
gets all three partials right and then computes `169136 + 1134240 + 11342400` as
12645776 instead of 12646776, off by a thousand, and faithfully transcribes its
own wrong sum.

So the model learned the procedure, and the procedure now has a last step the
model cannot do reliably: adding several large numbers at once. That is the same
computation the scaffold in [finding 03](03-scaffold.md) offloads precisely
because the model cannot do it. The fine-tune moved the wall; it did not remove
it.

## The price

Fine-tuning on arithmetic costs a little outside arithmetic.

Perplexity on fixed English prose that is not in the corpus: base 19.76, tuned
21.28, a factor of 1.08. Not catastrophic forgetting, but not free.

Spot checks on non-arithmetic prompts show the same mild drift:

- "Difference between a list and a tuple in Python" now opens with `Let me work
  through it step by step.` The arithmetic format bleeds into non-arithmetic
  questions that look like they want an explanation.
- "What year did the Berlin Wall fall" was `1989`, now `August 13, 1989`. The
  year is right, the date is the one the wall was built.
- The sky-is-blue explanation got longer and less accurate.
- Most other prompts are unchanged or slightly more verbose. The haiku is
  identical.

So the trade is a large arithmetic gain for a small general-purpose loss and a
format prior that occasionally misfires.

## What this shows

New parameters and gradient descent do what ten hand-edited rows could not. The
held-out numbers are far outside any paired interval, and they survive the
disjointness check, so this is generalisation to unseen problems and not a
lookup table.

It also shows why the earlier weight-edit work kept hitting a wall. The base
model was not storing a nearly-correct answer behind a bad readout, as finding
06 showed. It was missing the procedure. No edit to ten digit rows can install a
procedure, and a 3.4 percent adapter can.

## What this does not show

- It is not solved arithmetic. Multi-digit multiplication tops out at 0.42 on
  5x3, and the failure is a real computation the model still cannot do.
- The corpus is template-generated. These are clean bare problems, not a
  benchmark, and not word problems with distractors.
- Two epochs is a light run. The loss collapsed to 0.01 by mid-epoch, so more of
  the same data will not help; the next gain has to come from a curriculum that
  drills the final sum and the borrow chain, not from more epochs.
- The collateral numbers are one prose sample and twelve prompts. They are
  directionally clear, not a measurement of general capability.
