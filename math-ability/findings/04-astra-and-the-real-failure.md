# Finding 04: what Astra actually did, and what this model actually does

The Astra post is the reason to re-open the question, so this starts with what
it claims, then measures the same thing here.

Raw measurements: [`e14_loop_probe.json`](e14_loop_probe.json),
[`e15_continuation.json`](e15_continuation.json),
[`e16_stop_edit.json`](e16_stop_edit.json),
[`e17_word_problems.json`](e17_word_problems.json).

## What Astra did

Experiment 8 of "Sparks of Weight-Space Programming" (Takehi, Teoh, Zhang,
Khattab, November 2026).

The subject is Qwen3.5-0.8B. Its failure was not wrong arithmetic. It was
looping: 85 of its first 96 practice answers ran to the 32,768 token limit, and
93 percent of its MATH-500 answers hit that limit. A model that never finishes
scores near zero on a benchmark that needs a final answer, no matter how well
it computes.

Astra traced the loop to a direction written by the MLPs of layers 15 and 19,
subtracted that direction from 64 output columns in each layer, and changed
131,072 of 800 million weights, 0.016 percent. Result:

| Qwen3.5-0.8B | base | edited |
|---|---:|---:|
| MATH-500 | 7.9 | 48.1 |
| GSM8K | 13.8 | 67.1 |
| IFEval | 21.0 | 62.3 |
| answers hitting the token limit | 93.0 | 33.1 |

That is a real and large result. It is also, by the authors' own account, a
removal and not an addition. The control experiment says it plainly: they ran
the same procedure on Gemma 3 270M, which rarely loops (8.5 percent hit the
limit), and it did not improve.

| Gemma 3 270M | base | 5 passes | 10 | 25 | 50 |
|---|---:|---:|---:|---:|---:|
| MATH-500 | 6.8 | 7.0 | 7.6 | 7.7 | 7.0 |

Their conclusion, verbatim: "Editing weights could remove an obstacle that hid
an ability, but it could not add an ability the model lacked."

That is the same conclusion as
[`02-carry-edits.md`](02-carry-edits.md), reached from the other direction. The
Astra headline is a loop-removal result on a model whose baseline was crippled
by the loop. Their no-loop control is my entire experiment.

So the honest reading of the post is not "weight editing makes a model better
at maths". It is "weight editing can remove a degenerate behaviour, and if that
behaviour was capping the score, the score jumps".

## Does this model loop?

No. 20 prompts, 512 tokens, greedy, no prefill.

```
repetitive replies (loop score > 0.25):  0/20
median reply length:                     2 to 4 characters
longest reply:                           736 characters
first integer correct:                   0/20
```

Ten of the twenty replies were two to four characters: a single number and
EOS. The model does not loop, it stops. There is no loop direction to subtract,
so the Astra edit has no target here.

## What it does instead

It answers before it thinks. On 60 generated two-step word problems with exact
answers, greedy, 256 tokens:

| condition | exact | 95% CI | any correct integer |
|---|---:|---|---:|
| prefill `The answer is` | 8/60 = 0.13 | [0.07, 0.24] | 8/60 |
| no prefill | 14/60 = 0.23 | [0.14, 0.35] | 14/60 |
| prefill `Let me work through this step by step.` | 47/60 = 0.78 | [0.66, 0.87] | 56/60 |

One prefilled sentence, no weight change, moves exact accuracy from 0.23 to
0.78. The intervals do not overlap.

The `answer` prefill, which is the format used for every arithmetic number in
this folder, is the worst of the three. It forces an immediate answer and
suppresses the reasoning that would have produced one. That is a real
methodological lesson: the failure map in
[`01-failure-map.md`](01-failure-map.md) measures the model with its reasoning
switched off.

When it does reason, the arithmetic inside the reasoning is often correct. From
the hand-written set in [`e15_continuation.json`](e15_continuation.json):

```
There are 23 boxes with 14 apples each. 37 apples are rotten.

First, we need to find the total number of apples in all the boxes:
  23 boxes x 14 apples/box = 322 apples
Next, we subtract the number of rotten apples:
  322 apples - 37 rotten apples = 285 good apples
```

`23 x 14` is a 2x2 product. Asked directly with an answer prefill, this model
gets 2x2 multiplication 0.35 of the time. Inside chain of thought it computes
it correctly. The atom was never missing.

## Is the stop decision the lever?

If the failure is "stops too early", then the EOS row of `lm_head` is the
target. Scale that row and see. It is 896 numbers.

| EOS row scale | exact | any correct | median reply, math | median reply, non-math |
|---|---:|---:|---:|---:|
| 1.00 | 4/15 | 4/15 | 3 chars | 17 chars |
| 0.50 | 4/15 | 9/15 | 832 chars | 1088 chars |
| 0.25 | 4/15 | 9/15 | 832 chars | 1088 chars |
| 0.00 | 4/15 | 9/15 | 832 chars | 1088 chars |

Exact accuracy does not move at all. The model keeps talking and produces
`60.000000000000004 (rounded to an integer)`, and on a non-math prompt it
answers "Paris." and then emits timestamps in a loop. Suppressing EOS makes it
ramble, not reason.

So the stop is not a single bit that can be lowered. Forcing continuation and
prefilling chain of thought are not the same intervention:

| intervention | exact |
|---|---:|
| no prefill, EOS suppressed for 128 tokens | 8/15 |
| chain-of-thought prefill, normal stopping | 12/15 |

The prefill changes the trajectory. The EOS suppression only removes the exit.

## The margin, for the record

At the first assistant position, before any answer token, the best digit
outranks the best reasoning opener on 12 of 15 problems, with a mean margin of
1.15 nats (range -0.50 to +5.00). The top token is usually `'1'`, which is both
a digit and a list marker, so the model is genuinely torn between starting a
numbered list and blurting a digit.

That is a small, flippable margin, which is why the EOS experiment was worth
running. But a static bias on that position is not conditional on the prompt,
and the result above is what happens when you flip it globally: everything
becomes a list, including the prompts that should not.

## What this changes

**The negative result stands, with a better explanation.** Six edit families
did not move held-out arithmetic because the arithmetic was never the missing
piece. The model approximates multi-digit results, and it does not start
reasoning on word problems. Neither is a direction you can subtract.

**The positive result gets larger.** The scaffold in
[`03-scaffolds.md`](03-scaffolds.md) is one way to externalise composition. A
one-sentence chain-of-thought prefill is a cheaper way for word problems, and it
works because the same atoms are reliable. Both say the same thing: the model
has the parts, not the sequence.

**The Astra comparison is now exact.** Their model had a loop, and removing it
recovered an ability that was already there. This model has an early stop, and
removing it does not recover anything, because the stop is not the obstacle.
The obstacle is that it does not enter the reasoning trajectory at all, and that
is a different object.

## What this does not show

- 60 generated problems is a decent sample but they are template-generated. A
  real benchmark would be better. The effect is far outside the interval, so
  the direction is not in doubt; the exact size is.
- The chain-of-thought prefill is a harness change, not a weight edit. It is
  reported here as a measurement of the model, not as an edit.
- The EOS result rules out one specific hypothesis (that the stop is a logit
  threshold). It does not rule out every possible policy edit. A direction that
  is conditional on the prompt could in principle work, and no such direction
  was found.
