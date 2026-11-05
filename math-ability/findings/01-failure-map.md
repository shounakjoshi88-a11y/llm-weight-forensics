# Finding 01: where the arithmetic actually breaks

Model: Qwen2.5-0.5B-Instruct, greedy decoding, exact final-integer scoring.
Raw measurements: [`e01_failure_map.json`](e01_failure_map.json) and
[`e02_diagnosis`](../experiments/e02_diagnosis.py), which reads the same file
offline and makes no model calls.

All problems are generated with seed `20261105` and hashed into train and
held-out before anything is measured. The grid below uses 20 problems per cell,
so each cell has a standard error near 0.10. The 100-problem cells used for
fitting later are tighter.

## The map

Prompt format is the `answer` prefill: the assistant turn opens with
`The answer is`, so the first answer digit is one decision.

| operation | cell | exact | notes |
|---|---|---:|---|
| add | 1x1 | 1.00 | |
| add | 2x1 | 1.00 | |
| add | 2x2 | 1.00 | |
| add | 3x2 | 0.85 | |
| add | 3x3 | 0.90 | |
| add | 4x4 | 0.60 | |
| add | 5x5 | 0.60 | |
| add | 7x7 | 0.15 | |
| sub | 2x1 | 1.00 | |
| sub | 2x2 | 1.00 | |
| sub | 3x2 | 0.95 | |
| sub | 3x3 | 0.75 | |
| sub | 4x4 | 0.70 | |
| mul | 1x1 | 1.00 | |
| mul | 2x1 | 1.00 | |
| mul | 2x2 | 0.35 | |
| mul | 3x1 | 0.70 | |
| mul | 3x2 | 0.05 | |
| mul | 5x1 | 0.50 | |
| mul | 5x3 | 0.05 | |
| div | 2x1 | 1.00 | exact division only |
| div | 3x1 | 0.85 | |
| div | 4x2 | 0.50 | |

Two things stand out. Addition is near perfect to three digits and then falls
by about 0.3 between 3x3 and 4x4. Multiplication falls off a cliff at the
second factor digit: 2x1 is 1.00, 2x2 is 0.35, 3x2 is 0.05. The second factor
digit is where the model has to run the algorithm instead of recalling a table.

## Format does not rescue it

Same problems, four ways of asking. 20 per cell.

| cell | answer | bare | steps | column |
|---|---:|---:|---:|---:|
| add 3x3 | 0.90 | 0.90 | 0.35 | 0.40 |
| add 5x5 | 0.60 | 0.50 | 0.10 | 0.20 |
| sub 2x2 | 1.00 | 0.80 | 1.00 | 0.80 |
| sub 3x3 | 0.75 | 0.65 | 0.60 | 0.60 |
| mul 2x2 | 0.35 | 0.25 | 0.10 | 0.00 |
| mul 3x2 | 0.05 | 0.10 | 0.00 | 0.00 |

Writing the steps makes arithmetic worse, not better. The step-by-step replies
are fluent and correctly structured, and the final number is still wrong. The
model is generating the shape of a solution, not executing one. The `answer`
prefill is the best format and is the primary measure from here on. This is the
same lesson as the sycophancy work: a prefill concentrates the decision at one
token, and a free reply spreads it over several.

The column format is worse than useless. For `26 x 97` it produced a grid of
zeros with a digit header, which is a formatting prior winning over the
computation.

**Caveat added after the fact.** The `answer` prefill is the worst format for
word problems, not the best. On 60 generated two-step word problems it scores
0.13 against 0.23 with no prefill and 0.78 with a chain-of-thought prefill. The
grid above therefore measures the model with its reasoning suppressed, and the
numbers in it are a floor for anything that needs reasoning. They remain the
right numbers for raw column arithmetic, where chain of thought does not help
(steps is 0.35 on 3x3 addition and 0.10 on 5x5). See
[`04-astra-and-the-real-failure.md`](04-astra-and-the-real-failure.md).

## Carry count predicts the failure

460 addition problems from the grid, grouped by how many carries the correct
algorithm performs.

| carries | exact | |
|---:|---:|---|
| 0 | 0.90 | 37/41 |
| 1 | 0.86 | 43/50 |
| 2 | 0.72 | 23/32 |
| 3 | 0.72 | 13/18 |
| 4 | 0.50 | 5/10 |
| 5 | 0.14 | 1/7 |
| 6 | 0.00 | 0/2 |

Monotone, and steep at the top. That is the observation that motivated the
whole carry-editing program.

But the error shape does not match the carry story. If the model computed the
columns correctly and dropped the carry, the wrong answer would equal the
no-carry value.

```
addition errors equal to the no-carry value:        0/38 = 0.00
subtraction errors equal to the no-borrow value:    0/12 = 0.00
```

It never does that. Instead:

```
addition errors off by a power of ten:             21/38 = 0.55
subtraction errors with the correct ones digit:    10/12 = 0.83
multiplication errors within 10 percent:           62/67 = 0.93
multiplication errors matching a partial product:   0/67 = 0.00
```

The model is not dropping a carry. It is producing a nearby plausible number.
Half the addition errors are off by 10, 100 or 1000, in either direction. Nine
in ten multiplication errors are the right magnitude. Not one is a partial
product, which rules out "it computed the wrong row of the algorithm".

For multiplication, a 2x2 that is within 10 percent and is not a partial
product means the model has a smooth approximation of the product and is
rounding it to a token. That is a different failure from a missing carry, and
it explains why no carry edit moved multiplication.

## The atoms are reliable, the composition is not

Asked as separate prompts, the same model does the small steps well. 100
problems per row, majority over three phrasings.

| atom | first answer | majority |
|---|---:|---:|
| 2-digit x 1-digit | 0.98 | 0.98 |
| 3-digit x 1-digit | 0.82 | 0.83 |
| 4-digit x 1-digit | 0.73 | 0.75 |
| 5-digit x 1-digit | 0.59 | 0.62 |
| 3-digit x 2-digit | 0.12 | 0.15 |

And 3-digit addition is 0.86 on 100 held-out problems, rising to 0.92 with a
three-way majority vote.

The model can do `92 x 12` badly and `92 x 5` well. It can add two 3-digit
numbers at 0.86. It cannot chain those into `57892 x 512`. The failure is in
composition, not in the atomic facts.

## Eight digits is a span limit, not a carry limit

```
12345678 + 87654321  ->  90000000
99999999 + 1         ->  10000000
```

Both are wrong in a way that keeps the leading digits and zeroes the rest. This
is the model losing the operand tail, not mishandling a carry. It sets a hard
ceiling on any prompt-level fix for long addition, independent of any weight
edit.

## What this means for editing

The original hypothesis was that carry propagation is a state the model has but
does not propagate, and that strengthening that state would help. The map does
not support it. The errors are magnitude approximations, not dropped carries,
and the atoms are already reliable. There is no carry flag to turn up because
the model is not carrying in the first place.

That predicts the result in
[`02-carry-edits.md`](02-carry-edits.md): six edit families that target carry
and digit geometry all come back flat on held-out.

## What this does not show

- The 20-problem cells are noisy. The 3x3 to 4x4 drop is 0.90 to 0.60, which is
  about two standard errors, so the direction is real but the exact size is not.
- "Not a dropped carry" is established for addition and subtraction on this
  grid. It is not a claim about every error the model makes.
- The multiplication reading (smooth approximation, rounded to a token) is an
  inference from the error distribution. It is not a mechanistic claim.
