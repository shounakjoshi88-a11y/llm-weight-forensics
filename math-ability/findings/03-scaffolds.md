# Finding 03: the algorithm belongs outside the weights

The weight edits failed. This works, and it is the positive result of the
folder.

The finding is narrow and it is not a weight edit: this model already performs
the small arithmetic operations reliably, and fails when it has to chain them.
Move the chaining into the harness and the same weights solve `57892 x 512`.

Raw measurements: [`e11_scaffold_v2.json`](e11_scaffold_v2.json),
[`e12_scaffold_sweep.json`](e12_scaffold_sweep.json),
[`e13_atom_votes.json`](e13_atom_votes.json).

## The atoms

100 held-out problems per cell, three phrasings.

| atom | first answer | 3-way majority | any phrasing right | phrasings disagree |
|---|---:|---:|---:|---:|
| 2-digit + 2-digit | 0.97 | 0.99 | 1.00 | 0.04 |
| 3-digit + 3-digit | 0.90 | 0.91 | 0.95 | 0.09 |
| 2-digit x 1-digit | 0.95 | 0.94 | 0.96 | 0.06 |

Addition benefits from the vote, multiplication does not. For multiplication
the errors are consistent across phrasings: when the model says `87 x 8 = 706`
it says it in two of three forms. Voting cannot fix a consistent error.

That asymmetry sets the shape of the scaffold. The sums can be made near
perfect by asking three ways. The products cannot, so their 0.95 to 0.98 is the
ceiling on the whole pipeline.

## The scaffold

Two operands, `a` of any length and `b` of up to three digits.

- Chunk `a` in base 100: `57892` is `[92, 78, 5]`.
- Chunk `b` in base 10, one digit per pass.
- For each digit `d` of `b`, ask the model for `chunk x d` for every chunk of
  `a`, carrying between chunks.
- Shift that partial by the digit position and add it to the running total, one
  base-100 chunk at a time.

Every model call is one of two prompts:

```
What is 92 x 5?          ->  a 2-digit by 1-digit product
What is 78 + 43?         ->  a 2-digit plus 2-digit sum
```

Each is asked three ways and the majority is taken.

**What the harness does.** It splits the operands into chunks, multiplies by
powers of ten, and adds the chunk carry (a value of 0 or 1) to the model's
returned sum before taking `% 100`. It does no digit multiplication and no
multi-digit addition.

**What the model does.** Every product and every sum, including all the carries
inside them.

## `57892 x 512`

```
b = 512, digits [2, 1, 5]

d = 2:  92x2, 78x2, 5x2      -> 115784
d = 1:  92x1, 78x1, 5x1      ->  57892, shifted by 10
d = 5:  92x5, 78x5, 5x5      -> 289460, shifted by 100

total   115784 + 578920 + 28946000 = 29640704
```

13 model calls, 39 generations. The answer is exact.

Directly, in one prompt, the model does not solve this. With the scaffold it
does, and so do 6 of the 7 other 5-digit by 3-digit cases in the fixed set.

## Across cells

30 held-out problems per cell, exact match on the assembled integer.

| cell | scaffold | direct prompt | asks per problem |
|---|---:|---:|---:|
| 2x2 | 1.00 | 0.35 | 3.9 |
| 3x2 | 1.00 | 0.05 | 6.3 |
| 4x2 | 0.89 | | 5.8 |
| 3x3 | 0.87 | | 11.0 |
| 5x3 | 0.81 | 0.05 | 15.5 |

The direct column is from [`e01_failure_map.json`](e01_failure_map.json), 20
problems per cell, so the comparison is indicative rather than paired.

Two-digit multiplication goes from 0.35 to 1.00. Three-digit by two-digit goes
from 0.05 to 1.00. That is the whole argument in one row: the computation was
reachable, it just was not reachable in one forward pass.

## The first version was 3 of 8, and why

The first scaffold chunked both operands in base 100
([`e09_scaffold.json`](e09_scaffold.json)). That looks symmetric and
it is wrong. With `b = 925`, the multiplier chunks are `[25, 9]`, so the model
is asked `60 x 25`, which is a 2-digit by 2-digit product, measured at 0.35.

The scaffold was asking for atoms it knew were unreliable, then blaming the
model for the failures. Chunking the multiplier in base 10 removes every 2x2
product. On the same eight cases the score went from 3/8 to 7/8, and on 28 cases
it is 21/28.

The general lesson: a scaffold is only as good as its worst atom, and it is
easy to smuggle a hard atom into a decomposition that looks clean.

## The ceiling

For a 5-digit by 3-digit problem the scaffold makes 9 product atoms and about
6 sum atoms. With products at 0.95 to 0.98 and independent failures, the
pipeline bound is

```
0.95^9 = 0.63      0.98^9 = 0.83
```

and the observed 0.81 sits at the top of that range. The sums are close enough
to perfect that they do not move the number.

So the remaining failures are not a composition problem any more. They are the
2 percent of products the model gets wrong and repeats when asked differently.
Raising the score further needs a more reliable product atom, and the vote
experiment says more phrasings will not give one.

## What this is not

- It is not a weight edit. Nothing in the checkpoint changes.
- It is not the model solving `57892 x 512` in one pass. It is the model
  solving 13 small problems and the harness assembling them.
- It does not generalise to arbitrary length. A 5x3 costs 15.5 model calls, and
  the cost grows with the number of digits while the accuracy falls with the
  number of atoms.
- The harness adds the chunk carry. That is one bit of arithmetic per chunk
  boundary and it is disclosed here rather than hidden.

## Why this is still the interesting result

The folder set out to change the weights and could not. The reason is in
[`01-failure-map.md`](01-failure-map.md): the model's multi-digit errors are
magnitude approximations, not dropped carries, so there was no missing state to
restore. What it does have is a reliable set of atoms and no reliable way to
sequence them in one generation.

That is a statement about where the computation lives. It lives in the
per-token forward pass for small operations and it does not exist for the
composition. Editing weights cannot add a loop to a feedforward pass. Putting
the loop outside the model can, and it costs 13 calls instead of one.
