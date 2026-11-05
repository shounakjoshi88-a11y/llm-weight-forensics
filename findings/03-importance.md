# Finding 03: attention pattern does not predict importance

E01 identified heads by what their attention looked like. This measures what each
of the 336 heads actually does, on two different tasks, and compares.

Raw numbers: [`e04_head_importance.json`](e04_head_importance.json),
[`e05_induction_importance.json`](e05_induction_importance.json),
[`e06_layer0.json`](e06_layer0.json), [`e07_specificity.json`](e07_specificity.json).

## Method

Every head is ablated individually, one at a time, 336 forward passes per task.

- **language task**: perplexity increase on a 160-token passage
- **induction task**: induction margin decrease on a repeated random sequence

Damage is recorded as a positive number. Specificity is the z-scored induction
damage minus the z-scored perplexity damage, so a head that hurts copying more
than it hurts prose scores high.

## Result 1: the three "perfect induction heads" are near the bottom

| head | induction score (E01) | rank of 336 (language) | rank of 336 (induction) |
|---|---|---|---|
| L9H13 | 0.958 | 200 | 146 |
| L11H12 | 0.954 | **254** | **336 (last)** |
| L16H3 | 0.992 | 222 | 28 |

L11H12 is the single least important head for induction in the entire model, at
-0.482 damage, meaning removing it improves the margin. Its attention pattern is
as close to textbook induction as anything in the network, and it contributes
nothing measurable.

## Result 2: the two importance maps are unrelated

Correlation between per-head perplexity damage and per-head induction damage:

```
r = -0.016
shared variance = 0.0%
```

**The heads that matter for language modeling and the heads that matter for
in-context copying are almost entirely different heads.** Not negatively
related, not positively related. Independent.

That is the central result of this experiment. A single "importance" number for a
head does not exist. Importance is a property of a head and a task together.

## Result 3: the induction-critical heads are elsewhere

Ranked by specificity, the heads that carry in-context copying:

| rank | head | specificity | induction damage | ppl damage |
|---|---|---|---|---|
| 1 | L0H3 | +10.76 | +8.852 | -0.322 |
| 2 | L0H8 | +9.53 | +9.050 | +0.336 |
| 3 | L2H5 | +7.20 | +6.579 | +0.163 |
| 4 | L8H7 | +5.55 | +4.092 | -0.326 |
| 5 | L0H2 | +3.92 | +6.473 | +1.604 |
| 7 | L10H2 | +2.57 | +2.211 | +0.097 |
| 8 | L15H2 | +2.39 | +2.431 | +0.289 |

Two things stand out.

**Layer 0 dominates.** L0H3 and L0H8 are the largest single contributors to
induction in the model, at +8.85 and +9.05. Layer 0 total induction damage is
+30.5, four times the next highest layer. Layer 0 attends directly to token
embeddings, before any transformer block has processed them.

**L8H7 is the real previous-token head.** It has a previous-token score of 0.915
and specificity +5.55. That is the shape a real circuit component should have:
a clean, interpretable attention pattern that also survives ablation on the task
it is supposed to serve.

## Result 4: layer 0 heads are diffuse, not specialised

If layer 0 dominates induction, it should be doing something induction-like.
It is not.

| head | entropy | previous-token score | attention to offset -1 |
|---|---|---|---|
| L0H8 | 0.782 | 0.071 | 0.071 |
| L0H3 | 0.925 | 0.042 | 0.042 |
| L0H2 | 0.871 | 0.075 | 0.075 |

These heads attend almost uniformly. Entropy near 0.9 out of a maximum of 1.0
means the distribution is close to flat. They are not matching prefixes or
tracking position. They smear a broad average of everything seen so far into the
residual stream.

Their contribution to the stream is small in magnitude (L0H8 is 1.3% of the
stream norm, L0H3 is 3.0%), but removing them is the single most damaging
intervention on the induction task. A broad, low-magnitude, high-entropy signal
that the rest of the network depends on.

## Result 5: some heads are actively harmful

Several heads have negative perplexity damage, meaning removing them makes the
model better on this passage:

| head | perplexity damage |
|---|---|
| L4H12 | -0.779 |
| L12H0 | -0.757 |
| L5H10 | -0.598 |

A 0.5B model trained on a limited budget has heads that are net negative on some
inputs. This is a known consequence of a small model with a shared training
objective, and it is measurable directly.

## Result 6: layer importance is also task-dependent

| layer | induction damage | perplexity damage |
|---|---|---|
| 0 | **+30.50** | +4.82 |
| 1 | -0.22 | **+13.43** |
| 14 | +1.21 | +7.10 |
| 20 | +0.37 | +0.30 |

Layer 1 is the most important layer for prose and completely irrelevant for
induction. Layer 0 is the reverse. The network does not have a single
"importance profile", it has one per task.

## What this does not show

- Two tasks is not a characterisation of the model. There are many more behaviours
  and each would produce a different map.
- Perplexity on one passage is noisy at the head level. The large effects (above
  2.0) are reliable; differences of 0.1 are not.
- Zeroing removes a head's output but leaves its attention computation running.
  A cleaner intervention would also remove the head's effect on the key/value
  cache, which matters for later positions.
- "Specificity" is a normalisation choice, not a physical quantity. It ranks
  heads usefully but the absolute numbers mean nothing.

## Reproduce

```bash
python experiments/e04_head_importance.py
python experiments/e05_induction_importance.py
python experiments/e06_layer0.py
python experiments/e07_specificity.py
```
