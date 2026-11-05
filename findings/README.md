# Findings index

Mechanistic interpretability of Qwen2.5-0.5B-Instruct, run locally on a 6 GB
laptop GPU. Each finding stands alone: method, numbers, what it does not show.

| # | Finding | Claim in one line |
|---|---|---|
| [01](01-anatomy.md) | Anatomy | Residual norm is not monotonic, three layers hold massive fixed activations, three heads look like textbook induction heads. |
| [02](02-causality.md) | Causality | Massive activations are input-independent biases and removing all three collapses the model 18x. The induction heads are not load-bearing. |
| [03](03-importance.md) | Importance | Attention pattern does not predict importance. Language and induction use almost entirely disjoint heads. |
| [04](04-weight-space.md) | Weight space | Classical weight-only circuit readouts fail on this GQA + RoPE model, for identifiable structural reasons. |
| [05](05-neurons.md) | Neurons | Zero dead neurons, 98% activation density, and the sharpest neurons track subword continuations rather than concepts. |
| [06](06-weight-reading.md) | Reading weights | Five 129-parameter networks identified blind from their weights, 5/5, and a three-statistic reader holds at 20/20 on held-out seeds. |
| [07](07-weight-editing.md) | Editing weights | A gradient-free read-side weight edit flips 18% of 51 held-out wrong-suggestion facts with almost no neutral loss, but the editable direction is the user's assertion, not a wrongness detector. |

Raw measurements are the `e0N_*.json` files next to each writeup.

## The rule this follows

Every claim has to survive three tests before it goes in a finding:

1. **A control.** Random ablation of the same size. Without it, "removal changed
   the output" is meaningless, because removal always changes the output.
2. **A sharper measurement when the first one returns null.** E02 measured
   induction by accuracy and got exactly zero change. That was a blunt instrument,
   not a finding, so E03 replaced it with a logit margin.
3. **A stated failure mode.** Each writeup ends with what the experiment cannot
   show. The permutation caveat in the main README came from the same habit.

## Things that turned out to be wrong along the way

Recorded because negative results are cheap to produce and expensive to
rediscover:

- **Attention pattern is not function.** The three heads with near-perfect
  induction scores rank 200, 254 and 336 out of 336 on the induction task. One of
  them improves the model when removed.
- **The induction metric measured the wrong thing at first.** Removing a diffuse
  layer-0 head hurts the induction margin because it hurts everything. Splitting
  the sweeps by task (finding 03) was required to see this.
- **Per-head importance is not a property of a head.** It is a property of a head
  and a task together. The two maps are uncorrelated at r = -0.016.
- **The obvious weight statistics are the wrong statistics.** Clustering the
  individual hidden directions of a small network looks like it should identify
  the task and does not: 11 of 20 held out. The optimiser spreads a linear
  function over units pointing many ways. Only the output-weighted sum,
  `sum_i w_out_i * w_i`, recovers the direction, and that reader gets 20 of 20.
- **A 5/5 by-eye reading is not evidence until it is repeated blind.** The first
  written rule scored 3 of 5 on the very networks it was set on.
- **The write-side edit cascades.** Removing the suggestion direction from the
  early `o_proj` and `down_proj` writes shifts the held-out margin by +2.01, but
  flips only 8% and costs neutral accuracy (4.59 to 4.17). The same direction
  removed on the read side, above layer 18, flips 18% and leaves the neutral
  margin at 4.53. The surgical edit is the one that works.
- **The direction is not a wrongness detector.** A wrong suggestion and a correct
  suggestion produce nearly the same residual shift at layer 18 (norms 12.26 and
  12.95, cosine 0.985). The wrong-minus-correct content direction is small (norm
  2.31) and nearly orthogonal, and removing it barely moves the margin (+0.55,
  2% flips). What the edit removes is the user's assertion, not a judgement that
  the answer is false.
- **A missing ability is not editable.** Two-digit multiplication is 0 of 5 on
  this model. The same kind of edit has nothing to remove, which is the analogue
  of the blog's negative control on Gemma 3 270M. Sycophancy is editable because
  the ability is present and the obstacle is a direction.
