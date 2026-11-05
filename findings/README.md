# Findings index

Mechanistic interpretability of Qwen2.5-0.5B-Instruct, run locally on a 6 GB
laptop GPU. Each finding stands alone: method, numbers, what it does not show.

| # | Finding | Claim in one line |
|---|---|---|
| [01](01-anatomy.md) | Anatomy | Residual norm is not monotonic, three layers hold massive fixed activations, three heads look like textbook induction heads. |
| [02](02-causality.md) | Causality | Massive activations are input-independent biases and removing all three collapses the model 18x. The induction heads are not load-bearing. |
| [03](03-importance.md) | Importance | Attention pattern does not predict importance. Language and induction use almost entirely disjoint heads. |
| [04](04-weight-space.md) | Weight space | Classical weight-only circuit readouts fail on this GQA + RoPE model, for identifiable structural reasons. |

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
