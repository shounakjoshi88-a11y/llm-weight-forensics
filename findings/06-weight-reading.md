# Finding 06: reading weights directly

"Sparks of Weight-Space Programming" (Takehi, Teoh, Zhang, Khattab, 2026) opens
with an LLM that identifies what a network was trained to do by looking at its
raw weights. The obvious objection is scale: 942 MB of floats is not readable by
anyone. So this is the same experiment at a scale where reading is possible, and
then a test of whether the reading is real or hindsight.

Raw inputs: [`e11_dump.txt`](e11_dump.txt) (exactly what was shown, labels
hidden), [`e11_prediction.json`](e11_prediction.json) (written before the reveal),
[`e11_blind_mapping.json`](e11_blind_mapping.json) (the hidden mapping).

## Setup

Five networks, identical architecture, **129 parameters each**: `2 -> 32 -> 1`
with ReLU. Trained to convergence on five different 2D classification tasks.

| task | label |
|---|---|
| vertical | `x > 0` |
| horizontal | `y > 0` |
| diagonal | `x + y > 0` |
| xor | `(x > 0) ^ (y > 0)` |
| circle | `x^2 + y^2 < 0.35` |

The weights were dumped with the task labels replaced by the slots A to E in a
shuffled order, showing the eight hidden units with the largest weight norm. The
identification was written to a file before the mapping was revealed.

## Result: 5 of 5, and the reasoning holds up

| slot | identified as | true task | train accuracy |
|---|---|---|---|
| A | circle | **circle** | 1.000 |
| B | diagonal | **diagonal** | 0.999 |
| C | horizontal | **horizontal** | 1.000 |
| D | xor | **xor** | 1.000 |
| E | vertical | **vertical** | 1.000 |

Each task was used exactly once, which is a consistency check on the set. The
decisive signature for each:

| slot | the number that gave it away |
|---|---|
| E | share of first-layer weight energy on x is **0.998**; every `w_y` is within 0.55 of zero |
| C | share on x is **0.028**; the mirror case, all signal on y |
| B | directions collapse onto one axis: 46, 39, 39, 38 degrees, and their negatives at 224, 225, 230. A monotone function of `x + y` |
| D | biases are **0.001 to 0.039** against `|w|` near 15, so every boundary passes through the origin. Output signs alternate by quadrant |
| A | the only net with directions spread over the full circle, **all** `w_out` negative, **all** biases negative, and a large positive output bias of +1.493. That is the subtract-outside construction |

The pattern that separates them is not the size of any single weight. It is the
geometry of the whole set: where the directions point, whether the output signs
agree, and how far the boundaries sit from the origin.

## The important part: a by-eye story can be hindsight

Five out of five is easy to over-read, because the reasoning was written after
the numbers were visible. So the reading was turned into an explicit rule and
scored on networks it had not seen.

### The obvious statistics fail

The first rule used three things one would reach for immediately:

- `e_x`: share of first-layer weight energy on x
- `concentration`: how tightly the individual hidden directions cluster
- `bias_ratio`: mean of `|b_i| / ||w_i||`

Result: **3 of 5** on the seed the rule was set on, **11 of 20** on held-out
seeds. The failure is systematic, and it is instructive.

`concentration` is the wrong statistic. For the diagonal task the optimiser does
not put every unit on the 45 degree axis; it spreads the work over many units
pointing in different directions, and lets the output weights sort them out. The
individual directions look like noise (concentration 0.03 to 0.29) while the
function is perfectly linear. The naive reader called all five diagonal networks
and all five xor networks "circle".

### The corrected statistic

For a network whose output is monotone in some direction, that direction is the
**output-weighted sum** of the first-layer weight vectors:

```
v = sum_i  w_out_i * w_i
```

and the useful quantity is how coherently those contributions add:

```
alignment = ||v|| / sum_i |w_out_i| * ||w_i||     in [0, 1]
angle     = atan2(v_y, v_x), folded into [0, 180)
```

When the function is monotone, every unit pushes the output the same way and
`alignment` is near 1, with `angle` equal to the discriminant direction. When
the function is not monotone, the contributions cancel and `alignment` collapses.

| true task | alignment | angle | mean boundary radius `|b_i| / ||w_i||` |
|---|---|---|---|
| vertical | 0.955 to 0.999 | 0.0 to 1.3 | 0.186 to 0.277 |
| horizontal | 0.976 to 0.999 | 89.6 to 90.8 | 0.195 to 0.293 |
| diagonal | 0.989 to 0.996 | 44.8 to 45.2 | 0.208 to 0.298 |
| xor | 0.005 to 0.009 | not meaningful | **0.030 to 0.064** |
| circle | 0.027 to 0.143 | not meaningful | **0.413 to 0.902** |

Ranges over 5 seeds per task. The gaps are wide: 0.143 to 0.955 on alignment,
0.064 to 0.413 on radius.

Decision rule, four lines:

```
if alignment >= 0.50:
    angle near 0   -> vertical
    angle near 90  -> horizontal
    otherwise      -> diagonal
else:
    radius < 0.10  -> xor
    otherwise      -> circle
```

Result:

| reader | set the rule on | held out |
|---|---|---|
| obvious statistics | 3/5 | 11/20 (55%) |
| output-weighted | 5/5 | **20/20 (100%)** |

Twenty-five of twenty-five in total, with the thresholds fixed using only the
seed that was read by eye. All 25 networks trained to at least 0.999 accuracy, so
none of this is reading a failed network.

## Controls

**Permuting the hidden units.** Renumbering the 32 hidden units does not change
the function. The reader is unchanged on 25 of 25 permuted networks, because it
looks at the set of weight vectors and their output weights, not at neuron
indices. This is the only exact symmetry a network of this shape has, and the
reading respects it.

**Swapping the two inputs.** Swapping the columns of the first layer maps
vertical to horizontal and back, and must leave diagonal, xor and circle alone.
The reader gets this right on 25 of 25.

## What the reading recovers

Three things, in increasing order of ambition:

1. **The class of function**, from a three-number summary, with no forward pass.
2. **The direction of the discriminant**, to within 1.3 degrees, as an actual
   angle in input space.
3. **Sometimes a numeric parameter.** The circle task has radius 0.592. The mean
   boundary distance came out at 0.582, 0.573 and 0.597 for three of the five
   seeds, within 0.02 of the true radius. The other two seeds gave 0.413 and
   0.902, so this is a partial result: the geometry is often recoverable
   quantitatively, not reliably.

## Why the label is visible at all

Gradient descent on a 2D classification problem with 32 ReLU units converges to a
solution where the output-weighted direction equals the task discriminant. That
is not surprising, but it is the reason the weights are readable: the label is
present as a low-dimensional summary of the whole parameter set, even though no
individual neuron encodes it. Reading is possible because the solution is
degenerate in a structured way, not because any single number is a label.

This is also why reading by eye is fragile. The informative quantity is a sum
over all 32 units, and the eight largest units alone do not always show it.

## Limits

- **129 parameters, not 494 million.** Nothing here establishes that the same
  statistics work on a transformer. Finding 04 is the counterweight: the standard
  weight-only readouts on Qwen2.5-0.5B return noise, for structural reasons
  (GQA, RoPE, RMSNorm, superposition).
- **The tasks were chosen by me**, and they are all linearly or radially
  separable in 2D. A task whose solution is a union of disjoint regions, or a
  parity of three inputs, might not leave a readable summary.
- **The reader is a decision tree over three numbers**, which is a classifier I
  wrote after seeing the structure. It is not an LLM reasoning over weights. It
  sets a lower bound: if four lines of arithmetic recover the answer, an LLM
  certainly can, so this is evidence for the premise rather than a test of the
  LLM's ability.
- **Twenty held-out networks is a small sample**, all drawn from the same five
  task families. It measures robustness to the optimiser's choice of solution,
  not generalisation to arbitrary tasks.

## What this says about the blog

The claim is that weights are readable. At 129 parameters, they demonstrably
are, and the readable content is a geometric summary rather than a lookup table.
That supports the premise.

But the pair of results across this repo bounds it. Readability is not a property
of weights in general; it is a property of a small network whose solution is
low-dimensional. At 0.5B parameters with grouped keys, rotary positions and
dense superposition, the naive summary statistics that work perfectly here return
correlations of -0.2 and -0.02 against measured importance. The interesting
research question is what the readable summary becomes at that scale, not whether
reading is possible at all.

## Reproduce

```bash
python experiments/e11_read_weights.py --dump     # anonymised weights
python experiments/e11_read_weights.py --reveal   # ground truth
python experiments/e11b_weight_reader.py          # 25 networks, both readers
```
