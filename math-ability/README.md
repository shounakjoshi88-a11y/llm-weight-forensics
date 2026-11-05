# Arithmetic ability in Qwen2.5-0.5B-Instruct: weight edits and a scaffold

Can a small model be made better at exact arithmetic by editing its existing
weights, with no LoRA, no adapters, no new parameters and no gradients?

Seven edits aimed at arithmetic did nothing. One edit aimed at the model's
default output format does move held-out accuracy, from 0.23 to 0.59 on word
problems, and it costs half the 5x5 addition accuracy. The reason is the same
in both cases: the model does not lack arithmetic, it fails to start doing any.
Given a one-sentence chain-of-thought prefill it reaches 0.95 with no weight
change at all. Given a decomposition into 2-digit atoms, it solves
`57892 x 512 = 29640704` exactly.

The only positive weight edit is a policy edit, not a capability edit. It makes
the model begin with `Step` instead of blurting a number, and it is strictly
worse than simply prompting for the same thing.

Neither of those is a weight edit. Both are the same statement: the model has
the parts, not the sequence.

This is also the correct reading of the Astra result that prompted the
question. Their 7.9 to 48.1 on MATH-500 is a loop removal on a model whose
baseline was crippled by the loop, and their own no-loop control did not
improve. Details and the numbers in
[`findings/04-astra-and-the-real-failure.md`](findings/04-astra-and-the-real-failure.md).

## Results

**Direct prompts, 20 problems per cell.** Full grid in
[`findings/01-failure-map.md`](findings/01-failure-map.md).

| cell | add | sub | mul |
|---|---:|---:|---:|
| 1x1 | 1.00 | | 1.00 |
| 2x1 | 1.00 | 1.00 | 1.00 |
| 2x2 | 1.00 | 1.00 | 0.35 |
| 3x3 | 0.90 | 0.75 | |
| 3x2 | 0.85 | 0.95 | 0.05 |
| 4x4 | 0.60 | 0.70 | |
| 5x5 | 0.60 | | |
| 5x3 | | | 0.05 |

Addition falls with the number of carries, but the errors are not the no-carry
value: 0 of 38 match it, while 55 percent are off by a power of ten. 93 percent
of multiplication errors are within 10 percent of the answer and none match a
partial product. The model is rounding an approximation, not running the wrong
algorithm.

**Word problems, 60 generated two-step problems.** Details in
[`findings/04-astra-and-the-real-failure.md`](findings/04-astra-and-the-real-failure.md)
and [`findings/05-one-edit-that-works.md`](findings/05-one-edit-that-works.md).

| condition | exact | 95% CI |
|---|---:|---|
| prefill `The answer is` | 0.13 | [0.07, 0.24] |
| no prefill | 0.23 | [0.14, 0.35] |
| prefill `Let me work through this step by step.` | 0.78 | [0.66, 0.87] |
| prefill `Step 1: ` | 0.92 | [0.82, 0.96] |

The last row is at 384 tokens, the others at 256. The format used for every
arithmetic number above is the worst of them. It forces an answer before the
model has reasoned, so the failure map measures the model with its reasoning
switched off.

**Weight edits, held-out.** Details and controls in
[`findings/02-carry-edits.md`](findings/02-carry-edits.md) and
[`findings/05-one-edit-that-works.md`](findings/05-one-edit-that-works.md).

| edit | what changes | held-out |
|---|---|---:|
| global gain | all of q, o or down | 0.633 vs 0.625 |
| carry rank-one | one `down_proj` | null, regression flat |
| layer gain search | one scalar per block | 0.661 vs 0.679 (overfit) |
| carry neurons | top-k `down_proj` rows | 0.91 vs 0.88, selected on eval |
| base blend | every tensor toward `Qwen2.5-0.5B` | 0.82 vs 0.81 |
| ` is` row | 896 numbers in `embed_tokens` | 0.732 vs 0.732 |
| digit bias | 10 rows of `lm_head` | 0.89 vs 0.90 |
| EOS row | 896 numbers in `lm_head` | 0.27 vs 0.27, rambles |
| `Step` row | 896 numbers in `lm_head` | **0.59 vs 0.23** on word problems |

Every arithmetic gap is inside one standard error at n = 100. The pattern
across those failures is that any edit large enough to change behaviour changes
it everywhere, in directions that disagree across operations.

The `Step` row is the exception and it is not an arithmetic edit. It raises
word-problem accuracy from 0.23 to 0.59 (41 fixed, 5 broken, p = 4.4e-8) by
making the model start with `Step`, and it costs 5x5 addition, 0.55 to 0.05.
It is worse than the `Step 1: ` prefill, which gets 0.95.

**Scaffold, 30 problems per cell.** Details in
[`findings/03-scaffolds.md`](findings/03-scaffolds.md).

| cell | direct | scaffold | model calls |
|---|---:|---:|---:|
| 2x2 | 0.35 | 1.00 | 3.9 |
| 3x2 | 0.05 | 1.00 | 6.3 |
| 4x2 | | 0.89 | 5.8 |
| 3x3 | | 0.87 | 11.0 |
| 5x3 | 0.05 | 0.81 | 15.5 |

The scaffold chunks the multiplicand in base 100 and the multiplier in base 10,
so every product is a 2-digit by 1-digit and every sum is 2-digit by 2-digit.
Each atom is asked three ways with a majority. The harness sequences and shifts
chunks and adds the chunk carry, a value of 0 or 1. It does no digit
multiplication and no multi-digit addition.

## What this adds up to

Three measurements, one conclusion.

- The atoms are reliable: 2-digit products at 0.95 to 0.98, 2-digit sums at
  0.97 to 0.99.
- The composition is absent: no edit installs it, and the errors are
  approximations rather than wrong steps.
- The model will reason if pushed into it, and then its arithmetic inside the
  reasoning is often right.

A weight edit cannot add a loop to a feedforward pass. It can remove a
degenerate behaviour, which is what Astra did, and that only pays when the
degenerate behaviour is what is capping the score. Here it is not.

## Reproduce

Windows, Python 3.12 or newer, a CUDA card with 6 GB. Set the thread limits or
BLAS oversubscribes and the runs get slower.

```powershell
cd D:\Q_project\learning\research\model-weights\math-ability
$env:OPENBLAS_NUM_THREADS="1"
$env:OMP_NUM_THREADS="1"

python experiments\e01_failure_map.py       # the grid and the format comparison
python experiments\e02_diagnosis.py         # offline error analysis, no model calls
python experiments\e03_scaling.py           # global gain sweep
python experiments\e04_carry_rank_one.py    # rank-one carry direction
python experiments\e05_gain_search.py       # per-layer gain search
python experiments\e06_carry_neurons.py     # carry neuron scaling
python experiments\e07_interpolate.py       # base checkpoint blend
python experiments\e08_embedding_search.py  # ` is` embedding row
python experiments\e10_lm_head_carry.py     # carry-conditioned digit bias
python experiments\e11_scaffold_v2.py       # 57892 x 512 and 28 five-by-three cases
python experiments\e12_scaffold_sweep.py    # scaffold across cells
python experiments\e13_atom_votes.py        # first answer vs majority per atom
python experiments\e14_loop_probe.py        # does it loop? no
python experiments\e15_continuation.py      # stop against chain of thought
python experiments\e16_stop_edit.py         # the EOS row edit
python experiments\e17_word_problems.py     # 60 word problems, three formats
python experiments\e18_math_mode.py         # math-mode direction, the blunt version
python experiments\e19_start_direction.py   # start-of-reply direction, six rows
python experiments\e20_openers.py           # which prefill opener works
python experiments\e21_step_row.py          # one row, large alpha, loops
python experiments\e22_alpha_scan.py        # where the first token flips
python experiments\e23_step_small_alpha.py  # the usable alpha window
python experiments\e24_confirm.py           # 100 fresh problems, paired test
python experiments\e25_edit_raw_arithmetic.py  # what the edit costs on arithmetic
```

Each writes a JSON file into `findings/`. The writeups quote only numbers that
appear in those files.

## Layout

```
math-ability/
  SPEC.md                    hypothesis, methods, decision rule, fixed before fitting
  README.md
  experiments/
    common.py                problem generation, prompting, scoring, model loading
    edits.py                 apply/restore tensors, evaluate, accuracy
    e01..e25                 one file per experiment
  findings/
    01-failure-map.md        where it breaks, and the shape of the errors
    02-carry-edits.md        seven edit families, all null on arithmetic
    03-scaffolds.md          the decomposition that works
    04-astra-and-the-real-failure.md
                             the Astra comparison, and the stop that is not a lever
    05-one-edit-that-works.md
                             the `Step` row, and what it costs
    *.json                   raw measurements
```

`common.py` imports `Probe` from `../probe/core.py` by path rather
than copying it. Generation uses a `FastModel` wrapper with SDPA attention,
which is about 1.8 s for 120 prompts at batch 60 on a 4050.

## Constraints this folder worked under

- No LoRA, no adapters, no new parameters, no gradient descent. Every edit
  changes existing tensors.
- Held-out problems are hashed before any fitting and are not used for
  selection, except where a finding says so (e06 does, and it is flagged).
- Everything here is committed to the public `llm-weight-forensics` repo. The
  checkpoints are gitignored and fetched with `python fetch_models.py`.

## What is not claimed

- No claim that the model is "better at maths" after any edit. It is not.
- No claim that the scaffold is the model solving the problem. It is the model
  solving 13 small problems and the harness assembling them.
- No claim that the chain-of-thought prefill is an edit. It is a measurement of
  what the model can do when it does not stop early.
- No claim that no weight edit could ever work. Seven families were tested and
  the diagnosis explains why they failed; a trained update is a different
  experiment and was excluded by the brief.

The related result in
[`../findings/07-weight-editing.md`](../findings/07-weight-editing.md)
is the contrast that makes this one legible: there, the ability was present and
a direction was suppressing it, and a gradient-free edit recovered it. Here the
ability is present and no direction is suppressing it; the model simply does
not begin.
