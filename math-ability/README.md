# Arithmetic ability in Qwen2.5-0.5B-Instruct: weight edits, a scaffold, and a fine-tune

Can a small model be made better at exact arithmetic?

The first half of this folder answers that under a hard constraint: existing
weights only, no LoRA, no adapters, no new parameters and no gradients. Under
that constraint seven edits aimed at arithmetic did nothing, and the reason
turned out to be measurable. At every digit position a closed-form probe from
the residual matches the model's own accuracy, so there is no hidden answer
waiting for a better readout. The readout is nonetheless a couple of points
worse than a closed-form fit to those same states, and installing a quarter of
that correction raises held-out 5x5 addition from 0.60 to 0.65 while costing 3x2
multiplication, 0.15 to 0.11. That is the first arithmetic weight edit in this
folder that moves held-out accuracy. It is small because it is two points per
digit, and exact match multiplies six digits together.

The constraint was then lifted. With new parameters and gradient descent
allowed, a LoRA fine-tune on generated worked solutions does what none of the
edits could: held-out 6x6 addition 0.35 to 0.97, 3x3 multiplication 0.00 to
0.72, 5x3 multiplication 0.00 to 0.42, on problems the training corpus never
contained. The model learned the procedure. It did not learn to finish it: the
remaining errors are a dropped borrow in subtraction, a mis-transcribed final
answer in addition, and a wrong last addition in multiplication. Details in
[`findings/07-gradients-work.md`](findings/07-gradients-work.md).

Two other results are not weight edits at all. A chain-of-thought prefill takes
word problems from 0.23 to 0.95, and a base-100 decomposition solves
`57892 x 512 = 29640704` exactly. A policy edit can make the model start with
`Step` on its own, 0.23 to 0.59, but it is worse than the prefill and it costs
half the 5x5 addition accuracy.

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
| digit rows refit | 10 rows of `lm_head`, closed form | **0.65 vs 0.60** on 5x5 addition |

Every gap in the first eight rows is inside one standard error at n = 100. The
pattern across those failures is that any edit large enough to change behaviour
changes it everywhere, in directions that disagree across operations.

The `Step` row is not an arithmetic edit. It raises word-problem accuracy from
0.23 to 0.59 (41 fixed, 5 broken, p = 4.4e-8) by making the model start with
`Step`, and it costs 5x5 addition, 0.55 to 0.05. It is worse than the `Step 1: `
prefill, which gets 0.95.

The digit-row refit is an arithmetic edit and it is the only one that works. It
is covered below.

**The readout, held-out.** Details in
[`findings/06-readout-is-the-bottleneck.md`](findings/06-readout-is-the-bottleneck.md).

At each digit position, with the correct prefix fed in, a closed-form probe from
the normed residual matches the model's own accuracy almost exactly. For 5x5
addition: model 0.97, 0.90, 0.90, 0.87, 0.89, 0.98 against probe 1.00, 0.88,
0.90, 0.85, 0.92, 0.98, with the shuffled control at chance. There is no hidden
correct digit to recover.

The readout is still a couple of points worse than a least-squares fit to those
same states, so replacing a quarter of the ten digit rows of `lm_head` with the
fit buys a little:

| task | base | refit | fixed / broken | p |
|---|---:|---:|---:|---:|
| 5x5 addition, 300 held out | 0.60 | 0.65 | 18 / 3 | 0.001 |
| 5x5 addition, shuffled labels | 0.60 | 0.55 | 8 / 22 | 0.016 |
| 3x2 multiplication, 300 held out | 0.15 | 0.11 | 4 / 15 | 0.019 |

8,960 numbers, solved in closed form, no gradients and no new parameters. The
gain holds on five-digit sums (81 to 88) and six-digit sums (99 to 107), so it
is not the constant leading digit of a six-digit sum. It is a readout
correction, and it trades addition for multiplication because the ten rows are
shared.

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

**Fine-tune, held-out.** Details in
[`findings/07-gradients-work.md`](findings/07-gradients-work.md). LoRA r=32 on
all attention and MLP projections, 17.6M trainable parameters, 2 epochs, 572
steps, trained on 9,147 generated worked solutions. 40 held-out problems per
cell, natural prompting.

| cell | base | tuned | delta |
|---|---:|---:|---:|
| add 3x3 | 0.80 | 1.00 | +0.20 |
| add 5x5 | 0.57 | 0.97 | +0.40 |
| add 6x6 | 0.35 | 0.97 | +0.62 |
| sub 5x4 | 0.50 | 0.75 | +0.25 |
| mul 2x2 | 0.50 | 1.00 | +0.50 |
| mul 3x2 | 0.17 | 0.90 | +0.73 |
| mul 3x3 | 0.00 | 0.72 | +0.72 |
| mul 4x3 | 0.00 | 0.50 | +0.50 |
| mul 5x3 | 0.00 | 0.42 | +0.42 |

Zero of the 560 held-out eval problems appear in the training corpus, and the
cells that were barely trained (1x1) stayed flat, so this is generalisation and
not a lookup table. Word problems go 0.17 to 0.90 with no prefill.

The gain lives in the written procedure for multiplication. The `The answer is`
prefill, which skips the steps, leaves 3x3 and 5x3 multiplication at 0.00, while
addition improves even in one shot (5x5 prefill 0.65 to 0.93). So the weights
learned addition; multiplication only works when the model writes the algorithm
down.

The remaining failures are specific. Addition is locally correct in every
column and then transcribes the final answer wrong (3 to 5 percent of replies).
Subtraction drops a borrow into the next column (internal consistency 0.65 at
5x4, against 1.00 for the lines taken individually). Multiplication gets the
partial products right 82 to 100 percent of the time and then botches the final
sum of the shifted products, which is why `57892 x 512` is still wrong: all
three partials are correct, the sum of them is not. The cost is an 8 percent
rise in prose perplexity and occasional arithmetic formatting on non-arithmetic
questions.

## What this adds up to

Five measurements, one conclusion.

- The atoms are reliable: 2-digit products at 0.95 to 0.98, 2-digit sums at
  0.97 to 0.99.
- The per-digit error rate is the cap. At every position the residual holds
  what the readout extracts, and exact match multiplies those rates together.
  Addition is 0.87 to 0.98 per digit, which multiplies to 0.59 exact.
- The composition is absent: no large weight edit installs it, and the errors
  are approximations rather than wrong steps.
- The model will reason if pushed into it, and then its arithmetic inside the
  reasoning is often right.
- Composition is learnable with gradients. The fine-tune installs the
  procedure, and what is left is not composition but three specific endgame
  errors: the final addition, the borrow chain, and the transcription of the
  answer.

The one arithmetic edit that works moves the per-digit rate by about two
points, which is five points of exact answers. That is the size of the effect
available at the output layer, and it is why the large edits failed: they moved
everything at once.

A weight edit cannot add a loop to a feedforward pass. It can remove a
degenerate behaviour, which is what Astra did, and that only pays when the
degenerate behaviour is what is capping the score. Here the cap is a per-digit
error rate, not a degenerate behaviour. A trained update can add a procedure,
which is why it clears the cap that no edit could.

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
python experiments\e26_digit_trajectory.py  # is the digit in the residual?
python experiments\e27_readout_refit.py     # replace the digit rows, closed form
python experiments\e28_confirm_readout.py   # fresh problems, beta sweep, control
python experiments\e29_unrestricted.py      # does the gain survive varied lengths?
python experiments\e30_pooled_readout.py    # one readout for both operations

# fine-tune, new parameters and gradients allowed
python finetune\data.py                     # build and verify the corpus
python finetune\train.py                    # LoRA, about 30 minutes on a 4050
python finetune\eval.py                     # base vs adapter, held-out
python finetune\diagnose.py                 # localise the remaining failures
python finetune\collateral.py               # prose perplexity and non-math prompts
```

Each experiment writes a JSON file into `findings/`. The writeups quote only
numbers that appear in those files. The fine-tune scripts print their results;
`eval.py` also writes `findings/e31_finetune_eval.json`.

## Layout

```
math-ability/
  SPEC.md                    hypothesis, methods, decision rule, fixed before fitting
  README.md
  experiments/
    common.py                problem generation, prompting, scoring, model loading
    edits.py                 apply/restore tensors, evaluate, accuracy
    e01..e30                 one file per experiment
  finetune/
    data.py                  generated corpus, arithmetic validator
    train.py                 LoRA fine-tune
    eval.py                  base vs adapter on held-out cells
    diagnose.py              localise failures inside the worked steps
    collateral.py            prose perplexity and non-math prompts
    adapter/                 trained adapter, gitignored
  findings/
    01-failure-map.md        where it breaks, and the shape of the errors
    02-carry-edits.md        seven edit families, all null on arithmetic
    03-scaffolds.md          the decomposition that works
    04-astra-and-the-real-failure.md
                             the Astra comparison, and the stop that is not a lever
    05-one-edit-that-works.md
                             the `Step` row, and what it costs
    06-readout-is-the-bottleneck.md
                             the probe, the digit-row refit, and the trade
    07-gradients-work.md     the fine-tune, the held-out gain, and the three
                             remaining bugs
    *.json                   raw measurements
```

`common.py` imports `Probe` from `../probe/core.py` by path rather
than copying it. Generation uses a `FastModel` wrapper with SDPA attention,
which is about 1.8 s for 120 prompts at batch 60 on a 4050.

## Constraints this folder worked under

- The first thirty experiments: no LoRA, no adapters, no new parameters, no
  gradient descent. Every edit changes existing tensors. That constraint was
  lifted for the fine-tune in
  [`findings/07-gradients-work.md`](findings/07-gradients-work.md), which is the
  only gradient-based result here.
- Held-out problems are hashed before any fitting and are not used for
  selection, except where a finding says so (e06 does, and it is flagged). The
  fine-tune trains only on the `train` split, and zero of its 560 held-out eval
  problems appear in the corpus.
- Everything here is committed to the public `llm-weight-forensics` repo. The
  checkpoints and the trained adapter are gitignored and fetched or rebuilt
  locally.

## What is not claimed

- No claim that the model is "better at maths" after any weight edit. It is not.
- No claim that the fine-tune solves arithmetic. It raises held-out exact match
  a lot and still fails on multi-digit multiplication's final addition, on
  subtraction borrow chains, and on transcribing its own answer.
- No claim that the scaffold is the model solving the problem. It is the model
  solving 13 small problems and the harness assembling them.
- No claim that the chain-of-thought prefill is an edit. It is a measurement of
  what the model can do when it does not stop early.
- No claim that no weight edit could ever work. Seven families were tested and
  the diagnosis explains why they failed; the trained update is a different
  experiment and is reported separately.

The related result in
[`../findings/07-weight-editing.md`](../findings/07-weight-editing.md)
is the contrast that makes this one legible: there, the ability was present and
a direction was suppressing it, and a gradient-free edit recovered it. Here the
ability is present and no direction is suppressing it; the model simply does
not begin.
