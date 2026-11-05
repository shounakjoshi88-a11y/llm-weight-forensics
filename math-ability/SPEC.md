# Arithmetic editing: spec

Date: 2026-11-05
Status: done, results in `findings/`
Model: Qwen2.5-0.5B-Instruct
Folder: `D:\Q_project\learning\research\math-ability\`

## 1. Purpose

Make this model better at exact arithmetic by editing its existing weights
directly. No LoRA, no adapters, no new parameters, no gradient descent.

One method, E3, searches weights against a training objective without
gradients. It is included because it edits the real tensors and it is the
brute-force member of the portfolio, but it is flagged as tuning rather than
installing, and its result is reported separately from the interpretable edits.

The end target is `57892 x 512`. The route is to find where the model actually
breaks, edit that specific thing, and measure on numbers the edit never saw.

The working principle: change the content of the book, even if it is one page.

## 2. What the probes already show

Five probes on the loaded checkpoint, greedy decoding, exact final-integer
scoring:

| test | result |
|---|---|
| single-digit products | 5/5 |
| 2-digit x 1-digit | 4/5 |
| 2-digit x round tens | 4/4 |
| 3-digit addition | 3/3 |
| 2-digit subtraction with borrow | 0/3 |
| 5-digit x 1-digit | 4/4 |
| 57892 x 500, 57892 x 12, asked cleanly | both correct |
| 8-digit addition | 0/3 |
| 57892 x 512 direct | wrong |
| 2x2 bare | 1/5 |
| 2x2 step-by-step | 1/5 |
| 2x2 column format | 0/5 |

The subtraction errors are systematic:

```
72 - 38 = 14   (correct 34, borrow dropped)
91 - 47 = 34   (correct 44, borrow dropped)
63 - 28 = 15   (correct 35, borrow dropped)
```

The 2x2 addition errors are carry placement errors:

```
2820 + 141 = 3961   (correct 2961)
```

Writing the steps does not help. The model can do the atoms and loses the
carry.

## 3. Hypothesis

The limiting computation is multi-column carry and borrow propagation, not
arithmetic facts and not tokenisation. The model represents and executes
single-column operations, then fails to move the carry into the next column.

If that is true, a small edit that strengthens the carry state should improve
held-out addition and subtraction, and should improve the final addition step
of a decomposed `57892 x 512`.

The hypothesis is falsifiable. If all five edits fail on held-out carry-heavy
addition while the no-carry controls stay flat, the carry state is not a
direction or a set of neurons that a local edit can reach. That is a result and
will be reported as one.

## 4. Non-goals

- No LoRA, no fine-tuning, no adapters, no extra parameters.
- No changes to the tokenizer.
- No claims about general intelligence or "smarter in general".
- No memorising a benchmark of specific problems and calling it arithmetic.
- This folder is part of the public `llm-weight-forensics` repo. It ships code,
  writeups and the small result JSONs; the checkpoints themselves are
  gitignored and fetched with `python fetch_models.py`.

## 5. Model and tokenizer facts

- 24 layers, 896 hidden, 4864 intermediate, 14 query heads, 2 KV heads.
- The tokenizer splits every number into single digit tokens. `57892` is
  `['5','7','8','9','2']`, ids `[20,22,23,24,17]`. Digits `0` to `9` are ids
  `15` to `24`, each a single token.
- `lm_head` and `embed_tokens` are tied in the checkpoint. The experiment
  separates them into distinct tensors with identical values before any edit,
  as in e12, so an output-side edit does not rewrite the input embeddings.
- Qwen2 attention has `q_proj.bias`, `k_proj.bias` and `v_proj.bias`. The MLP
  has no biases. This matters for E1.

## 6. Layout

```
math-ability/
  SPEC.md
  README.md
  experiments/
    common.py            problem generation, prompting, scoring
    edits.py             tensor apply/restore, evaluation
    e01_failure_map.py   the grid and the format comparison
    e02_diagnosis.py     offline error-shape analysis
    e03_scaling.py       global gain sweep
    e04_carry_rank_one.py
    e05_gain_search.py
    e06_carry_neurons.py
    e07_interpolate.py
    e08_embedding_search.py
    e10_lm_head_carry.py
    e11_scaffold_v2.py   base-100 chunked scaffold
    e12_scaffold_sweep.py
    e13_atom_votes.py
  findings/
    01-failure-map.md
    02-carry-edits.md
    03-scaffolds.md
    *.json
```

`common.py` imports `Probe` from `../probe/core.py` by path.
There is one implementation of the instrumentation, not two.

## 7. Data

Generated, deterministic, seeded.

**Operations and cells.**

| op | digit cells |
|---|---|
| add | 1+1, 2+1, 2+2, 3+2, 3+3, 4+4, 5+5, 7+7 |
| sub | 2-1, 2-2, 3-2, 3-3, 4-4 |
| mul | 1x1, 2x1, 2x2, 3x1, 3x2, 5x1, 5x3 |
| div | 2/1, 3/1, 4/2, exact only |

**Splits.** Every problem is hashed and assigned to train or held-out by hash.
The same problem never appears in both. The held-out set is generated before
any edit is fitted and is not touched until the final evaluation of a method.

**Labels.** Each addition records the number of carries, the carry positions,
and the longest carry chain. Each subtraction records the same for borrows.
These labels drive the matched sets and the diagnosis.

**Matched carry sets.** For E1 and E2, pairs of addition problems matched on
digit length, total digit sum, and position of the changed digit, differing
only in whether a carry occurs. The matching is recorded, and the unmatched
residual is reported.

**Primary evaluation set.** 100 held-out additions with at least two carries,
operands of 3 to 5 digits. This is fixed before any edit is fitted.

**Size.** About 20 problems per cell for the map, 100 per cell for the primary
sets. Small enough to run on a 6 GB card, large enough for intervals.

## 8. Evaluation

**Prompt formats.**

```
bare     What is 47 + 68?
answer   What is 47 + 68?          assistant prefill: The answer is
steps    What is 47 + 68? Work through it step by step.
column   Compute 47 + 68 in column form. Write each column, then the total.
```

The `answer` format is the primary measure because it puts the first answer
digit at one decision point and removes formatting noise. `bare` and `steps`
are reported alongside it. This is the same lesson as e12: a prefill is a
stronger pressure than an open reply, and the difference is reported.

**Scoring.** Parse the last integer in the reply, with commas and spaces
removed. Exact match is the primary score. Also recorded: the first integer,
whether the reply contains the no-carry or no-borrow answer, and digit-level
edit distance to the correct answer.

**Intervals.** Wilson 95% interval per cell. Paired bootstrap over problems
for edit against base, 10,000 resamples. Exact sign test on per-problem
correctness.

**Regression checks.** No-carry addition, single-digit products, 20 short
language prompts, and perplexity on a 2,000-token public-domain slice. An edit
that buys carry accuracy by damaging language is not kept.

## 9. Methods

All five change existing tensors. None adds parameters. None uses gradient
descent.

### E1. Carry direction through a bias vector

Build `d = mean h(carry) - mean h(no carry)` at the last token before the
answer, from the matched pairs. Normalise.

Inject it by editing one `v_proj.bias` vector at layer L. Attention weights sum
to one, so adding `delta` to `v_proj.bias` adds `delta @ o_proj.weight` to the
attention output at every position. Choose

```
delta = alpha * d @ pinv(o_proj.weight[L])
```

so the residual shift is `alpha * d`. This changes 896 numbers in one vector.
Report the norm of `delta`.

Sweep L in {6, 10, 14, 18} and alpha in {0.5, 1, 2, 4} on the train split.
Select on train only. Evaluate the selected configuration once on held-out.

Controls: a random unit vector at the same norm, the sign-flipped direction, a
digit-magnitude direction built from pairs that differ in digit size but not in
carry, and the same edit at layer 0 and layer 23.

Known confound: carry and digit identity are correlated. The magnitude
direction and the within-problem column contrast are the checks. The column
contrast compares the residual at a column that carries against a column that
does not, inside the same column-format problem, so the operands are held
fixed. If the carry direction does not beat the magnitude direction on held-out
carry problems, the effect is not specific to carry.

### E2. Carry neurons

For each layer, score every MLP neuron by the standardised difference in mean
activation between carry and no-carry problems at the answer position. Take the
top k. Scale the corresponding `down_proj` rows by `(1 + beta)`.

Sweep k in {8, 32, 128} and beta in {0.25, 0.5, 1.0} on train. Evaluate once on
held-out.

Controls: random k neurons, bottom-k neurons, and the same edit in a layer
whose score distribution is flat.

### E3. Gradient-free low-rank search

Parameterise a rank-r update `sum_i u_i v_i^T` to one matrix in one layer.
Optimise with CMA-ES, no gradients, on 200 train problems. Objective is exact
accuracy plus a small weight on the likelihood of the correct answer digits.
Early stop on an internal validation split.

Budget: population 16, at most 300 generations, hard cap two hours. Report the
number of evaluations and the selected rank, layer and matrix.

Controls: a random update of the same norm, and the same search against shuffled
answers, which must not beat base.

This is the brute-force member of the portfolio. It is still weight editing:
the tensors change and nothing is added.

### E4. Digit embedding geometry

Extract the ten digit rows from `embed_tokens` and from `lm_head`. Measure
whether digit value is linearly ordered: Spearman correlation between digit
value and projection on the first principal component, and a ridge probe that
predicts digit value from the row.

If the ordering is weak, apply a linear map to those ten rows that enforces
order while preserving the subspace and the row norms. Test held-out
arithmetic.

Controls: a random rotation of the same ten rows, and the same map applied to
ten non-digit rows.

### E5. Fact injection control

Pick five products the model gets wrong, for example `26 x 4 = 104`. Apply a
closed-form rank-one edit to an MLP `down_proj`, ROME style, with the key taken
from the prompt's final hidden state and the value set to the target.

Test recall in the exact prompt and in paraphrases. Then test whether injecting
`26 x 4 = 104` improves `26 x 54`, which it should not.

This separates "an edit can add a fact" from "an edit can add an algorithm".
It is the control that keeps the rest honest.

## 10. Final evaluation

`57892 x 512` is measured three ways:

1. **Raw.** One prompt, no help.
2. **Decomposed.** Three separate prompts: `57892 x 500`, `57892 x 12`, then
   the sum of the two returned values. The model does every arithmetic step;
   the harness only sequences and passes values.
3. **Carry set.** The primary held-out carry-heavy addition set, which is the
   actual bottleneck.

The writeup reports how much of any final gain comes from the edit and how much
from the scaffold. A decomposed win with no edit gain is reported as a
scaffold result, not an editing result.

## 11. Success criteria and decision rule

**Primary.** Held-out carry-heavy addition exact accuracy improves over base
with a paired bootstrap 95% interval excluding zero and a sign test p < 0.05.

**Secondary.** Held-out 2x2 multiplication improves, or the decomposed
`57892 x 512` pipeline improves.

**Kept.** A method is kept only if the primary criterion passes and no
regression check drops by more than 2 points. If two methods pass, they are
combined only after the combination is itself evaluated on held-out.

**Negative.** A method that improves only the train split is reported as
overfitting. A null across all five is reported as a null, with the same detail
as a win. No method gets quietly dropped.

## 12. Risks

- The carry direction may be confounded with digit identity. Mitigated by the
  magnitude direction and the within-problem column contrast, not assumed away.
- A global bias injection can damage language. Mitigated by small alpha and the
  regression checks.
- Carry errors may be a depth limit rather than a reachable state. If so, E1 to
  E3 fail and the writeup says that the failure is architectural.
- Eight-digit addition may be out of reach for any local edit. The decomposed
  route is reported separately so this does not hide inside a headline number.
- CMA-ES may overfit 200 problems. Mitigated by a validation split, a
  shuffled-label control, and a held-out set that is never used for selection.
- The `answer` prefill is a strong pressure. Bare and step formats are reported
  so the prefill result is not the only result.

## 13. Artifacts

Each experiment writes a JSON file next to its writeup with per-problem
results, the selected configuration, the sweep, and the control results.
Findings are standalone: method, numbers, controls, and what the result does
not show.

The README states the headline result, the negative results, and the exact
command to reproduce each number.

## 14. Open questions for the review

- Is `math-ability` the folder name you want, or something else?
- Is the decomposed `57892 x 512` route acceptable, given that the harness
  sequences the steps but does none of the arithmetic?
- Should the base model `Qwen2.5-0.5B` be included as a second checkpoint, or
  keep the scope to the instruct model?
