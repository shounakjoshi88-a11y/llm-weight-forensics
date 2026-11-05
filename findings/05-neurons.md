# Finding 05: the neuron atlas

What each of the 116,736 MLP neurons responds to, measured over 30,148 tokens of
public domain text.

Raw data: [`e10_neuron_atlas.json.gz`](e10_neuron_atlas.json.gz) (every neuron,
its top 8 contexts and statistics, gzipped because the raw file is 121 MB),
[`e10_summary.json`](e10_summary.json).

```bash
gunzip -k findings/e10_neuron_atlas.json.gz
```

## Method

The model was run over a corpus of four Gutenberg books, and for every MLP neuron
the eight positions with the largest absolute activation were retained along with
twelve tokens of surrounding context.

This is descriptive. A neuron is characterised by what excites it, which is the
only way to describe one without guessing. It does not establish that the neuron
causes the behaviour. Causal claims need ablation, which is finding 02.

Statistics per neuron:

- `rms`: root mean square activation over the whole corpus
- `freq`: fraction of positions with activation above 1e-3
- `selectivity`: max divided by rms, a measure of how peaked the neuron is

## Result 1: there are no dead neurons

| | count |
|---|---|
| total neurons | 116,736 |
| dead (max activation below 0.01) | **0** |

Not one neuron in the model is dead. Every single one fires above threshold
somewhere in 30k tokens of ordinary text.

This matters because a common claim about transformer MLPs is that they are
sparse, with only a small fraction of neurons active per token. That claim is
false here. The median activation density is **0.98**: 98% of neurons have a
non-zero activation at a typical position.

Sparse autoencoders are still useful, but they are finding structure inside a
dense superposition, not pruning a mostly empty layer.

## Result 2: selectivity spans a wide range

| percentile | selectivity (max / rms) |
|---|---|
| 50 | 17.4 |
| 90 | 36.0 |
| 99 | 63.8 |
| 99.9 | 91.2 |
| max | 138.6 |

Most neurons are broadly active, with a peak about 17 times their typical level.
The top 0.1% peak nearly 100 times their typical level. Those are the ones that
look like detectors rather than mixers.

## Result 3: the sharpest neurons fire on word continuations

The most selective neurons, with `[[ ]]` marking the firing token:

| neuron | selectivity | context |
|---|---|---|
| L23N301 | 118.2 | `Sense and [[S]]ensibility` / `playing and [[s]]inging too long` |
| L23N1202 | 114.0 | `Elizabeth [[rem]]ained with no very cordial feelings` |
| L17N2611 | 113.3 | `by [[net]]ting a purse` / `[[net]] purses` |
| L1N1701 | 112.3 | `You take [[del]]ight in vexing me` |
| L19N4258 | 128.8 | `five-and-tw[[enty]]` / `early eight[[eenth]] century` |

A pattern repeats across the sharpest neurons: they fire on the **second token of
a word that the tokenizer split**, or on a rare word. `rem|ained`, `net|ting`,
`del|ight`, `am|usement`, `an|xiety`, `pass|ing`.

The honest reading is that these are not "concept neurons" for remaining or
delight. They are closer to detectors for the continuation of a low frequency
subword, which is exactly the kind of thing a next-token predictor needs. The
model spends neurons on the long tail of the vocabulary.

This is also a warning about the method: the highest absolute activations cluster
on rare tokens, so "top activating contexts" over-represents the long tail.

## Result 4: activation magnitude grows with depth

Median RMS activation per layer rises steadily:

| layer | median rms |
|---|---|
| 0 | 0.070 |
| 6 | 0.118 |
| 12 | 0.098 |
| 18 | 0.175 |
| 23 | 0.388 |

The final layer is about 5.5 times the first. This is the same depth trend as the
residual stream norm in finding 01, which grew 90x across the network. The MLP
layers near the output are writing with much larger magnitude than the early
ones, consistent with the model committing to a token at the end.

## What this does not show

- The corpus is 19th century English fiction. A different corpus would produce
  different top contexts for every neuron, and probably different selectivity
  rankings.
- Contexts are selected by largest activation, which biases toward rare tokens.
  A neuron that reliably tracks a common concept at moderate activation would not
  surface this way.
- Descriptive only. Nothing here says a neuron is necessary for the behaviour it
  appears to detect.
- 30k tokens is a small sample for 116,736 neurons: about 0.26 activations per
  neuron per position on average. Rare detectors are undersampled.

## Reproduce

```bash
python experiments/e10_neuron_atlas.py --tokens 30000
```
