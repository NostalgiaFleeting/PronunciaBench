# Confidence Model Documentation

## Overview

`ReliabilityScorer` computes a heuristic confidence score from five observable features.
This is **NOT** a calibrated probability estimate. It is a relative indicator of
prediction reliability that should be used for ranking and thresholding, not as
`P(correct)`.

## Components

| Component | Weight | Description |
|-----------|--------|-------------|
| `model_agreement` | 30% | 1 - mean pairwise normalized phoneme distance across model predictions |
| `sequence_confidence` | 25% | Mean of model-provided confidence scores, or length-consistency heuristic |
| `locale_support` | 15% | Heuristic: 1.0 for well-supported locales, 0.7 for valid BCP 47 codes, 0.3 otherwise |
| `prediction_stability` | 20% | Fraction of identical normalized prediction pairs |
| `oov_signal` | 10% | 1 - fraction of predictions matching fallback pattern `/[...]/` |

## Weighting

The weights are **hand-selected** and have NOT been empirically validated on any
held-out dataset. They reflect engineering intuition about which signals are
most informative for pronunciation reliability:

- Model agreement is weighted highest because disagreement is the strongest
  empirical signal of uncertainty.
- Sequence confidence is secondary because individual models can be overconfident.
- Locale support and stability are supporting signals.
- OOV signal is a last-resort detector for completely unknown inputs.

## Interpretation

The output `confidence` should be interpreted as:

> "A relative reliability indicator on [0, 1], where higher values suggest
> the prediction is more likely to be correct compared to lower values."

It should **NOT** be interpreted as:

> "The probability that this prediction is correct."

## Calibration

To convert this heuristic score into a calibrated probability, you would need
to:

1. Collect a large held-out dataset with verified ground truth
2. Compute the heuristic score for every example
3. Fit a calibration curve (e.g., Platt scaling, isotonic regression)
4. Apply the calibration function to transform scores to probabilities

This has **not** been done in the current implementation.

## Relation to Conformal Abstention

For statistically grounded abstention, use `ConformalAbstainer` instead. It uses
a held-out calibration set to set a nonconformity threshold with finite-sample
guarantees. The two systems are independent:

- `ReliabilityScorer`: heuristic, fast, no calibration set needed
- `ConformalAbstainer`: statistically guaranteed, requires calibration set