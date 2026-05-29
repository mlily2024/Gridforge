# ADR-0008: Uncertainty quantification via a deep ensemble

**Status:** Accepted
**Date:** 2026-05-29
**Relates to:** ADR-0004 (PINN architecture), ADR-0001 (IEC oracle)

## Context

The IEC-surrogate PINN (ADR-0004) returns a **point estimate** of conductor
temperature. Operationally that is not enough: an asset manager acting on a
70 °C prediction needs to know whether the model means "70 ± 2 °C" (actionable)
or "70 ± 15 °C" (don't trust it). Remaining-useful-life and CI/CML penalty
figures downstream inherit that uncertainty, so a temperature confidence
interval propagates directly into decision confidence.

We need an uncertainty estimate that is:

- **principled** (reflects genuine model uncertainty, not a fudge factor),
- **cheap to add** (no architecture rewrite, no new physics), and
- **reproducible and torch-optional**, consistent with the rest of GridForge.

## Decision

Quantify **epistemic** uncertainty with a **deep ensemble**
(Lakshminarayanan et al., 2017): train K independent PINNs with different
weight initialisations *and* different training-data samples, and treat the
spread of their predictions as uncertainty. For a query point the ensemble
reports the mean and a Gaussian interval `mean ± z·σ` (z = 1.96 → ~95%), where
σ is the inter-member standard deviation.

Implementation:

- `gridforge.models.ensemble.DeepEnsemblePINN` — holds K trained members;
  `predict()` returns `(mean, std, lower, upper)`; `__call__` returns the mean
  so it is drop-in compatible with `TrainedPINNSurrogate`; `save()`/`load()`
  persist the member state-dicts.
- `gridforge.training.ensemble.train_ensemble()` — trains K members by reusing
  the existing `train_pinn` loop (distinct init + data seeds per member); a
  `coverage()` helper measures empirical CI calibration on held-out data.

No change to the network, the loss, or the IEC oracle.

## Consequences

- Predictions become **actionable** (confidence intervals), and the same
  intervals can be surfaced through the decision stack (the C1 API endpoint).
- A one-member ensemble degrades exactly to the existing point estimate.
- Training cost scales linearly in K (members are independent and trivially
  parallelisable).
- Provides the foundation for the planned Bayesian-PINN / conformal work
  (B1/B3) and gives a calibration baseline to compare them against.

## Alternatives considered

- **Bayesian PINN (variational / SVGD).** More principled posteriors but a
  substantial architecture + training rewrite. Deferred (B1) — the deep
  ensemble is the pragmatic first step and a fair calibration baseline.
- **MC-dropout.** Cheaper, but requires dropout layers and dropout-at-inference,
  changing the network; ensembles are generally better-calibrated.
- **Conformal prediction.** Distribution-free guarantees, but wraps a base
  predictor — complementary, planned later (B3), not a replacement.
