# ADR-0006: Benchmark suite (GF-006 / GF-007)

**Status:** Accepted
**Date:** 2026-04-28
**Author:** Lilliane Linnet Musoke

## Context

GF-006 calls for a benchmark suite with five sealed-test tasks, and
GF-007 calls for three reference baselines spanning pure physics, ML,
and the GridForge PINN. The benchmark is the primary adoption surface
for GridForge — anyone with a competing technique can submit and be
ranked, and the reference baselines anchor the leaderboard.

## Decision

### Five tasks

| ID | Task | Headline metric | Better |
|---|---|---|---|
| T1 | 60-day damage-threshold-crossing prediction from a 30-day window | Brier | lower |
| T2 | RUL regression: time to damage threshold from full-history features | MAE [years] | lower |
| T3 | Per-day anomaly detection (impulse hours + thermal-ageing days as positives) | Precision @ recall=0.9 | higher |
| T4 | Conductor-temperature prediction from (I, ambient, soil_rho_t) at each hour | RMSE [degC] | lower |
| T5 | Counterfactual attribution of damage fractions to load / ambient / condition-mode driver | MAE on fractions | lower |

Task definitions are kept lightweight (just metadata) so they can be
referenced from external papers and submissions without requiring the
rest of the benchmark infrastructure.

### Three reference baselines

1. **`IECOracleBaseline`** — pure physics, no ML. Uses IEC 60287 +
   Crine kinetics directly. Implementation detail: for T4 it returns
   the dataset's own `conductor_C` (i.e. the lumped-transient ground
   truth) — see "Known limitation" below.
2. **`GradientBoostedBaseline`** — sklearn `HistGradientBoostingRegressor`
   on engineered features (current, ambient, soil_rho_t). Falls back to
   a training-mean predictor if sklearn is not installed.
3. **`PINNBaseline`** — loads the Day-4 trained surrogate and uses it
   for T4. Other tasks fall back to the IEC oracle's downstream logic
   for now (rate extrapolation, z-score anomaly, etc).

### Stdlib-only loader

`gridforge.bench.loader.load_mini_dataset()` reads the artefact layout
produced by `assemble_dataset` using only `csv` and `numpy` — no
pandas. This keeps the bench module dependency-light and lets external
users load the dataset without extra installs.

### Numpy-only metrics

`gridforge.bench.metrics` implements Brier, AUC-PR, MAE, RMSE,
precision-at-recall, and quantile loss directly in numpy. Each implementation
is short, exact, and audited by the test suite (14 tests). No sklearn
dependency for evaluation.

### Runner

`run_benchmark(view, baselines, tasks)` walks the cartesian product,
calls each baseline's `predict()` once per task, evaluates against
ground-truth labels constructed from the test split, and returns a list
of `LeaderboardEntry` records. The demo script
`scripts/06_run_benchmark.py` prints a table and writes
`scripts/output/06_benchmark/leaderboard.csv`.

## Known limitation: T4 ground truth

The mini synthetic dataset's `conductor_C` column is the **lumped
transient** output of the cable-year simulator — it includes the
first-order thermal lag (tau ≈ 1 hour) integrated against time-varying
load and ambient. A pure steady-state surrogate (the PINN, or
gradient-boosted features-only models) cannot reproduce this exactly
because the steady-state response and the transient response differ by
a lag-dependent margin (typically 5–15 degC at peak-load transitions).

This means:

- `IECOracleBaseline.predict_t4` returns the dataset truth itself
  (representing "physics + perfect time history") and scores ~0 RMSE.
  This is an **upper-bound reference**, not a fair physics-only
  comparison.
- `PINNBaseline.predict_t4` and `GradientBoostedBaseline.predict_t4`
  predict steady-state from instantaneous inputs and score ~10 degC
  RMSE on the lumped-transient ground truth. They are on equal footing
  with each other; the physics-informed advantage of the PINN over the
  GBT shows in the **secondary** MAE metric (5.84 vs 8.77 in the demo
  run), where the PINN is consistently closer to the truth's central
  tendency.

A cleaner T4 framing — adding a steady-state `conductor_C_steady`
column to the dataset, or a separate steady-state-only test split — is
on the post-paper roadmap. For the v0.0.x benchmark, the current
formulation is documented and reproducible.

## Alternatives considered

- **Add pandas to the runtime dependency list.** Rejected. Stdlib csv
  is sufficient for the dataset's row-oriented layout, and pandas
  would add ~30 MB to a wheel that's currently ~50 KB.

- **Use sklearn for metrics.** Rejected. AUC-PR, Brier, etc. are short
  enough to write in numpy directly; the implementations are auditable
  and don't drag in a heavy dependency.

- **Vectorise solve_steady_state for honest IEC T4 evaluation.**
  Tracked as future work — would require a refactor of
  `gridforge.physics.thermal.solve_steady_state` to accept array inputs.

- **Per-cable train/val/test split.** Rejected. The current SHA-256
  cable-id split keeps cables atomic across telemetry streams — once a
  cable lands in test, its 5-year trace is sealed.

## Consequences

**Positive**
- 22 new tests (14 metric + 5 loader + 3 runner) all passing.
- Leaderboard runs end-to-end against the 64-cable mini dataset in
  ~5 seconds.
- Three reference baselines installed; PINN auto-loads from the Day-4
  state dict if present.
- Externally submittable infrastructure: a third party with their own
  technique can subclass `Baseline`, implement `predict_t*`, pass it
  to `run_benchmark`, and have their score evaluated against the same
  sealed labels as the references.

**Negative**
- T4 framing is currently asymmetric (see "Known limitation"). The
  ADR is the authoritative documentation; users should read it before
  interpreting the leaderboard.
- T1 / T3 / T5 baselines for `GradientBoostedBaseline` and `PINNBaseline`
  fall back to the IEC oracle's downstream logic. Differentiated
  implementations (e.g. a GBT on the 30-day window for T1) are on the
  post-paper roadmap.

**Neutral**
- The bench module reuses the data loader/dataset machinery from Day 3
  unchanged. Day 4 PINN is consumed read-only (state-dict load).
  No refactor needed in upstream modules.

## References

- Internal: ADR-0001 (physics foundation), ADR-0002 (transient + ageing),
  ADR-0003 (synthetic dataset), ADR-0004 (PINN architecture),
  ADR-0005 (GridOptima integration).
- Davis & Goadrich (2006), *The relationship between precision-recall
  and ROC curves*, ICML — basis for AUC-PR computation.
- Brier (1950), *Verification of forecasts expressed in terms of
  probability*, Monthly Weather Review.
