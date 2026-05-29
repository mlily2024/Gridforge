# ADR-0003: Synthetic dataset design

**Status:** Accepted
**Date:** 2026-04-27
**Author:** Lilliane Linnet Musoke

## Context

GF-004 in the consolidated requirements calls for a calibrated synthetic
dataset of UK 11 kV underground distribution cables, large enough to train
a PINN and benchmark downstream tasks (failure prediction, RUL regression,
anomaly detection, virtual sensors, counterfactual attribution). No public
real-data benchmark exists for this domain — DNOs cannot publish operational
data due to commercial sensitivity and security constraints.

The dataset must:

- Cover ≥4 archetypes and ≥4 failure modes
- Provide hourly-resolution telemetry across multi-year horizons
- Be reproducible from a small set of seeds
- Be calibrated to public physical constants (no employer data)
- Ship deterministic train / val / test splits with sealed test labels
- Be small enough to publish on Zenodo (≤10 GB) and reproduce locally

This ADR records the design choices that turn the GridForge physics
primitives into a dataset generator that satisfies these constraints.

## Decision

### Composition

A cable-year is a tuple of:

  - **archetype**     (geometry + materials, one of four canonical UK 11 kV
                       configurations registered in
                       `gridforge.physics.cable_archetype.ARCHETYPES`)
  - **installation**  (burial depth + soil thermal resistivity + ambient mean)
  - **load profile**  (residential / commercial / industrial / mixed)
  - **weather seed**  (per-cable noise + climatology)
  - **failure mode**  (healthy / water-ingress / thermal-ageing / accelerated-dielectric)
  - **duration**      (default 1 year, configurable up to multi-year — the
                       mini demo uses 5 years to surface failure-mode
                       differentiation)

Every output cable-year is fully reproducible from this 6-tuple plus a
deterministic random seed.

The four registered archetypes are:

| Name | Description |
|---|---|
| `11kV_240mm2_Cu_XLPE_3c` | Modern 240 mm² Cu XLPE 3-core — workhorse of UK distribution |
| `11kV_95mm2_Cu_XLPE_3c`  | Smaller 95 mm² Cu XLPE 3-core — radial feeders to ~600 kVA secondaries |
| `11kV_300mm2_Cu_XLPE_1c` | Larger 300 mm² Cu XLPE single-core — primary trefoil installations |
| `11kV_240mm2_Cu_PILC_3c` | Older 240 mm² Cu PILC 3-core — paper-insulated lead-covered, still in service |

### Modules

| Module | Role |
|---|---|
| `gridforge.data.load_profiles` | Four load-shape functions with diurnal, weekly, and seasonal modulation |
| `gridforge.data.weather` | UK soil-temperature climatology at 0.8 m + soil-moisture index |
| `gridforge.data.failure_modes` | Four failure-mode injectors (field multiplier + temperature offset + PD multiplier) |
| `gridforge.data.cable_year` | Hourly forward integration of the lumped first-order thermal lag, Crine damage integration, failure-time detection |
| `gridforge.data.dataset` | Multi-cable-year assembler with deterministic split and on-disk layout |

### Time stepping

Hourly resolution is the right granularity:

- The thermal time constant of a typical 11 kV cable is about an hour.
  Hourly steps capture the dominant transient.
- Crine damage integrates well at hourly resolution (the inverse-power-law
  dependence on field is smooth on this timescale).
- An hourly cable-year produces 8,766 samples — large enough for ML
  training, small enough to fit in a few hundred kilobytes per cable.

A second-order RK2 (midpoint) update is used for the thermal ODE. RK4 is
overkill — the lag is first-order linear in T and the non-linearity in
R(T_c) is mild over the operating range.

### Output format

CSV with optional gzip compression, not Parquet. Reasoning:

- Parquet requires `pyarrow` (~50 MB extra dependency).
- A 12-cable mini dataset at hourly resolution fits in roughly 1 MB as CSV.
- A 1000-cable production dataset is on the order of 100 MB as CSV — well
  inside the Zenodo deposit limit and easily compressible to <40 MB.
- Conversion to Parquet is a one-line dataframe round-trip if a downstream
  user needs it.

The one-row-per-hour-per-cable layout matches the task harness format used
by the Day-7 benchmark suite without any further transformation.

### Splits

Train / val / test ratios default to 70 / 15 / 15. Split assignment is by
SHA-256 of the cable ID modulo the configured ratios. This guarantees:

- Determinism: regenerating the dataset with the same cable IDs yields the
  same split assignment.
- Stability under expansion: adding new cable IDs does not move existing
  IDs between splits.
- Sealed test labels: ground-truth failure times are stored in a separate
  file from the telemetry, so a benchmark submission can withhold the test
  labels at evaluation time.

### Failure-mode parameterisation

| Mode | Mechanism | Knobs |
|---|---|---|
| `healthy` | Nominal Crine kinetics | none |
| `water_ingress` | Linear-in-time field rise from `onset_year` to `saturation_year`; quadratic PD-rate growth | onset, saturation, max boost |
| `thermal_ageing` | Constant temperature offset on top of the transient solution | offset_C |
| `accelerated_dielectric` | Discrete impulse train with configured per-year rate; field boosts during impulses | per-year rate, boost factor, duration |

These four cover the dominant non-thermal failure mechanisms documented in
Mazzanti & Marzinotto (2013) for medium-voltage XLPE distribution cables.

### Calibration sources

Every numerical constant is sourced to a public reference:

- Cable geometry and materials — BS EN 60228, BS 7870, IEC 60287-2-1
  (already in `cable_archetype.py`)
- Soil temperature climatology — Met Office UK 1991-2020 reference period
- Soil moisture climatology — UK precipitation pattern, public Met Office data
- Crine ageing parameters — Crine 2005, IEEE Std 1407-2007
- Failure-mode taxonomy — Mazzanti & Marzinotto 2013

No employer data, no NDA-held data, no commercially confidential parameters.

## Alternatives considered

- **Generate the dataset at hourly resolution but with only daily damage
  accumulation.** Rejected: at hourly resolution Crine damage integration
  is exact; coarsening to daily averages introduces a known bias near
  high-temperature impulse events.

- **Use a pre-trained ML model to generate synthetic telemetry.** Rejected:
  introduces ML lineage in the dataset itself, undermining the "synthetic
  data calibrated to physics, not to a model" claim.

- **Skip the soil-moisture feature in v0.0.x.** Rejected: even though the
  current thermal model holds soil resistivity constant, recording soil
  moisture in the telemetry stream lets later versions of the PINN learn
  the coupling without re-generating the dataset.

- **Tie failure-mode parameters to Weibull failure-time distributions
  directly.** Rejected for now: keeping the failure-mode injectors physics-
  inspired (field stress, temperature, impulse train) means the resulting
  failure-time distribution is an emergent quantity, not a fitted curve.
  This is more defensible for the paper.

## Consequences

**Positive**
- A complete dataset-generation pipeline is now in place.
- The mini dataset (12 cables × 1 year) reproduces in under a minute on
  modest hardware, suitable for CI smoke tests.
- The full 1000-cable-year production dataset is the same code path with a
  larger spec list — a single configuration change.
- Every output is reproducible from a small set of seeds.

**Negative**
- CSV output bloats compared to Parquet for the largest deposits; a
  follow-up commit can switch to Parquet once `pyarrow` is added.

**Neutral**
- The cable-year simulator is independent of `simulate_transient` from the
  physics package. They share the same lumped-model mathematics but the
  cable-year version takes time-varying ambient and runs an explicit RK2.
  This avoids changes to `simulate_transient` (no regression risk on the
  existing 50 tests) at the cost of a small amount of code duplication.

## References

- Mazzanti, G., Marzinotto, M. (2013), *Extruded Cables for High-Voltage
  Direct-Current Transmission*, IEEE Press — Chapter 5 on degradation modes.
- Crine, J.-P. (2005), *On the interpretation of some electrical-ageing
  and life-test results*, IEEE Trans. Dielectr. Electr. Insul. 12(6).
- IEEE Std 1407-2007 — *Guide for Accelerated Aging Tests for Medium-Voltage
  Cables*.
- Met Office UK soil-temperature climatology, 1991-2020 reference period
  (publicly available via the Met Office DataPoint API).
- Open Networks Project secondary-substation studies — typical UK demand
  shapes.
- Internal: ADR-0001 (physics foundation), ADR-0002 (transient + ageing
  models).
