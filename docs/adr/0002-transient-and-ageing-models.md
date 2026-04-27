# ADR-0002: Transient thermal model and ageing kinetics

**Status:** Accepted
**Date:** 2026-04-27
**Author:** Lylliam Musoke

## Context

The Day-1 IEC 60287 steady-state solver answers "what is the conductor
temperature at this load, indefinitely sustained?" It cannot answer the two
questions a digital twin must address:

1. How does the conductor temperature respond to time-varying load over
   minutes, hours, and days?
2. How does that operating profile translate into accumulated insulation
   damage, and a remaining-useful-life (RUL) estimate?

These are the next two physics modules. They sit beneath the dataset
generator (which needs realistic transient temperature traces and failure-
time ground truth) and beneath the PINN training loss (which needs the
governing PDE forms whose discrete approximations these solvers represent).

## Decision

### Transient thermal model

Use a **single-node lumped capacitance** formulation (the IEC 60853-2
"simplified short-time rating" model). The cable cross-section is collapsed
to one effective thermal capacitance C_c with effective per-phase value
C_c / n. The surrounding soil is treated as a constant-temperature heat sink
at the ambient boundary, accessed through the IEC 60287 thermal resistances
(T1, T3, T4).

The governing ODE recovers IEC 60287 at steady state by construction:

    (C_c / n) dT_c/dt = ( I^2 R(T_c) + W_d ) - ( T_c - T_amb ) / R_total
        with R_total = T1 + n (1 + lambda_1) (T3 + T4)

equivalently expressed as a first-order lag toward the IEC steady-state
target temperature with time constant tau = (C_c / n) * R_total. The
implementation uses the lag form because it makes the steady-state
correctness obvious and avoids accidentally double-counting the n-phase
geometry.

### Ageing kinetics

Use **Crine's 2005 unified ageing model**:

    L(E, T) = L_ref * (E_ref / E)^n * exp( (Phi / k_B) * (1/T - 1/T_ref) )

This is the Arrhenius-Inverse-Power-Law form, the de facto standard in cable-
ageing literature (Mazzanti 2013, IEEE Std 1407-2007). Cumulative damage
under time-varying stress is computed by Miner-rule integration:

    D(t) = integral_0^t  d_tau / L( E(tau), T(tau) )

with failure at D = 1. The default (n, Phi, E_ref, T_ref, L_ref) tuple is
calibrated so that XLPE at design field (4 MV/m) and design conductor
temperature (90 degC) gives a 40-year mean life — matching the standard
distribution-cable design assumption.

## Alternatives considered

### Transient thermal

- **Full 1-D radial PDE solver on a cylindrical mesh.** Higher fidelity. Out
  of scope for v0.0.x because (a) the lumped model captures the dominant
  transient behaviour adequately for load-following studies, (b) the PDE
  form will return as the physics-residual constraint inside the PINN where
  it belongs, and (c) the cost of building, validating, and benchmarking a
  PDE solver against analytical solutions is several days of work that is
  better invested elsewhere on Day 2.
- **Two-node lumped (cable + soil annulus).** Captures multi-day soil-
  thermal-mass dynamics. Postponed: the soil time constant is on the order
  of weeks, so a single-day or single-week analysis horizon does not need
  it. Will revisit when the dataset generator simulates seasonal cycles.
- **CIGRE Working Group 09 (1972) "alpha-beta" formulation.** Mathematically
  equivalent to the chosen first-order-lag form, just with different
  notation. We use the lag form because it makes the IEC 60287 steady-state
  correctness immediately apparent in the code.

### Ageing kinetics

- **Pure Arrhenius (no field term).** Inadequate. Field stress is the
  dominant driver of XLPE life at 11 kV.
- **Pure inverse-power-law (no temperature term).** Used in some short-
  duration accelerated tests but unsuitable for in-service life prediction
  where conductor temperature varies by 50 K over an operating cycle.
- **Weibull-distributed time-to-failure.** Useful for population-level
  reliability statements but does not provide a physical rate function for
  damage accumulation under variable stress. Crine's model gives the rate;
  Weibull can be layered on top later for population reliability.
- **Cellular automaton or first-principles physics-of-failure.** Out of
  scope: requires a level of microstructural data that is not publicly
  available for distribution-grade XLPE.

## Consequences

**Positive**
- Both modules are short, self-contained, well-tested, and grounded in
  cited public literature.
- Together with the Day-1 steady-state solver they cover everything the
  synthetic dataset generator needs to produce realistic time-series and
  failure-time ground truth.
- The Crine module's RUL output is a directly demoable feature once
  integrated into GridOptima.

**Negative**
- The lumped transient does not resolve sub-cable spatial gradients (e.g.
  conductor versus screen during a fast transient). For step events shorter
  than ~1 minute this matters; for the load-following timescales this
  module targets, it does not.
- Crine parameters are calibrated to typical XLPE values from the
  literature, not to a specific manufacturer's product. Per-archetype
  calibration is a downstream task that requires DNO-partner data.

**Neutral**
- Both modules are pure-NumPy / SciPy. No PyTorch dependency yet. PINN
  training (Day 3+) layers on top by replacing the lumped ODE solve with a
  neural-network surrogate constrained to satisfy the same governing
  equations.

## References

- IEC 60853-2:2008 — Calculation of cyclic and emergency current rating of
  cables, Part 2: Cyclic and emergency rating of cables greater than 18/30 kV.
- IEC 60287-1-1:2006, IEC 60287-2-1:2015 — steady-state thermal model
  components used as building blocks here.
- Anders, G. J. (1998), Rating of Electric Power Cables, IEEE Press.
- Crine, J.-P. (2005), On the interpretation of some electrical-ageing and
  life-test results, IEEE Trans. Dielectr. Electr. Insul. 12(6), 1089-1107.
- Mazzanti, G. (2013), Life and reliability models for high-voltage DC
  extruded cables, IEEE Electr. Insul. Mag. 29(2), 36-44.
- IEEE Std 1407-2007 — Guide for Accelerated Aging Tests for Medium-Voltage
  Cables.
