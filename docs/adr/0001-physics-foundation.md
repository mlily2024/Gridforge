# ADR-0001: Start with an IEC 60287 oracle, not a PINN

**Status:** Accepted
**Date:** 2026-04-27
**Author:** Lylliam Musoke

## Context

GridForge is intended to ship a physics-informed neural network (PINN) trained
to satisfy the coupled heat, electric-field, and dielectric-ageing equations
that govern UK 11 kV underground distribution cables (see GF-001..GF-003 in the
consolidated requirements document).

A PINN is only as trustworthy as the physics it is checked against. Without a
reliable analytical or semi-analytical reference solution, a network can
appear to converge while violating energy balance, because the data-fitting
loss happily drives down on a bad solution. Reviewers — internal users,
academic peers, and downstream adopters — will reasonably ask: "where is the
ground truth that says this network is right?"

The right ground truth for steady-state cable thermal behaviour is IEC 60287.
It is the international standard for cable rating calculations, has been
benchmarked against decades of operational experience, and has closed-form
expressions for every thermal-resistance component for canonical cable
geometries.

## Decision

Implement an IEC 60287-1-1 / 60287-2-1 steady-state thermal solver as the
**first** module in `gridforge.physics`. No neural network code goes in until
the IEC solver:

- Reproduces hand-calculable thermal-resistance components (T1, T3, T4) to
  numerical precision.
- Solves the steady-state heat balance with documented convergence behaviour.
- Passes a verification suite covering boundary cases and monotone-scaling
  properties.
- Is wired into a single canonical UK 11 kV 240 mm^2 XLPE 3-core archetype
  whose every parameter is sourced to a public standard (BS EN 60228, BS 7870,
  IEC 60287-2-1) or a manufacturer datasheet conforming to those standards.

Subsequent PINN training will use this solver to:

1. Generate physics-residual training points (the network must satisfy the
   PDE form whose closed-form steady-state solution the IEC solver provides
   for boundary checks).
2. Provide a closed-form validation set the PINN can be measured against
   independently of the synthetic monitoring data.

## Alternatives considered

- **Skip IEC 60287, use only the heat PDE residual at training time.** Rejected.
  The PDE residual on its own does not pin the steady-state operating point;
  many trivial solutions satisfy the residual locally. A closed-form steady
  state from the same physics provides the necessary anchor.

- **Use a finite-element thermal solver as the oracle.** Rejected for v0.0.1.
  An FE solver is heavier infrastructure than the project needs at this stage,
  and IEC 60287 already encapsulates the steady-state result for canonical
  geometries with no numerical error. FE is a sensible Phase 2 addition for
  non-canonical geometries (cable groups, ducts, bends).

- **Use a vendor or DNO rating table directly as the oracle.** Rejected.
  Tabulated ratings depend on assumptions (sheath bonding scheme, soil
  moisture state, group derating) that vary between vendors and standards.
  Asserting agreement with one tabulated number at one set of assumptions
  is a calibration claim, not a verification of physics.

## Consequences

**Positive**
- Every later artefact (PINN, dataset generator, virtual-sensor inference)
  has a documented oracle to be checked against.
- The IEC solver is itself useful: rating-table reconstruction, what-if
  analyses, sensitivity studies.
- Public, citable references — IEC 60287 is the recognised standard.

**Negative**
- Adds a module of physics code before any ML lands. Mitigated by keeping
  the implementation small (~250 lines) and well-tested.
- The IEC formulation is steady state only. Transient behaviour requires a
  separate solver path. Acceptable for v0.0.1; transient handling is a
  later concern (GridForge transient roadmap, post-paper).

**Neutral**
- Material parameters (rho_T for soil, tan_delta for XLPE, etc.) are
  configurable, not baked in. Future work can swap in alternative
  reference values or operate over a calibration distribution.

## References

- IEC 60287-1-1:2006 — Calculation of the current rating, Part 1: General.
- IEC 60287-2-1:2015 — Thermal resistance, Part 2: Cables in air, buried, etc.
- IEC 60228:2004 — Conductors of insulated cables.
- BS 7870 — UK distribution cables, multiple parts.
- Raissi, Perdikaris, Karniadakis (2019), *Physics-informed neural networks:
  a deep-learning framework for solving forward and inverse problems involving
  nonlinear partial differential equations*, J. Comput. Phys. 378, 686–707.
- Cuomo et al. (2022), *Scientific machine learning through physics-informed
  neural networks: where we are and what's next*, J. Sci. Comput. 92, 88.
