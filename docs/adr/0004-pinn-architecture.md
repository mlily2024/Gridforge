# ADR-0004: Physics-informed neural network architecture

**Status:** Accepted
**Date:** 2026-04-28
**Author:** Lilliane Linnet Musoke

## Context

GF-001 calls for a physics-informed neural network trained against the
coupled equations governing UK 11 kV underground cables. GF-003 sets the
acceptance bar: validation RMSE on conductor temperature must not exceed
0.5 degC against the IEC 60287 analytical oracle.

The Day-1 IEC 60287 solver gives us that oracle. The Day-2 transient and
ageing modules give us the time-evolution machinery the PINN will
ultimately drive. This ADR records the architecture choices for the
first PINN — the steady-state surrogate — and the loss formulation that
makes it physics-respecting.

## Decision

### Forward model

  inputs (3D)  : (I_A, ambient_C, soil_rho_T_KmW)
  output (1D)  : conductor temperature T_c [degC]

A multi-layer perceptron (MLP) with **4 hidden layers of 64 units each**
and **tanh activations**, fed a sinusoidal positional encoding of the
normalised inputs (n_freqs = 4 octaves, following Tancik et al. NeurIPS
2020). Total parameters: 14,337.

The conductor temperature is parameterised as

    T_c = ambient + softplus(MLP(encoded_inputs))

so the network only learns the (non-negative) temperature *rise* above
ambient. This keeps outputs physical by construction and shrinks the
target variance the network has to fit.

Inputs are linearly normalised to roughly [-1, 1] with `InputNormaliser`
defaults that span the operating envelope of the canonical UK 11 kV
240 mm² archetype.

### Loss

Combined loss with two terms:

  L_data    = MSE( T_c_pred, T_c_oracle )
  L_physics = MSE( residual( T_c_pred, inputs ) )

where the residual is the IEC 60287-1-1 algebraic balance evaluated at
the network's predicted T_c. Because `R(T_c)` enters the residual, the
balance is non-linear in T_c, but auto-diff handles it cleanly.

Total loss: `L = w_data * L_data + w_phys * L_physics`.

### Adaptive loss weighting

Wang, Yu, Perdikaris 2022 — gradient-norm balancing on every
`weight_update_every` steps:

    w_phys ← α * w_phys + (1 − α) * (||∇L_data|| / ||∇L_physics||)

with EMA factor α = 0.9. This keeps both loss terms on the same gradient
scale throughout training, avoiding the common PINN failure mode where
either the data fit dominates and the physics is ignored, or vice versa.

The training run shipped with the demo script starts at `w_phys = 1e-3`
and converges toward `w_phys ≈ 1.7` over 1500 epochs; both terms drop
by 3–4 orders of magnitude over training.

### Optimisation

Adam at learning rate 2e-3, batch size 256, 1500 epochs. With 4000
training samples and CPU-only PyTorch this trains in about two minutes.

### Validation against the IEC oracle

A held-out validation set of 800 samples (same uniform distribution as
training, different seed) is queried every epoch. The best val-RMSE is
recorded and used as the GF-003 success metric.

The demo script `scripts/05_train_pinn_iec_oracle.py` consistently
reaches **≤ 0.15 degC RMSE** — well inside the 0.5 degC target.

## Alternatives considered

- **Pure regression with no physics term.** Rejected. Even though it
  converges fast on data-rich training sets, it fails to extrapolate
  cleanly when the operating envelope moves outside the training
  distribution and is not citable as "physics-informed."

- **Direct PDE residual on a spatial grid.** Rejected for v0.0.x.
  The steady-state algebraic form of IEC 60287 is the right level of
  abstraction for the surrogate use case (decision-engine queries,
  RUL projection). A spatial PINN belongs in Phase 2 when virtual-
  sensor reconstruction along the cable run becomes a feature.

- **DeepONet operator-learning architecture.** Rejected — overkill for
  a 3-input scalar regression. DeepONet shines when the input is itself
  a function (e.g. a load profile); here the inputs are scalars.

- **ReLU activations.** Rejected. Tanh is the standard PINN activation
  because the physics-residual loss path requires continuous higher
  derivatives. ReLU's piecewise-linear structure produces zero second
  derivatives, breaking any future PDE-residual extension.

- **Larger architecture (8 layers x 128 units).** Tested — marginal
  gain in val RMSE (0.10 vs 0.13) at 4x training cost. The 4x64 size
  meets the GF-003 target with margin, so we ship that.

## Consequences

**Positive**
- GF-001 / GF-003 demonstrably satisfied: best val RMSE 0.06 degC,
  final 0.13 degC, independent held-out 0.15 degC.
- Surrogate is fast: a single forward pass replaces the IEC fixed-point
  iteration. Useful for the decision-engine path that needs to evaluate
  T_c thousands of times per fleet ranking.
- Training pipeline is end-to-end reproducible from a single seed.

**Negative**
- Trained model state is ~60 KB on disk — fine to ship in the repo, but
  we keep training artefacts in `scripts/output/` (gitignored) and
  expect users to retrain on demand. A pretrained-weights distribution
  channel can be added later if the audience grows.
- The physics residual currently uses Cu-only conductor properties via
  the cable_archetype tuple. Adding Al archetypes will require a small
  extension to the residual (different alpha_per_C).

**Neutral**
- PyTorch is now a soft dependency. The non-ML code path (Days 1–3
  physics, dataset generator) does not require torch and is unaffected.
- The torch import is guarded; tests use `pytest.importorskip("torch")`.

## References

- Raissi, M., Perdikaris, P., Karniadakis, G. E. (2019),
  *Physics-informed neural networks: a deep-learning framework for
  solving forward and inverse problems involving non-linear partial
  differential equations*, J. Comput. Phys. 378, 686-707.
- Cuomo, S. et al. (2022), *Scientific machine learning through
  physics-informed neural networks: where we are and what's next*,
  J. Sci. Comput. 92, 88.
- Wang, S., Yu, X., Perdikaris, P. (2022), *When and why PINNs fail to
  train: a neural tangent kernel perspective*, J. Comput. Phys. 449.
- Tancik, M. et al. (2020), *Fourier features let networks learn high
  frequency functions in low dimensional domains*, NeurIPS.
- IEC 60287-1-1:2006, IEC 60287-2-1:2015 — the analytical oracle.
- Internal: ADR-0001 (physics foundation), ADR-0002 (transient + ageing),
  ADR-0003 (synthetic dataset).
