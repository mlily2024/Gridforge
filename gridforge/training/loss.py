"""
Combined data + physics loss for the IEC 60287 surrogate PINN.

The data term is straight MSE between predicted and oracle conductor
temperatures. The physics term is the residual of the IEC 60287-1-1
algebraic balance:

    ( T_c - T_amb )
        - I^2 R(T_c) [ T1 + n (1 + lambda_1) (T3 + T4(rho_t)) ]
        - W_d        [ 0.5 T1 + n (1 + lambda_1) (T3 + T4(rho_t)) ]
    = 0

Inputs to this residual depend only on the (I, ambient, rho_t) feature
vector and the cable+install constants. The residual is therefore
differentiable w.r.t. the network's predicted T_c, so back-propagation
through the physics term is well-defined.

Adaptive loss weighting follows Wang, Yu, Perdikaris 2022 — at each step
the physics weight is rescaled by the ratio of mean gradient magnitudes
between the data and physics terms, keeping both terms contributing on
the same scale throughout training. This avoids the common PINN failure
mode where one loss dominates and the other gets ignored.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import log, pi, sqrt

try:
    import torch

    _TORCH_AVAILABLE = True
except ImportError:  # pragma: no cover
    torch = None  # type: ignore[assignment]
    _TORCH_AVAILABLE = False

from ..physics.thermal import (
    CableGeometry,
    CableMaterials,
    InstallationConditions,
    dielectric_loss_per_phase,
    thermal_resistance_T1,
    thermal_resistance_T3,
)


def _require_torch() -> None:
    if not _TORCH_AVAILABLE:
        raise ImportError(
            "PyTorch is required for gridforge.training.loss. "
            "Install with: pip install 'gridforge[ml]'"
        )


@dataclass(frozen=True)
class PhysicsConstants:
    """Pre-computed cable+install constants used inside the physics residual."""

    T1_KmW: float
    T3_KmW: float
    T4_geometric_factor: float  # the (1/2pi) * arccosh(2L/D_e) part
    n_conductors: int
    sheath_loss_factor: float
    R_dc_20C_ohm_per_m: float
    alpha_per_C: float
    R_ac_dc_ratio: float
    W_d_W_per_m: float


def precompute_physics_constants(
    geom: CableGeometry,
    mat: CableMaterials,
    install: InstallationConditions,
    line_voltage_V_rms: float = 11_000.0,
    sheath_loss_factor: float = 0.05,
) -> PhysicsConstants:
    """Cache the cable+install constants the physics residual needs each step."""
    voltage_phase = line_voltage_V_rms / sqrt(3.0)
    W_d = dielectric_loss_per_phase(voltage_phase, mat)
    T1 = thermal_resistance_T1(geom, mat)
    T3 = thermal_resistance_T3(geom, mat)
    # T4 is rho_t * geometric factor; bake the geometric part now so the
    # network can be queried at any rho_t at training time without a graph
    # rebuild.
    D_e_m = geom.D_e_mm * 1.0e-3
    two_u = 2.0 * install.burial_depth_m / D_e_m
    if two_u < 1.0:
        raise ValueError("burial depth too shallow for IEC formula")
    geometric = (1.0 / (2.0 * pi)) * log(two_u + sqrt(two_u * two_u - 1.0))
    return PhysicsConstants(
        T1_KmW=T1,
        T3_KmW=T3,
        T4_geometric_factor=geometric,
        n_conductors=geom.n_conductors,
        sheath_loss_factor=sheath_loss_factor,
        R_dc_20C_ohm_per_m=mat.R_dc_20C_ohm_per_m,
        alpha_per_C=mat.alpha_per_C,
        R_ac_dc_ratio=mat.R_ac_dc_ratio,
        W_d_W_per_m=W_d,
    )


def physics_residual(
    T_c_pred: "torch.Tensor",
    inputs: "torch.Tensor",
    constants: PhysicsConstants,
) -> "torch.Tensor":
    """IEC 60287 algebraic balance residual at each (I, ambient, rho_t).

    Args:
        T_c_pred: Predicted conductor temperature, shape (B, 1).
        inputs:   Feature vector (I_A, ambient_C, rho_t_KmW), shape (B, 3).
        constants: Pre-computed cable + install constants.

    Returns:
        Residual tensor, shape (B, 1).
    """
    _require_torch()
    I = inputs[..., 0:1]
    amb = inputs[..., 1:2]
    rho_t = inputs[..., 2:3]

    # AC resistance at predicted T_c (Cu temperature correction + skin factor)
    R_dc = constants.R_dc_20C_ohm_per_m * (1.0 + constants.alpha_per_C * (T_c_pred - 20.0))
    R = R_dc * constants.R_ac_dc_ratio
    I2R = I * I * R

    # T4 = rho_t * geometric factor
    T4 = rho_t * constants.T4_geometric_factor
    n = constants.n_conductors
    lam1 = constants.sheath_loss_factor
    R_total = constants.T1_KmW + n * (1.0 + lam1) * (constants.T3_KmW + T4)
    W_d = constants.W_d_W_per_m
    layer_factor_W_d = 0.5 * constants.T1_KmW + n * (1.0 + lam1) * (constants.T3_KmW + T4)

    delta_T_target = I2R * R_total + W_d * layer_factor_W_d
    return (T_c_pred - amb) - delta_T_target


def combined_loss(
    T_c_pred: "torch.Tensor",
    T_c_target: "torch.Tensor",
    inputs: "torch.Tensor",
    constants: PhysicsConstants,
    w_data: float = 1.0,
    w_phys: float = 1.0,
) -> dict[str, "torch.Tensor"]:
    """Compute data + physics MSE losses (and total).

    Returns a dict with `data`, `physics`, `total`, all 0-dim tensors that
    require_grad through the network.
    """
    _require_torch()
    err_data = T_c_pred - T_c_target
    L_data = (err_data * err_data).mean()

    res = physics_residual(T_c_pred, inputs, constants)
    L_phys = (res * res).mean()

    L_total = w_data * L_data + w_phys * L_phys
    return {"data": L_data, "physics": L_phys, "total": L_total}


def adaptive_loss_weights(
    L_data: "torch.Tensor",
    L_phys: "torch.Tensor",
    parameters,
    alpha: float = 0.9,
    prev_w_phys: float = 1.0,
) -> float:
    """Wang et al. 2022 gradient-norm balancing for the physics weight.

    Returns the new physics-loss weight. Data weight is fixed at 1; only
    the physics weight is adapted, with an exponential moving average of
    the gradient-norm ratio so the weight is stable across mini-batches.

    The data-term gradient norm is |dL_data / dtheta|; the physics-term
    gradient norm is |dL_phys / dtheta|. We push w_phys toward the ratio
    that equalises the two contributions, keeping them on the same scale.
    """
    _require_torch()

    # Materialise the parameter iterable once — generators get exhausted on
    # first traversal, leaving the second `torch.autograd.grad` with an
    # empty inputs list.
    param_list = list(parameters)
    if not param_list:
        return prev_w_phys

    grad_data = torch.autograd.grad(L_data, param_list, retain_graph=True, allow_unused=True)
    grad_phys = torch.autograd.grad(L_phys, param_list, retain_graph=True, allow_unused=True)

    def _l2(grads) -> float:
        s = 0.0
        for g in grads:
            if g is not None:
                s += float(g.detach().pow(2).sum().item())
        return s**0.5

    norm_data = _l2(grad_data)
    norm_phys = _l2(grad_phys)
    if norm_phys < 1e-12:
        return prev_w_phys
    target = norm_data / norm_phys
    return float(alpha * prev_w_phys + (1.0 - alpha) * target)
