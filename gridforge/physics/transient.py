"""
Transient thermal model — single-node lumped formulation.

Distribution-cable rating studies use IEC 60853-2 to compute short-time
emergency ratings. The standard simplification is a lumped single-node
model in which the entire cable cross-section is represented by one
effective thermal capacitance, and the surrounding soil is treated as a
heat sink at the ambient temperature reached through the steady-state
thermal resistances T1, T3 and T4.

The governing ODE is

    C_c dT_c/dt = Q(t, T_c) - (T_c - T_amb) / R_total

with

    Q(t, T_c)   = ( I(t)^2 R(T_c) + W_d ) * n
    R_total     = T1 + n (1 + lambda_1) (T3 + T4)
    R(T_c)      = a.c. conductor resistance at T_c

This recovers the IEC 60287 steady state when dT_c/dt = 0, and gives the
correct first-order time response (time constant tau = C_c * R_total) for
load-following, emergency-overload, and diurnal-cycle studies.

Soil is held at the constant ambient. The implicit assumption is that the
soil time constant (weeks-months) is much longer than the analysis horizon
(hours-days), so soil response can be treated as quasi-static. For multi-
day analyses with seasonal soil changes, a separate soil-thermal-mass node
should be added; that is on the post-paper roadmap.

References:
    IEC 60853-2:2008 — Calculation of cyclic and emergency current rating
        of cables, Part 2: Cyclic and emergency rating of cables greater
        than 18/30 kV.
    Anders, G. J. (1998), Rating of Electric Power Cables, IEEE Press.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Sequence

import numpy as np
from scipy.integrate import solve_ivp

from .thermal import (
    CableGeometry,
    CableMaterials,
    InstallationConditions,
    ac_resistance_at_temp,
    dielectric_loss_per_phase,
    thermal_resistance_T1,
    thermal_resistance_T3,
    thermal_resistance_T4,
)

# ---------------------------------------------------------------------------
# Cable thermal-capacitance estimation
# ---------------------------------------------------------------------------
#
# Effective per-unit-length thermal capacitance of the cable cross-section.
# For a 3-core XLPE distribution cable, the dominant contributors are:
#
#   conductor (Cu)        rho 8960 kg/m^3, cp 385  J/(kg K)
#   insulation (XLPE)     rho 920  kg/m^3, cp 2300 J/(kg K)
#   jacket (HDPE)         rho 950  kg/m^3, cp 1900 J/(kg K)
#
# The screen and any fillers are smaller contributions and lumped into the
# insulation term for this simplified model.

CU_DENSITY_KG_PER_M3: float = 8960.0
CU_CP_J_PER_KG_K: float = 385.0
XLPE_DENSITY_KG_PER_M3: float = 920.0
XLPE_CP_J_PER_KG_K: float = 2300.0
HDPE_DENSITY_KG_PER_M3: float = 950.0
HDPE_CP_J_PER_KG_K: float = 1900.0


def cable_thermal_capacitance(geom: CableGeometry) -> float:
    """Per-unit-length effective thermal capacitance of the cable [J/(K m)].

    Sum of: 3 x conductor + 3 x insulation annulus + jacket annulus.
    """
    n = geom.n_conductors
    r_c_m = geom.d_c_mm * 0.5e-3
    r_i_m = (geom.d_c_mm * 0.5 + geom.t_i_mm) * 1.0e-3
    r_e_m = geom.D_e_mm * 0.5e-3
    r_under_jacket_m = (geom.D_e_mm * 0.5 - geom.t_j_mm) * 1.0e-3

    # Conductor cross-sectional area per phase
    A_conductor = np.pi * r_c_m * r_c_m
    cap_conductor = n * A_conductor * CU_DENSITY_KG_PER_M3 * CU_CP_J_PER_KG_K

    # Insulation annulus per phase (3 separate cores)
    A_insulation = np.pi * (r_i_m * r_i_m - r_c_m * r_c_m)
    cap_insulation = n * A_insulation * XLPE_DENSITY_KG_PER_M3 * XLPE_CP_J_PER_KG_K

    # Outer jacket — a single annulus around the assembly
    A_jacket = np.pi * (r_e_m * r_e_m - r_under_jacket_m * r_under_jacket_m)
    cap_jacket = A_jacket * HDPE_DENSITY_KG_PER_M3 * HDPE_CP_J_PER_KG_K

    return float(cap_conductor + cap_insulation + cap_jacket)


# ---------------------------------------------------------------------------
# Transient solver
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TransientResult:
    """Output of a transient simulation.

    Attributes:
        time_s: Output times [s].
        conductor_temp_C: Conductor temperature trace [degC].
        total_loss_W_per_m: Total cable loss per metre at each time [W/m].
        R_total_KmW: IEC R_total = T1 + n(1+lambda_1)(T3+T4) [K m/W].
        C_c_J_per_K_m: Total cable thermal capacitance per metre [J/(K m)].
        time_constant_s: First-order lag time constant tau = R_total * C_c / n [s].
    """

    time_s: np.ndarray
    conductor_temp_C: np.ndarray
    total_loss_W_per_m: np.ndarray
    R_total_KmW: float
    C_c_J_per_K_m: float
    time_constant_s: float


def total_thermal_resistance(
    geom: CableGeometry,
    mat: CableMaterials,
    install: InstallationConditions,
    sheath_loss_factor: float = 0.05,
) -> float:
    """Total cable-to-ambient thermal resistance for the lumped model [K m/W]."""
    T1 = thermal_resistance_T1(geom, mat)
    T3 = thermal_resistance_T3(geom, mat)
    T4 = thermal_resistance_T4(geom, install)
    n = geom.n_conductors
    return T1 + n * (1.0 + sheath_loss_factor) * (T3 + T4)


def simulate_transient(
    current_profile: Callable[[float], float],
    times_s: Sequence[float],
    line_voltage_V_rms: float,
    geom: CableGeometry,
    mat: CableMaterials,
    install: InstallationConditions,
    initial_temp_C: float | None = None,
    sheath_loss_factor: float = 0.05,
) -> TransientResult:
    """Simulate the lumped single-node thermal response to a time-varying load.

    Args:
        current_profile: Function `t_s -> I_A` returning the phase current
            at time `t_s` seconds. Must be defined over `times_s`.
        times_s: Output times (s). Must be strictly increasing.
        line_voltage_V_rms: Line-to-line voltage, r.m.s.
        geom: Cable geometry.
        mat: Cable material properties.
        install: Installation conditions.
        initial_temp_C: Conductor temperature at t = times_s[0]. If None,
            uses the steady-state temperature for `current_profile(t0)`.
        sheath_loss_factor: lambda_1.

    Returns:
        TransientResult with the conductor temperature trace.
    """
    times = np.asarray(times_s, dtype=float)
    if times.ndim != 1 or len(times) < 2:
        raise ValueError("times_s must be a 1-D sequence of length >= 2")
    if not np.all(np.diff(times) > 0.0):
        raise ValueError("times_s must be strictly increasing")

    voltage_phase = line_voltage_V_rms / np.sqrt(3.0)
    W_d = dielectric_loss_per_phase(voltage_phase, mat)
    n = geom.n_conductors

    R_total = total_thermal_resistance(geom, mat, install, sheath_loss_factor)
    C_total = cable_thermal_capacitance(geom)
    # Per-phase thermal capacitance: each phase shares C_total / n. With
    # per-phase heat input (I^2 R + W_d) and the IEC R_total coefficient,
    # the lumped ODE recovers the IEC 60287 steady state exactly.
    C_per_phase = C_total / n
    tau = R_total * C_per_phase  # first-order lag time constant [s]

    # Pre-compute the IEC layer factors used in the steady-state target
    # (recovered exactly when dT/dt = 0).
    T1 = thermal_resistance_T1(geom, mat)
    T3 = thermal_resistance_T3(geom, mat)
    T4 = thermal_resistance_T4(geom, install)
    layer_factor_W_d = 0.5 * T1 + n * (1.0 + sheath_loss_factor) * (T3 + T4)
    # I^2 R coefficient is exactly R_total (= T1 + n(1+l1)(T3+T4))

    if initial_temp_C is None:
        # Approximate steady-state at I(t0). The temperature dependence of R
        # is captured by a one-shot iteration since this is just an initial
        # guess for the integrator.
        I0 = float(current_profile(float(times[0])))
        R0 = ac_resistance_at_temp(install.ambient_soil_temp_C + 50.0, mat)
        I2R0 = I0 * I0 * R0
        delta_T0 = I2R0 * R_total + W_d * layer_factor_W_d
        initial_temp_C = install.ambient_soil_temp_C + delta_T0

    def rhs(t: float, y: np.ndarray) -> np.ndarray:
        T_c = float(y[0])
        I_t = float(current_profile(t))
        R = ac_resistance_at_temp(T_c, mat)
        I2R = I_t * I_t * R
        # IEC 60287 steady-state target for the present (I, T_c)
        delta_T_target = I2R * R_total + W_d * layer_factor_W_d
        T_target = install.ambient_soil_temp_C + delta_T_target
        # First-order lag toward target with time constant tau = C_phase * R_total
        dTdt = (T_target - T_c) / tau
        return np.array([dTdt])

    sol = solve_ivp(
        rhs,
        t_span=(float(times[0]), float(times[-1])),
        y0=np.array([initial_temp_C]),
        t_eval=times,
        method="LSODA",
        rtol=1e-6,
        atol=1e-6,
    )
    if not sol.success:
        raise RuntimeError(f"transient solver failed: {sol.message}")

    T_trace = sol.y[0]

    # Recompute losses at each output point for downstream use
    losses: List[float] = []
    for t, T in zip(times, T_trace):
        I_t = float(current_profile(float(t)))
        R = ac_resistance_at_temp(float(T), mat)
        losses.append((I_t * I_t * R + W_d) * n)

    return TransientResult(
        time_s=times.copy(),
        conductor_temp_C=T_trace,
        total_loss_W_per_m=np.asarray(losses),
        R_total_KmW=R_total,
        C_c_J_per_K_m=C_total,
        time_constant_s=tau,
    )
