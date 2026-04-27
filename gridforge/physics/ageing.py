"""
Crine 2005 unified dielectric ageing kinetics for XLPE-insulated cables.

The Crine model expresses time-to-failure under combined electric-field and
thermal stress as

    L(E, T) = L_ref * (E_ref / E)^n * exp( (Phi / k_B) * (1/T - 1/T_ref) )

where L is the mean time to failure under constant stress (E, T), n is the
voltage-endurance exponent, Phi is an activation energy, k_B is the
Boltzmann constant, and (E_ref, T_ref, L_ref) is a calibration triple
(typical: design rated stress at design conductor temperature with the
manufacturer-stated design life).

Under time-varying stress, the cumulative damage is

    D(t) = integral_0^t  d_tau / L( E(tau), T(tau) )

with failure at D = 1. Remaining useful life under a forward profile
(E_fwd, T_fwd) is then

    RUL = (1 - D(t_now)) / mean_rate( E_fwd, T_fwd )

This is the Miner-rule generalisation of the inverse-power-law / Arrhenius
model (Crine 2005), and it is the de facto standard in cable-ageing
literature (Mazzanti 2013, IEEE Trans. Dielectr. Electr. Insul.).

References:
    Crine, J.-P. (2005), On the interpretation of some electrical-ageing and
        life-test results, IEEE Trans. Dielectr. Electr. Insul. 12(6), 1089-1107.
    Mazzanti, G. (2013), Life and reliability models for high-voltage DC
        extruded cables, IEEE Electr. Insul. Mag. 29(2), 36-44.
    IEEE Std 1407-2007 — Guide for Accelerated Aging Tests for Medium-Voltage
        Cables.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


BOLTZMANN_eV_PER_K: float = 8.617333262e-5  # k_B in eV/K
KELVIN_OFFSET: float = 273.15
SECONDS_PER_YEAR: float = 365.25 * 24.0 * 3600.0


@dataclass(frozen=True)
class CrineParameters:
    """Crine ageing-law parameters for a given insulation material.

    Defaults below correspond to typical XLPE distribution-cable values from
    the literature. All three calibration scalars (E_ref_V_per_m, T_ref_C,
    L_ref_years) must be provided together: they fix the L_ref point on the
    surface defined by (n, Phi).

    Args:
        n_voltage_endurance: Inverse-power-law exponent on field. XLPE
            distribution cables typically n in [9, 13]. Default 11.
        activation_energy_eV: Phi in the Arrhenius-like temperature term.
            XLPE typical 1.0-1.4 eV. Default 1.1 eV.
        E_ref_V_per_m: Reference electric field (calibration point).
        T_ref_C: Reference temperature (calibration point).
        L_ref_years: Mean time to failure at (E_ref, T_ref).
    """

    n_voltage_endurance: float = 11.0
    activation_energy_eV: float = 1.1
    E_ref_V_per_m: float = 4.0e6        # 4 MV/m, typical XLPE design field
    T_ref_C: float = 90.0
    L_ref_years: float = 40.0           # XLPE design life


def life_at_constant_stress(
    e_field_V_per_m: float,
    temperature_C: float,
    params: CrineParameters | None = None,
) -> float:
    """Mean time to failure [seconds] under constant (E, T) stress."""
    p = params if params is not None else CrineParameters()
    if e_field_V_per_m <= 0.0:
        raise ValueError("electric field must be strictly positive")
    T = temperature_C + KELVIN_OFFSET
    T_ref = p.T_ref_C + KELVIN_OFFSET
    arrhenius = np.exp(
        (p.activation_energy_eV / BOLTZMANN_eV_PER_K) * (1.0 / T - 1.0 / T_ref)
    )
    field_factor = (p.E_ref_V_per_m / e_field_V_per_m) ** p.n_voltage_endurance
    L_years = p.L_ref_years * field_factor * arrhenius
    return float(L_years * SECONDS_PER_YEAR)


def damage_rate(
    e_field_V_per_m: float,
    temperature_C: float,
    params: CrineParameters | None = None,
) -> float:
    """Instantaneous damage rate [1/second]. Reciprocal of life at the same stress."""
    return 1.0 / life_at_constant_stress(e_field_V_per_m, temperature_C, params)


def cumulative_damage(
    times_s: np.ndarray,
    e_field_V_per_m: np.ndarray,
    temperature_C: np.ndarray,
    params: CrineParameters | None = None,
) -> np.ndarray:
    """Damage accumulated over a stress history.

    Trapezoidal integration of the per-step damage rate. Failure at D = 1.

    Args:
        times_s: Strictly increasing time stamps [s], length N.
        e_field_V_per_m: Field at each time, length N.
        temperature_C: Conductor temperature at each time, length N.
        params: Crine parameters; defaults to standard XLPE.

    Returns:
        Array of cumulative damage values, length N. D[0] = 0 by convention.
    """
    times_s = np.asarray(times_s, dtype=float)
    e_arr = np.asarray(e_field_V_per_m, dtype=float)
    T_arr = np.asarray(temperature_C, dtype=float)

    if times_s.shape != e_arr.shape or times_s.shape != T_arr.shape:
        raise ValueError("times_s, e_field, and temperature_C must have the same shape")
    if times_s.ndim != 1 or len(times_s) < 2:
        raise ValueError("inputs must be 1-D and length >= 2")
    if not np.all(np.diff(times_s) > 0.0):
        raise ValueError("times_s must be strictly increasing")

    rates = np.array(
        [damage_rate(float(e), float(T), params) for e, T in zip(e_arr, T_arr)]
    )
    # Trapezoidal cumulative integral
    dt = np.diff(times_s)
    avg_rate = 0.5 * (rates[:-1] + rates[1:])
    increments = avg_rate * dt
    D = np.empty_like(times_s)
    D[0] = 0.0
    D[1:] = np.cumsum(increments)
    return D


def remaining_useful_life(
    current_damage: float,
    forward_e_field_V_per_m: float,
    forward_temperature_C: float,
    params: CrineParameters | None = None,
) -> float:
    """RUL [seconds] given current damage and an assumed forward stress level.

    Linear extrapolation under constant forward stress: RUL = (1 - D) * L_fwd.

    For variable forward stress, integrate damage_rate forward over the
    expected profile until D crosses 1; that path is reserved for
    `gridforge.inference.rul`.
    """
    if not (0.0 <= current_damage <= 1.0):
        raise ValueError("current_damage must lie in [0, 1]")
    L_fwd = life_at_constant_stress(forward_e_field_V_per_m, forward_temperature_C, params)
    return (1.0 - current_damage) * L_fwd


def remaining_useful_life_years(
    current_damage: float,
    forward_e_field_V_per_m: float,
    forward_temperature_C: float,
    params: CrineParameters | None = None,
) -> float:
    """RUL in years under constant forward stress."""
    return remaining_useful_life(
        current_damage, forward_e_field_V_per_m, forward_temperature_C, params
    ) / SECONDS_PER_YEAR
