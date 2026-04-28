"""
Cable-year simulator — composes load profile, weather, failure mode, and
the GridForge physics primitives into a full year (or multi-year) telemetry
trace plus failure-time ground truth.

The simulator runs a hourly forward integration of the lumped first-order
thermal lag with time-varying ambient and load:

    dT_c / dt = ( T_target(t, T_c) - T_c ) / tau

with

    T_target(t, T_c) = T_amb(t)
                     + I(t)^2 R(T_c) * R_total
                     + W_d * ( 0.5 T1 + n (1+lambda_1)(T3+T4) )

where T1, T3, T4 come from `gridforge.physics.thermal`. Failure-mode
multipliers are then applied to the conductor temperature (offset) and
electric field (multiplier) before Crine damage is integrated step by
step.

Unlike `gridforge.physics.transient.simulate_transient`, this routine
accepts time-varying ambient temperature and uses an explicit RK2 step at
hourly resolution — sufficient for the dataset-generation timescale.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from ..physics.ageing import (
    SECONDS_PER_YEAR,
    CrineParameters,
    damage_rate,
)
from ..physics.cable_archetype import (
    ARCHETYPES,
    UK_11KV_240MM2_XLPE_3CORE,
    UK_TYPICAL_INSTALLATION,
    archetype_by_name,
)
from ..physics.electrical import max_e_field
from ..physics.thermal import (
    CableGeometry,
    CableMaterials,
    InstallationConditions,
    ac_resistance_at_temp,
    dielectric_loss_per_phase,
    thermal_resistance_T1,
    thermal_resistance_T3,
    thermal_resistance_T4,
)
from ..physics.transient import cable_thermal_capacitance
from .failure_modes import FailureMode, HealthyMode
from .load_profiles import LoadSpec
from .weather import WeatherSpec


@dataclass(frozen=True)
class CableYearSpec:
    """Reproducible specification for one cable-year simulation.

    `archetype_name`, when provided, resolves to a (geometry, materials)
    tuple via `gridforge.physics.cable_archetype.ARCHETYPES`. If left as
    None, the simulator falls back to the canonical 240 mm^2 XLPE
    archetype, preserving backward compatibility with v0.0.2 callers.
    """

    cable_id: str
    duration_years: float
    load: LoadSpec
    weather: WeatherSpec
    failure_mode: FailureMode = field(default_factory=HealthyMode)
    line_voltage_V_rms: float = 11_000.0
    sheath_loss_factor: float = 0.05
    sample_period_s: float = 3600.0  # hourly
    crine: CrineParameters = field(default_factory=CrineParameters)
    archetype_name: str | None = None


@dataclass(frozen=True)
class CableYearResult:
    """Output of one cable-year simulation."""

    cable_id: str
    times_s: np.ndarray
    current_A: np.ndarray
    ambient_C: np.ndarray
    moisture: np.ndarray
    conductor_C: np.ndarray
    e_field_V_per_m: np.ndarray
    pd_rate_relative: np.ndarray
    cumulative_damage: np.ndarray
    failure_time_s: Optional[float]
    R_total_KmW: float
    C_cable_J_per_K_m: float
    time_constant_s: float
    failure_mode_name: str
    load_profile_name: str
    seed: int
    archetype_name: str


def simulate_cable_year(
    spec: CableYearSpec,
    geom: CableGeometry | None = None,
    mat: CableMaterials | None = None,
    install: InstallationConditions | None = None,
) -> CableYearResult:
    """Simulate one cable's history over `spec.duration_years` years.

    Returns a CableYearResult with hourly telemetry: load, ambient, soil
    moisture, conductor temperature, electric field, partial-discharge
    rate (relative), cumulative damage, and failure time (if reached).
    """
    if geom is None or mat is None:
        if spec.archetype_name is not None:
            geom, mat = archetype_by_name(spec.archetype_name)
        else:
            geom, mat = UK_11KV_240MM2_XLPE_3CORE
    if install is None:
        install = UK_TYPICAL_INSTALLATION

    archetype_name_resolved = (
        spec.archetype_name
        if spec.archetype_name is not None
        else "11kV_240mm2_Cu_XLPE_3c"
    )

    # Pre-compute physics constants
    voltage_phase = spec.line_voltage_V_rms / np.sqrt(3.0)
    W_d = dielectric_loss_per_phase(voltage_phase, mat)
    n = geom.n_conductors
    T1 = thermal_resistance_T1(geom, mat)
    T3 = thermal_resistance_T3(geom, mat)
    T4 = thermal_resistance_T4(geom, install)
    lam1 = spec.sheath_loss_factor
    R_total = T1 + n * (1.0 + lam1) * (T3 + T4)
    layer_factor_W_d = 0.5 * T1 + n * (1.0 + lam1) * (T3 + T4)
    C_total = cable_thermal_capacitance(geom)
    C_per_phase = C_total / n
    tau = R_total * C_per_phase
    E_baseline = max_e_field(voltage_phase, geom)

    # Time grid
    duration_s = spec.duration_years * SECONDS_PER_YEAR
    times = np.arange(0.0, duration_s, spec.sample_period_s, dtype=float)
    if len(times) < 2:
        raise ValueError("duration too short for chosen sample_period_s")
    n_pts = len(times)

    # Pre-allocate output arrays
    currents = np.empty(n_pts)
    ambients = np.empty(n_pts)
    moistures = np.empty(n_pts)
    T_c = np.empty(n_pts)
    E_max = np.empty(n_pts)
    pd_rate_rel = np.empty(n_pts)
    damage = np.empty(n_pts)

    # Initial state — start at the steady-state for I(0), ambient(0)
    I0 = spec.load(0.0)
    amb0 = spec.weather.ambient_C(0.0)
    R0 = ac_resistance_at_temp(amb0 + 50.0, mat)
    delta_T0 = (I0 * I0 * R0) * R_total + W_d * layer_factor_W_d
    T_now = amb0 + delta_T0

    D_now = 0.0
    failure_time: Optional[float] = None
    prev_damage_increment_per_s = 0.0

    for i in range(n_pts):
        t = float(times[i])
        I = spec.load(t)
        amb = spec.weather.ambient_C(t)
        moist = spec.weather.moisture(t)

        # RK2 (midpoint) update of conductor temperature with first-order lag
        R_now = ac_resistance_at_temp(T_now, mat)
        I2R_now = I * I * R_now
        delta_T_target = I2R_now * R_total + W_d * layer_factor_W_d
        T_target = amb + delta_T_target
        k1 = (T_target - T_now) / tau

        T_mid = T_now + 0.5 * spec.sample_period_s * k1
        R_mid = ac_resistance_at_temp(T_mid, mat)
        I2R_mid = I * I * R_mid
        delta_T_target_mid = I2R_mid * R_total + W_d * layer_factor_W_d
        T_target_mid = amb + delta_T_target_mid
        k2 = (T_target_mid - T_mid) / tau

        T_next = T_now + spec.sample_period_s * k2

        # Apply failure-mode temperature offset (post-physics)
        T_eff = T_next + spec.failure_mode.temp_offset_C(t)

        # Electric field with failure-mode multiplier
        E = E_baseline * spec.failure_mode.field_multiplier(t)

        # Partial-discharge rate (relative to baseline)
        pd_rel = spec.failure_mode.pd_rate_multiplier(t)

        # Damage increment via trapezoidal rule using current and prior rates
        rate_now = damage_rate(E, T_eff, spec.crine)
        if i == 0:
            damage_inc = 0.0
        else:
            damage_inc = 0.5 * (prev_damage_increment_per_s + rate_now) * spec.sample_period_s
        prev_damage_increment_per_s = rate_now
        D_now = D_now + damage_inc

        if failure_time is None and D_now >= 1.0:
            failure_time = t

        # Record
        currents[i] = I
        ambients[i] = amb
        moistures[i] = moist
        T_c[i] = T_eff
        E_max[i] = E
        pd_rate_rel[i] = pd_rel
        damage[i] = D_now

        T_now = T_next

    return CableYearResult(
        cable_id=spec.cable_id,
        times_s=times,
        current_A=currents,
        ambient_C=ambients,
        moisture=moistures,
        conductor_C=T_c,
        e_field_V_per_m=E_max,
        pd_rate_relative=pd_rate_rel,
        cumulative_damage=damage,
        failure_time_s=failure_time,
        R_total_KmW=R_total,
        C_cable_J_per_K_m=C_total,
        time_constant_s=tau,
        failure_mode_name=spec.failure_mode.name,
        load_profile_name=spec.load.profile_name,
        seed=spec.load.seed,
        archetype_name=archetype_name_resolved,
    )
