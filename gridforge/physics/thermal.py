"""
IEC 60287 steady-state thermal model for buried distribution cables.

Implements the closed-form steady-state heat equation for a single buried
3-core XLPE cable with individually screened cores (the dominant configuration
in UK 11 kV distribution networks).

The IEC 60287-1-1 governing equation for conductor temperature rise above
ambient soil temperature is:

    delta_theta = (I^2 R + 0.5 W_d) * T1
                + (I^2 R + W_d) * n * T2
                + (I^2 R + W_d) * (1 + lambda_1) * n * T3
                + (I^2 R + W_d) * (1 + lambda_1 + lambda_2) * n * T4

where:
    I        : current per conductor [A]
    R        : a.c. conductor resistance per unit length at theta_c [ohm/m]
    W_d      : dielectric loss per phase per unit length [W/m]
    n        : number of load-carrying conductors (3 for three-phase)
    lambda_1 : sheath/screen loss factor [-]
    lambda_2 : armour loss factor [-] (zero for non-armoured cables)
    T1..T4   : thermal resistances per unit length [K.m/W]

R(theta_c) appears on both sides, so the equation is solved by fixed-point
iteration on the conductor temperature.

References:
    IEC 60287-1-1:2006 — Calculation of the current rating, Part 1: General.
    IEC 60287-2-1:2015 — Thermal resistance, Part 2: Cables in air, buried, etc.
    IEC 60228:2004     — Conductors of insulated cables.

This module covers a single buried cable. Cable groups, ducts, riser sections,
and forced ventilation are out of scope for v0.0.1.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import log, pi, sqrt


# ---------------------------------------------------------------------------
# Physical constants
# ---------------------------------------------------------------------------

OMEGA_50HZ: float = 2.0 * pi * 50.0  # rad/s, UK mains angular frequency

# Material thermal resistivities [K.m/W] — IEC 60287-2-1 Table 1
RHO_T_XLPE: float = 3.5
RHO_T_PILC: float = 5.0
RHO_T_PVC: float = 5.0
RHO_T_HDPE: float = 3.5
RHO_T_PE: float = 3.5

# Soil thermal resistivity defaults [K.m/W] — typical UK ranges (BS 7870)
RHO_T_SOIL_TYPICAL_UK: float = 1.0
RHO_T_SOIL_DRY_SAND: float = 2.5

# Copper temperature coefficient of resistance [1/K] — IEC 60228
COPPER_ALPHA: float = 3.93e-3

# Default a.c./d.c. resistance ratio for 240 mm^2 stranded Cu @ 50 Hz
# (skin and proximity effects per IEC 60287-1-1 Section 2.1)
R_AC_DC_RATIO_DEFAULT: float = 1.02


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class CableGeometry:
    """Cable cross-sectional geometry.

    All linear dimensions in millimetres. The geometry assumes a 3-core
    cable with individually screened cores enclosed by a common outer jacket.
    """

    d_c_mm: float           # conductor diameter
    t_i_mm: float           # insulation thickness, conductor surface to screen
    t_j_mm: float           # outer jacket thickness
    D_e_mm: float           # cable overall outside diameter
    n_conductors: int = 3   # load-carrying conductors


@dataclass(frozen=True)
class CableMaterials:
    """Material properties governing thermal and electrical behaviour."""

    rho_t_insulation_KmW: float = RHO_T_XLPE
    rho_t_jacket_KmW: float = RHO_T_HDPE
    R_dc_20C_ohm_per_m: float = 7.55e-5     # 240 mm^2 Cu @ 20 degC, IEC 60228
    alpha_per_C: float = COPPER_ALPHA       # temperature coefficient
    R_ac_dc_ratio: float = R_AC_DC_RATIO_DEFAULT
    capacitance_F_per_m: float = 2.0e-10    # ~0.2 uF/km, typical 11 kV XLPE
    tan_delta: float = 1.0e-3               # XLPE dielectric loss tangent


@dataclass(frozen=True)
class InstallationConditions:
    """Burial geometry and ambient soil conditions."""

    burial_depth_m: float
    soil_thermal_resistivity_KmW: float
    ambient_soil_temp_C: float = 15.0


@dataclass(frozen=True)
class ThermalResistances:
    """Per-unit-length thermal resistances [K.m/W]."""

    T1_KmW: float
    T2_KmW: float
    T3_KmW: float
    T4_KmW: float


@dataclass(frozen=True)
class ThermalSolution:
    """Result of a steady-state thermal solve."""

    conductor_temp_C: float
    sheath_temp_C: float
    jacket_temp_C: float
    soil_interface_temp_C: float
    R_ac_at_temp_ohm_per_m: float
    W_dielectric_W_per_m: float
    I2R_loss_W_per_m: float
    total_loss_W_per_m: float
    thermal_resistances: ThermalResistances
    iterations: int
    converged: bool


# ---------------------------------------------------------------------------
# Thermal-resistance components (IEC 60287-2-1)
# ---------------------------------------------------------------------------

def thermal_resistance_T1(geom: CableGeometry, mat: CableMaterials) -> float:
    """T1 — between conductor and screen, per phase [K.m/W].

    For a screened single-core or 3-core SL cable (IEC 60287-2-1, eq. 4):

        T1 = (rho_T / 2 pi) * ln(1 + 2 t_i / d_c)

    where t_i is the insulation thickness from conductor surface to screen.
    """
    if geom.d_c_mm <= 0.0 or geom.t_i_mm < 0.0:
        raise ValueError("conductor diameter must be positive, insulation thickness non-negative")
    return (mat.rho_t_insulation_KmW / (2.0 * pi)) * log(1.0 + 2.0 * geom.t_i_mm / geom.d_c_mm)


def thermal_resistance_T3(geom: CableGeometry, mat: CableMaterials) -> float:
    """T3 — outer covering / jacket [K.m/W].

    IEC 60287-2-1 eq. 22 form, using overall diameter and jacket thickness:

        T3 = (rho_j / 2 pi) * ln(D_e / (D_e - 2 t_j))
    """
    D_under_jacket = geom.D_e_mm - 2.0 * geom.t_j_mm
    if D_under_jacket <= 0.0:
        raise ValueError("jacket thickness exceeds overall radius")
    return (mat.rho_t_jacket_KmW / (2.0 * pi)) * log(geom.D_e_mm / D_under_jacket)


def thermal_resistance_T4(geom: CableGeometry, install: InstallationConditions) -> float:
    """T4 — surrounding medium for a single buried cable [K.m/W].

    IEC 60287-2-1 eq. 51 (single isolated buried cable):

        T4 = (rho_s / 2 pi) * ln(2 u + sqrt((2 u)^2 - 1))

    where u = 2 L_b / D_e, L_b is depth to cable centre, and D_e is the
    cable overall diameter. Equivalently arccosh(2 L_b / D_e) for
    2 L_b / D_e >= 1.
    """
    D_e_m = geom.D_e_mm * 1.0e-3
    two_u = 2.0 * install.burial_depth_m / D_e_m
    if two_u < 1.0:
        raise ValueError(
            f"cable too close to surface for IEC formula (2L_b/D_e = {two_u:.3f} < 1)"
        )
    return (install.soil_thermal_resistivity_KmW / (2.0 * pi)) * log(
        two_u + sqrt(two_u * two_u - 1.0)
    )


# ---------------------------------------------------------------------------
# Loss components
# ---------------------------------------------------------------------------

def dielectric_loss_per_phase(voltage_V_phase_to_ground: float, mat: CableMaterials) -> float:
    """Dielectric loss per phase per unit length [W/m].

        W_d = omega * C * U_0^2 * tan(delta)

    where U_0 is the phase-to-ground r.m.s. voltage.
    """
    return (
        OMEGA_50HZ
        * mat.capacitance_F_per_m
        * voltage_V_phase_to_ground * voltage_V_phase_to_ground
        * mat.tan_delta
    )


def ac_resistance_at_temp(temperature_C: float, mat: CableMaterials) -> float:
    """A.c. conductor resistance at operating temperature [ohm/m].

    Combines temperature correction (IEC 60228) with the configured a.c./d.c.
    resistance ratio that bundles skin and proximity effects.
    """
    R_dc_T = mat.R_dc_20C_ohm_per_m * (1.0 + mat.alpha_per_C * (temperature_C - 20.0))
    return R_dc_T * mat.R_ac_dc_ratio


# ---------------------------------------------------------------------------
# Steady-state solver
# ---------------------------------------------------------------------------

def solve_steady_state(
    current_per_phase_A: float,
    line_voltage_V_rms: float,
    geom: CableGeometry,
    mat: CableMaterials,
    install: InstallationConditions,
    sheath_loss_factor: float = 0.05,
    armour_loss_factor: float = 0.0,
    max_iterations: int = 50,
    tol_C: float = 1.0e-3,
) -> ThermalSolution:
    """Solve the IEC 60287 steady-state heat balance for conductor temperature.

    Fixed-point iteration on theta_c, since R = R(theta_c). Convergence is
    monotone and rapid (typically 5-8 iterations) because R is a slowly varying
    function of theta_c over the operating range.

    Args:
        current_per_phase_A: Steady-state phase current, r.m.s.
        line_voltage_V_rms: Line-to-line voltage, r.m.s.
        geom: Cable geometry.
        mat: Cable material properties.
        install: Installation conditions.
        sheath_loss_factor: lambda_1, induced losses in the metallic screen.
            Typical: 0.05 for 3-core common-jacket, larger for single-core.
        armour_loss_factor: lambda_2, induced losses in armour wires. Zero for
            non-armoured 11 kV distribution cables.
        max_iterations: Hard cap on iteration count.
        tol_C: Convergence tolerance on theta_c.

    Returns:
        ThermalSolution with conductor / sheath / jacket / soil-interface
        temperatures, loss components, and the per-component thermal resistances.
    """
    if current_per_phase_A < 0.0:
        raise ValueError("current must be non-negative")
    if max_iterations < 1:
        raise ValueError("max_iterations must be >= 1")

    voltage_phase = line_voltage_V_rms / sqrt(3.0)
    W_d = dielectric_loss_per_phase(voltage_phase, mat)

    T1 = thermal_resistance_T1(geom, mat)
    T2 = 0.0  # no separate armour layer in 3-core XLPE distribution cables
    T3 = thermal_resistance_T3(geom, mat)
    T4 = thermal_resistance_T4(geom, install)
    n = geom.n_conductors

    # Initial guess: 50 C above ambient. Iteration is robust to a wide range
    # of starting points.
    theta_c = install.ambient_soil_temp_C + 50.0
    converged = False
    iteration = 0

    for iteration in range(1, max_iterations + 1):
        R = ac_resistance_at_temp(theta_c, mat)
        I2R = current_per_phase_A * current_per_phase_A * R

        delta_theta = (
            (I2R + 0.5 * W_d) * T1
            + (I2R + W_d) * (n * T2)
            + (I2R + W_d) * (1.0 + sheath_loss_factor) * (n * T3)
            + (I2R + W_d) * (1.0 + sheath_loss_factor + armour_loss_factor) * (n * T4)
        )
        theta_c_new = install.ambient_soil_temp_C + delta_theta

        if abs(theta_c_new - theta_c) < tol_C:
            theta_c = theta_c_new
            converged = True
            break
        theta_c = theta_c_new

    R_final = ac_resistance_at_temp(theta_c, mat)
    I2R_final = current_per_phase_A * current_per_phase_A * R_final
    total_W = (I2R_final + W_d) * n

    # Back-compute intermediate temperatures along the heat-flow path
    drop_insulation = (I2R_final + 0.5 * W_d) * T1
    theta_sheath = theta_c - drop_insulation
    drop_jacket = (I2R_final + W_d) * (1.0 + sheath_loss_factor) * (n * T3)
    theta_jacket = theta_sheath - drop_jacket
    drop_soil = (I2R_final + W_d) * (1.0 + sheath_loss_factor) * (n * T4)
    theta_soil_iface = theta_jacket - drop_soil

    return ThermalSolution(
        conductor_temp_C=theta_c,
        sheath_temp_C=theta_sheath,
        jacket_temp_C=theta_jacket,
        soil_interface_temp_C=theta_soil_iface,
        R_ac_at_temp_ohm_per_m=R_final,
        W_dielectric_W_per_m=W_d,
        I2R_loss_W_per_m=I2R_final,
        total_loss_W_per_m=total_W,
        thermal_resistances=ThermalResistances(
            T1_KmW=T1, T2_KmW=T2, T3_KmW=T3, T4_KmW=T4
        ),
        iterations=iteration,
        converged=converged,
    )
