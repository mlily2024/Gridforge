"""
Radial electric-field model for coaxial XLPE-insulated cables.

For a single-core cable (or one core of a 3-core cable with individually
screened cores), the radial electric field in the insulation is governed by
Laplace's equation under cylindrical symmetry. The closed-form solution
under steady-state a.c. operation is

    E(r) = U_0 / (r * ln(r_screen / r_conductor))

where r is the radial coordinate, r_conductor is the outer radius of the
conductor (equal to the inner radius of the insulation), r_screen is the
inner radius of the metallic screen (equal to the outer radius of the
insulation), and U_0 is the phase-to-ground r.m.s. voltage.

The maximum field sits at the conductor surface (r = r_conductor) and the
minimum at the screen surface. Most insulation degradation processes are
driven by E_max, so this is the relevant quantity to feed into the Crine
ageing law.

Reference: any standard text on power-cable design, e.g. Heinhold &
Stubbe, *Power Cables and Their Applications*.
"""

from __future__ import annotations

from math import log

from .thermal import CableGeometry


def conductor_outer_radius_m(geom: CableGeometry) -> float:
    """Conductor outer radius, in metres."""
    return geom.d_c_mm * 0.5e-3


def insulation_outer_radius_m(geom: CableGeometry) -> float:
    """Insulation outer radius (inner side of the metallic screen), in metres."""
    return (geom.d_c_mm * 0.5 + geom.t_i_mm) * 1.0e-3


def radial_e_field(
    radius_m: float,
    voltage_V_phase_to_ground: float,
    geom: CableGeometry,
) -> float:
    """Radial electric field [V/m] at a given radius inside the insulation.

    Args:
        radius_m: Radial coordinate, must lie within the insulation.
        voltage_V_phase_to_ground: U_0, r.m.s. phase-to-ground voltage.
        geom: Cable geometry.

    Raises:
        ValueError: if `radius_m` is outside the insulation annulus.
    """
    r_c = conductor_outer_radius_m(geom)
    r_s = insulation_outer_radius_m(geom)
    if not (r_c <= radius_m <= r_s):
        raise ValueError(
            f"radius {radius_m:.4g} m outside insulation [{r_c:.4g}, {r_s:.4g}] m"
        )
    return voltage_V_phase_to_ground / (radius_m * log(r_s / r_c))


def max_e_field(voltage_V_phase_to_ground: float, geom: CableGeometry) -> float:
    """Maximum radial field [V/m], at the conductor surface."""
    return radial_e_field(conductor_outer_radius_m(geom), voltage_V_phase_to_ground, geom)


def min_e_field(voltage_V_phase_to_ground: float, geom: CableGeometry) -> float:
    """Minimum radial field [V/m], at the insulation outer surface."""
    return radial_e_field(insulation_outer_radius_m(geom), voltage_V_phase_to_ground, geom)


def average_e_field(voltage_V_phase_to_ground: float, geom: CableGeometry) -> float:
    """Average radial field across the insulation thickness [V/m].

    For a coaxial geometry this is simply U_0 / t_insulation. Useful as a
    quick reference but the Crine ageing model is driven by E_max.
    """
    return voltage_V_phase_to_ground / (geom.t_i_mm * 1.0e-3)
