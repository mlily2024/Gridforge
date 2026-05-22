"""
Soil moisture to thermal resistivity coupling.

Soil thermal resistivity rho_T [K.m/W] is the dominant thermal resistance
in the IEC 60287 buried-cable rating model — typically 50-80 % of the
total resistance from conductor to ambient. It is highly sensitive to
volumetric water content theta: a dry summer can shift soil rho_T from
roughly 0.7 K.m/W (wet) to over 2.5 K.m/W (very dry), more than tripling
the external thermal resistance and roughly doubling the conductor
temperature rise above ambient under the same loading.

This module provides a continuous, monotonic mapping from theta to rho_T
using a two-point log-linear interpolation between the dry and saturated
endpoints of a soil:

    rho_T(theta) = rho_dry * (rho_sat / rho_dry) ** (theta / theta_sat)        (1)

This form is the simplest function that

  * is monotone decreasing in theta (more water = less resistivity),
  * passes through (theta = 0, rho_dry) and (theta = theta_sat, rho_sat)
    exactly,
  * interpolates resistivity in log-space, which matches the physical
    behaviour: dry-soil resistivity rises sharply as the last water
    bridges between grains evaporate.

Constants are tabulated per soil type. The default "loam" corresponds to
typical UK distribution-cable backfill: rho_dry ~ 2.8 K.m/W,
rho_sat ~ 0.5 K.m/W, theta_sat ~ 0.45. All numbers sit inside the
ranges reported in

  * IEC 60287-3-1 Section 4.2 (backfill thermal resistivity classes),
  * Anders, G.J. (1997), *Rating of Electric Power Cables*, IEEE Press,
    Ch. 7,
  * Brakelmann, H. (2004), "Re-thinking the dimensioning of buried
    cables in the light of moisture migration", CIGRE B1-203.

Future work may calibrate per-DNO backfill types or move to the full
Campbell & Norman (1998) five-parameter form for higher fidelity. The
current form is intentionally minimal and auditable.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import numpy as np


# ---------------------------------------------------------------------------
# Soil-type constants
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SoilType:
    """Endpoint constants for one soil type's moisture-to-resistivity curve.

    Attributes:
        name:           Human-readable identifier (e.g. "loam").
        rho_dry_KmW:    Thermal resistivity at theta = 0 [K.m/W].
        rho_sat_KmW:    Thermal resistivity at theta = theta_sat [K.m/W].
        theta_sat:      Volumetric water content at saturation [m3/m3].
    """

    name: str
    rho_dry_KmW: float
    rho_sat_KmW: float
    theta_sat: float


# Loam — typical UK distribution-cable backfill. Default for this module.
LOAM: Final[SoilType] = SoilType(
    name="loam",
    rho_dry_KmW=2.8,
    rho_sat_KmW=0.5,
    theta_sat=0.45,
)

# Sandy — coarser grain, lower theta_sat, drains and dries faster than loam.
# Higher dry resistivity; lower saturated resistivity.
SANDY: Final[SoilType] = SoilType(
    name="sandy",
    rho_dry_KmW=3.5,
    rho_sat_KmW=0.4,
    theta_sat=0.35,
)

# Clay — finer grain, higher theta_sat, retains moisture longer. Lower dry
# resistivity than loam (more bound water at low theta) but higher saturated
# resistivity (slower convective transport between grains).
CLAY: Final[SoilType] = SoilType(
    name="clay",
    rho_dry_KmW=2.0,
    rho_sat_KmW=0.6,
    theta_sat=0.50,
)

KNOWN_SOIL_TYPES: Final[dict[str, SoilType]] = {
    "loam": LOAM,
    "sandy": SANDY,
    "clay": CLAY,
}

DEFAULT_SOIL: Final[SoilType] = LOAM


def _resolve_soil(soil: SoilType | str) -> SoilType:
    """Accept either a SoilType instance or a string key from KNOWN_SOIL_TYPES."""
    if isinstance(soil, SoilType):
        return soil
    if soil not in KNOWN_SOIL_TYPES:
        raise ValueError(
            f"unknown soil type {soil!r}; "
            f"available: {sorted(KNOWN_SOIL_TYPES)}"
        )
    return KNOWN_SOIL_TYPES[soil]


# ---------------------------------------------------------------------------
# Public functions
# ---------------------------------------------------------------------------


def theta_to_rho_t(
    theta: float,
    soil: SoilType | str = DEFAULT_SOIL,
) -> float:
    """Soil thermal resistivity [K.m/W] from volumetric water content theta.

    Implements equation (1) of the module docstring. The function is
    monotone decreasing in theta and passes exactly through both
    endpoints (rho_dry at theta=0; rho_sat at theta=theta_sat).

    Inputs outside [0, theta_sat] are clamped rather than rejected — a
    moisture index slightly negative due to noise or marginally above
    saturation due to interpolation rounding is treated as the nearest
    endpoint, not an error. Callers that need strict validation should
    check theta themselves before calling.

    Args:
        theta:  Volumetric water content [m3/m3 or unitless 0-1].
        soil:   SoilType instance OR a string key from KNOWN_SOIL_TYPES
                ("loam", "sandy", "clay"). Defaults to LOAM.

    Returns:
        Soil thermal resistivity in K.m/W.
    """
    soil_resolved = _resolve_soil(soil)
    theta_clamped = float(np.clip(theta, 0.0, soil_resolved.theta_sat))
    ratio = soil_resolved.rho_sat_KmW / soil_resolved.rho_dry_KmW
    exponent = theta_clamped / soil_resolved.theta_sat
    return float(soil_resolved.rho_dry_KmW * (ratio ** exponent))


def theta_array_to_rho_t(
    thetas: np.ndarray,
    soil: SoilType | str = DEFAULT_SOIL,
) -> np.ndarray:
    """Vectorised version of `theta_to_rho_t`.

    Args:
        thetas:  Array of volumetric water-content values [m3/m3].
        soil:    Soil type (see `theta_to_rho_t`).

    Returns:
        Array of soil thermal resistivities [K.m/W], same shape as `thetas`.
    """
    soil_resolved = _resolve_soil(soil)
    thetas_arr = np.asarray(thetas, dtype=float)
    thetas_clamped = np.clip(thetas_arr, 0.0, soil_resolved.theta_sat)
    ratio = soil_resolved.rho_sat_KmW / soil_resolved.rho_dry_KmW
    exponent = thetas_clamped / soil_resolved.theta_sat
    return soil_resolved.rho_dry_KmW * np.power(ratio, exponent)


__all__ = [
    "SoilType",
    "LOAM",
    "SANDY",
    "CLAY",
    "KNOWN_SOIL_TYPES",
    "DEFAULT_SOIL",
    "theta_to_rho_t",
    "theta_array_to_rho_t",
]
