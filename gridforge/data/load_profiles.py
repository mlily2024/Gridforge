"""
Load-profile library for synthetic cable-year generation.

Each profile is a function `f(t_s, peak_A, base_A, seed) -> I_A` that returns
the instantaneous phase current at time `t_s` (seconds since the start of the
simulation). The profiles capture diurnal, weekly, and seasonal modulation
typical of UK distribution networks.

Reference shapes loosely follow the published Elexon BMRS settlement-period
demand patterns (2023-2025 winter / summer) and the Open Networks Project
secondary-substation studies. Magnitudes are intentionally normalised: the
caller scales by `peak_A` and `base_A` to match a particular asset.

The seed parameter selects a reproducible stochastic component (small noise
band) that lets identical (profile, seed) pairs return identical traces.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

LoadProfile = Callable[[float, float, float, int], float]

# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

SECONDS_PER_DAY: float = 24.0 * 3600.0
SECONDS_PER_WEEK: float = 7.0 * SECONDS_PER_DAY
SECONDS_PER_YEAR: float = 365.25 * SECONDS_PER_DAY


def _seasonal_factor(t_s: float) -> float:
    """Winter peak / summer trough.

    UK demand peaks in January (heating, lighting) and troughs in July.
    Returns a multiplier in [0.85, 1.15] applied to the diurnal shape.
    """
    phase = 2.0 * np.pi * (t_s / SECONDS_PER_YEAR - 14.0 / 365.25)  # peak ~14 Jan
    return 1.0 + 0.15 * float(np.cos(phase))


def _weekend_factor(t_s: float) -> float:
    """Weekday peaks slightly higher than weekend on residential / commercial.

    Returns 1.0 on weekdays, 0.85 on weekends. Day index 0 = Monday.
    """
    day_of_week = int((t_s // SECONDS_PER_DAY) % 7)
    return 0.85 if day_of_week >= 5 else 1.0


def _stochastic_noise(t_s: float, seed: int, amplitude: float = 0.02) -> float:
    """Reproducible small noise band, deterministic in (t_s, seed).

    Uses a hash-based RNG so the same (t_s, seed) always returns the same
    value — required for deterministic dataset regeneration.
    """
    rng = np.random.default_rng(int(seed) * 1_000_003 + int(t_s) // 60)
    return 1.0 + amplitude * float(rng.standard_normal())


# ---------------------------------------------------------------------------
# Profile library
# ---------------------------------------------------------------------------


def residential(t_s: float, peak_A: float, base_A: float, seed: int = 0) -> float:
    """UK domestic profile — two daily peaks (morning ~08:00, evening ~18:00).

    Captures heating-driven morning rise and TV/cooking evening peak. Weekend
    morning peak is later and lower; weekend evening shifted slightly.
    Evening is the dominant peak (max shape = 1.0); morning is ~70 % of evening.
    """
    hour_of_day = (t_s / 3600.0) % 24.0
    is_weekend = ((t_s // SECONDS_PER_DAY) % 7) >= 5

    morning_peak_h = 9.0 if is_weekend else 8.0
    evening_peak_h = 19.0 if is_weekend else 18.0
    morning_amp = 0.6 if is_weekend else 0.7

    morning = morning_amp * np.exp(-(((hour_of_day - morning_peak_h) / 2.0) ** 2))
    evening = np.exp(-(((hour_of_day - evening_peak_h) / 2.5) ** 2))
    shape = max(morning, evening)

    seasonal = _seasonal_factor(t_s)
    noise = _stochastic_noise(t_s, seed)
    return float((base_A + (peak_A - base_A) * shape) * seasonal * noise)


def commercial(t_s: float, peak_A: float, base_A: float, seed: int = 0) -> float:
    """Commercial / office profile — flat daytime peak 08:00-18:00, low overnight.

    Strong weekday/weekend asymmetry: weekend demand drops to near base load.
    """
    hour_of_day = (t_s / 3600.0) % 24.0
    is_weekend = ((t_s // SECONDS_PER_DAY) % 7) >= 5

    if is_weekend:
        return float((base_A + 0.15 * (peak_A - base_A)) * _stochastic_noise(t_s, seed))

    if hour_of_day < 7.0 or hour_of_day > 19.0:
        shape = 0.1
    else:
        # Plateau with smoothed edges (logistic-like)
        shape = (
            1.0
            / (1.0 + np.exp(-3.0 * (hour_of_day - 7.5)))
            * 1.0
            / (1.0 + np.exp(-3.0 * (18.5 - hour_of_day)))
        )
    seasonal = _seasonal_factor(t_s)
    noise = _stochastic_noise(t_s, seed)
    return float((base_A + (peak_A - base_A) * shape) * seasonal * noise)


def industrial(t_s: float, peak_A: float, base_A: float, seed: int = 0) -> float:
    """Industrial profile — high base load, modest peaks, weekends still active.

    Continuous-process loads (refrigeration, water treatment, telecoms) keep
    the cable warm 24/7. Diurnal swing is small.
    """
    hour_of_day = (t_s / 3600.0) % 24.0
    shape = 0.7 + 0.3 * np.cos(2.0 * np.pi * (hour_of_day - 14.0) / 24.0)  # peak ~14:00
    seasonal = _seasonal_factor(t_s) * 0.5 + 0.5  # weaker seasonality
    noise = _stochastic_noise(t_s, seed, amplitude=0.01)
    return float((base_A + (peak_A - base_A) * shape) * seasonal * noise)


def mixed(t_s: float, peak_A: float, base_A: float, seed: int = 0) -> float:
    """Mixed urban feeder — average of residential + commercial."""
    res = residential(t_s, peak_A, base_A, seed)
    com = commercial(t_s, peak_A, base_A, seed)
    return 0.5 * (res + com)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

PROFILES: dict[str, LoadProfile] = {
    "residential": residential,
    "commercial": commercial,
    "industrial": industrial,
    "mixed": mixed,
}


@dataclass(frozen=True)
class LoadSpec:
    """Reproducible load specification for one cable-year simulation."""

    profile_name: str
    peak_A: float
    base_A: float
    seed: int = 0

    def __call__(self, t_s: float) -> float:
        if self.profile_name not in PROFILES:
            raise KeyError(f"unknown load profile: {self.profile_name}")
        return PROFILES[self.profile_name](t_s, self.peak_A, self.base_A, self.seed)
