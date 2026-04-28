"""
Synthetic UK weather profile for cable-year simulation.

Provides a callable ambient soil temperature `T(t_s)` calibrated to UK
climatology at 0.8 m burial depth. Soil temperature lags air temperature
by roughly two months and has reduced amplitude — at 0.8 m the seasonal
swing is about +/- 6 K around an annual mean of 11 degC.

Reference data: Met Office UK soil-temperature climatology (1991-2020
reference period) and the UKCP18 baseline. Magnitudes are typical UK-wide
averages; per-site variation is on the order of +/- 2 K and is captured
by the per-seed stochastic component.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

SECONDS_PER_DAY: float = 24.0 * 3600.0
SECONDS_PER_YEAR: float = 365.25 * SECONDS_PER_DAY

# UK climatology at 0.8 m burial depth
SOIL_TEMP_MEAN_C: float = 11.0
SOIL_TEMP_AMPLITUDE_K: float = 6.0
SOIL_TEMP_PEAK_DOY: float = 220.0   # day-of-year of warmest soil ~ early Aug

# Diurnal coupling at 0.8 m is small (< 0.5 K) — included for completeness
DIURNAL_AMPLITUDE_K: float = 0.3


def soil_ambient_C(t_s: float, seed: int = 0) -> float:
    """Synthetic UK soil temperature at 0.8 m depth [degC]."""
    seasonal = SOIL_TEMP_AMPLITUDE_K * np.cos(
        2.0 * np.pi * (t_s / SECONDS_PER_YEAR - SOIL_TEMP_PEAK_DOY / 365.25)
    )
    diurnal = DIURNAL_AMPLITUDE_K * np.cos(
        2.0 * np.pi * (t_s / SECONDS_PER_DAY - 14.0 / 24.0)  # peak ~14:00
    )
    rng = np.random.default_rng(seed * 7919 + int(t_s) // 3600)
    weather_noise = 0.5 * float(rng.standard_normal())
    return float(SOIL_TEMP_MEAN_C + seasonal + diurnal + weather_noise)


def soil_moisture_index(t_s: float, seed: int = 0) -> float:
    """Synthetic soil-moisture index in [0, 1].

    Higher in winter, lower in summer (UK precipitation pattern). Used as a
    placeholder feature in telemetry; not yet coupled to the thermal model
    (soil thermal resistivity is held constant in v0.0.x).
    """
    seasonal = 0.5 + 0.3 * np.cos(
        2.0 * np.pi * (t_s / SECONDS_PER_YEAR - 14.0 / 365.25)  # wettest ~mid-Jan
    )
    rng = np.random.default_rng(int(seed) * 6151 + int(int(t_s) // int(SECONDS_PER_DAY)))
    storm_noise = 0.1 * float(rng.standard_normal())
    return float(np.clip(seasonal + storm_noise, 0.0, 1.0))


@dataclass(frozen=True)
class WeatherSpec:
    """Reproducible weather specification for one cable-year simulation."""

    seed: int = 0
    mean_C: float = SOIL_TEMP_MEAN_C
    amplitude_K: float = SOIL_TEMP_AMPLITUDE_K

    def ambient_C(self, t_s: float) -> float:
        seasonal = self.amplitude_K * np.cos(
            2.0 * np.pi * (t_s / SECONDS_PER_YEAR - SOIL_TEMP_PEAK_DOY / 365.25)
        )
        diurnal = DIURNAL_AMPLITUDE_K * np.cos(
            2.0 * np.pi * (t_s / SECONDS_PER_DAY - 14.0 / 24.0)
        )
        rng = np.random.default_rng(self.seed * 7919 + int(t_s) // 3600)
        weather_noise = 0.5 * float(rng.standard_normal())
        return float(self.mean_C + seasonal + diurnal + weather_noise)

    def moisture(self, t_s: float) -> float:
        return soil_moisture_index(t_s, self.seed)
