"""Verification tests for the synthetic weather profile."""

from __future__ import annotations

import numpy as np
import pytest

from gridforge.data.weather import (
    SOIL_TEMP_MEAN_C,
    WeatherSpec,
    soil_ambient_C,
    soil_moisture_index,
)

SECONDS_PER_DAY = 24.0 * 3600.0
SECONDS_PER_YEAR = 365.25 * SECONDS_PER_DAY


class TestSoilAmbient:
    def test_summer_warmer_than_winter(self) -> None:
        winter = soil_ambient_C(15.0 * SECONDS_PER_DAY, seed=0)  # mid-Jan
        summer = soil_ambient_C(220.0 * SECONDS_PER_DAY, seed=0)  # early Aug
        assert summer > winter

    def test_amplitude_in_climatology_range(self) -> None:
        """Across a year, swing should be roughly +/- 6 K from mean."""
        ts = np.linspace(0.0, SECONDS_PER_YEAR, 365)
        Ts = np.array([soil_ambient_C(t, seed=0) for t in ts])
        # Allow noise to extend the apparent range slightly
        assert Ts.min() > SOIL_TEMP_MEAN_C - 8.0
        assert Ts.max() < SOIL_TEMP_MEAN_C + 8.0

    def test_deterministic(self) -> None:
        a = soil_ambient_C(50_000.0, seed=11)
        b = soil_ambient_C(50_000.0, seed=11)
        assert a == b

    def test_different_seeds_differ(self) -> None:
        a = soil_ambient_C(50_000.0, seed=1)
        b = soil_ambient_C(50_000.0, seed=2)
        assert a != b


class TestSoilMoisture:
    def test_within_unit_interval(self) -> None:
        ts = np.linspace(0.0, SECONDS_PER_YEAR, 100)
        for t in ts:
            m = soil_moisture_index(float(t), seed=3)
            assert 0.0 <= m <= 1.0

    def test_winter_wetter_than_summer(self) -> None:
        # Average a couple of weeks for noise smoothing
        winter = np.mean(
            [soil_moisture_index(t, seed=0) for t in np.linspace(0.0, 14 * SECONDS_PER_DAY, 14)]
        )
        summer = np.mean(
            [
                soil_moisture_index(t, seed=0)
                for t in np.linspace(180 * SECONDS_PER_DAY, 194 * SECONDS_PER_DAY, 14)
            ]
        )
        assert winter > summer


class TestWeatherSpec:
    def test_ambient_method_matches_module_function(self) -> None:
        spec = WeatherSpec(seed=99)
        t = 100_000.0
        a = spec.ambient_C(t)
        b = soil_ambient_C(t, seed=99)
        assert a == pytest.approx(b, abs=1e-12)

    def test_moisture_method_matches_module_function(self) -> None:
        spec = WeatherSpec(seed=42)
        t = 200_000.0
        m1 = spec.moisture(t)
        m2 = soil_moisture_index(t, seed=42)
        assert m1 == pytest.approx(m2, abs=1e-12)
