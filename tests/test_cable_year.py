"""Verification tests for the cable-year simulator."""

from __future__ import annotations

import numpy as np
import pytest

from gridforge.data.cable_year import CableYearSpec, simulate_cable_year
from gridforge.data.conditions import (
    AcceleratedDielectricMode,
    HealthyMode,
    ThermalAgeingMode,
    WaterIngressMode,
)
from gridforge.data.load_profiles import LoadSpec
from gridforge.data.weather import WeatherSpec


def _make_spec(condition=None, duration_years=0.1, seed=0) -> CableYearSpec:
    return CableYearSpec(
        cable_id=f"test_{seed}",
        duration_years=duration_years,
        load=LoadSpec("residential", peak_A=350.0, base_A=80.0, seed=seed),
        weather=WeatherSpec(seed=seed),
        condition=condition if condition is not None else HealthyMode(),
    )


class TestSimulatorBasics:
    def test_returns_correct_shape(self) -> None:
        spec = _make_spec(duration_years=0.05)  # ~ 18 days hourly
        result = simulate_cable_year(spec)
        n = len(result.times_s)
        assert n > 0
        for arr in (
            result.current_A,
            result.ambient_C,
            result.moisture,
            result.conductor_C,
            result.e_field_V_per_m,
            result.pd_rate_relative,
            result.cumulative_damage,
        ):
            assert len(arr) == n

    def test_times_strictly_increasing(self) -> None:
        result = simulate_cable_year(_make_spec(duration_years=0.05))
        assert np.all(np.diff(result.times_s) > 0)

    def test_damage_starts_at_zero_and_monotone(self) -> None:
        result = simulate_cable_year(_make_spec(duration_years=0.05))
        assert result.cumulative_damage[0] == 0.0
        assert np.all(np.diff(result.cumulative_damage) >= -1e-12)

    def test_conductor_temperature_is_finite(self) -> None:
        result = simulate_cable_year(_make_spec(duration_years=0.05))
        assert np.all(np.isfinite(result.conductor_C))

    def test_conductor_above_ambient_when_loaded(self) -> None:
        result = simulate_cable_year(_make_spec(duration_years=0.05))
        # At peak load periods conductor should sit above ambient
        rises = result.conductor_C - result.ambient_C
        assert rises.max() > 1.0


class TestDeterminism:
    def test_same_seed_same_output(self) -> None:
        a = simulate_cable_year(_make_spec(duration_years=0.05, seed=42))
        b = simulate_cable_year(_make_spec(duration_years=0.05, seed=42))
        assert np.array_equal(a.conductor_C, b.conductor_C)
        assert np.array_equal(a.cumulative_damage, b.cumulative_damage)

    def test_different_seeds_different_output(self) -> None:
        a = simulate_cable_year(_make_spec(duration_years=0.05, seed=1))
        b = simulate_cable_year(_make_spec(duration_years=0.05, seed=2))
        # Currents differ → temperatures differ
        assert not np.array_equal(a.current_A, b.current_A)


class TestConditionModeEffects:
    """Different condition modes must produce visibly different damage trajectories
    over a multi-year horizon."""

    def test_thermal_ageing_more_damaging_than_healthy(self) -> None:
        healthy = simulate_cable_year(_make_spec(condition=HealthyMode(), duration_years=1.0))
        ageing = simulate_cable_year(
            _make_spec(
                condition=ThermalAgeingMode(overheat_offset_C=30.0),
                duration_years=1.0,
            )
        )
        assert ageing.cumulative_damage[-1] > healthy.cumulative_damage[-1]

    def test_water_ingress_field_grows(self) -> None:
        spec = _make_spec(
            condition=WaterIngressMode(onset_year=0.0, saturation_year=2.0),
            duration_years=1.0,
        )
        result = simulate_cable_year(spec)
        # Field should rise across the year for this mode
        assert result.e_field_V_per_m[-1] > result.e_field_V_per_m[0]

    def test_accelerated_dielectric_has_field_excursions(self) -> None:
        spec = _make_spec(
            condition=AcceleratedDielectricMode(impulse_per_year=400.0, seed=0),
            duration_years=1.0,
        )
        result = simulate_cable_year(spec)
        # Some hours must have field above the baseline
        baseline = float(np.median(result.e_field_V_per_m))
        assert (result.e_field_V_per_m > 1.5 * baseline).sum() > 100


class TestSpecValidation:
    def test_too_short_duration_raises(self) -> None:
        spec = CableYearSpec(
            cable_id="too_short",
            duration_years=1e-6,  # under one hour
            load=LoadSpec("residential", 300.0, 80.0),
            weather=WeatherSpec(),
        )
        with pytest.raises(ValueError):
            simulate_cable_year(spec)
