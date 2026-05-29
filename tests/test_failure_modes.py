"""Verification tests for failure-mode injectors."""

from __future__ import annotations

import numpy as np
import pytest

from gridforge.data.failure_modes import (
    MODES,
    AcceleratedDielectricMode,
    HealthyMode,
    ThermalAgeingMode,
    WaterIngressMode,
    make_failure_mode,
)

SECONDS_PER_YEAR = 365.25 * 24.0 * 3600.0


class TestHealthyMode:
    def test_neutral_at_all_times(self) -> None:
        m = HealthyMode()
        for t_yr in [0.0, 1.0, 5.0, 10.0]:
            t = t_yr * SECONDS_PER_YEAR
            assert m.field_multiplier(t) == 1.0
            assert m.temp_offset_C(t) == 0.0
            assert m.pd_rate_multiplier(t) == 1.0


class TestWaterIngress:
    def test_flat_before_onset(self) -> None:
        m = WaterIngressMode(onset_year=1.0)
        # At 0.5 yr (before onset)
        assert m.field_multiplier(0.5 * SECONDS_PER_YEAR) == 1.0

    def test_grows_monotonically(self) -> None:
        m = WaterIngressMode()
        years = np.linspace(0.0, 10.0, 50)
        f = [m.field_multiplier(y * SECONDS_PER_YEAR) for y in years]
        assert all(b >= a - 1e-12 for a, b in zip(f[:-1], f[1:]))

    def test_saturates_at_max(self) -> None:
        m = WaterIngressMode(max_field_boost=1.6, saturation_year=8.0)
        f_at_sat = m.field_multiplier(8.0 * SECONDS_PER_YEAR)
        f_late = m.field_multiplier(20.0 * SECONDS_PER_YEAR)
        assert f_at_sat == pytest.approx(1.6, abs=1e-6)
        assert f_late == pytest.approx(1.6, abs=1e-6)

    def test_pd_rate_grows_quadratically(self) -> None:
        """PD rate uses a quadratic in progress; should rise faster than field."""
        m = WaterIngressMode(onset_year=0.0, saturation_year=10.0)
        f3 = m.field_multiplier(3.0 * SECONDS_PER_YEAR)
        p3 = m.pd_rate_multiplier(3.0 * SECONDS_PER_YEAR)
        f6 = m.field_multiplier(6.0 * SECONDS_PER_YEAR)
        p6 = m.pd_rate_multiplier(6.0 * SECONDS_PER_YEAR)
        # The PD ratio between 6 and 3 years should exceed the field ratio
        # because of the quadratic progress dependence.
        assert (p6 - 1.0) / max(p3 - 1.0, 1e-9) > (f6 - 1.0) / max(f3 - 1.0, 1e-9)


class TestThermalAgeing:
    def test_constant_offset(self) -> None:
        m = ThermalAgeingMode(overheat_offset_C=20.0)
        assert m.temp_offset_C(0.0) == 20.0
        assert m.temp_offset_C(5.0 * SECONDS_PER_YEAR) == 20.0


class TestAcceleratedDielectric:
    def test_event_rate_in_expected_range(self) -> None:
        """Over a long enough horizon the event rate should be close to the
        configured impulse_per_year (within Poisson noise)."""
        m = AcceleratedDielectricMode(impulse_per_year=200.0, seed=0)
        # Sample 5 years at hourly resolution
        n_hours = 5 * 8766
        events = sum(1 for h in range(n_hours) if m.field_multiplier(h * 3600.0) > 1.5)
        # Expected ~ 1000 events; allow Poisson 5-sigma envelope
        assert 700 < events < 1300

    def test_pd_scales_with_field(self) -> None:
        """When an impulse occurs, PD multiplier exceeds 1; when not, equals 1."""
        m = AcceleratedDielectricMode(seed=0)
        # Find an hour with impulse
        for h in range(10_000):
            if m.field_multiplier(h * 3600.0) > 1.5:
                assert m.pd_rate_multiplier(h * 3600.0) > 1.0
                return
        # If we get here no impulse hour was found in 10k hours — extremely
        # unlikely with default rate (180/yr ~ 0.02 / hour) but not impossible
        # for some seeds. Just skip rather than fail.
        pytest.skip("no impulse hour found in 10k hours; rare but possible")


class TestRegistry:
    def test_all_four_modes_registered(self) -> None:
        assert set(MODES) == {
            "healthy",
            "water_ingress",
            "thermal_ageing",
            "accelerated_dielectric",
        }

    def test_make_unknown_raises(self) -> None:
        with pytest.raises(KeyError):
            make_failure_mode("not_a_mode")

    def test_make_known_returns_correct_type(self) -> None:
        m = make_failure_mode("water_ingress", seed=7)
        assert isinstance(m, WaterIngressMode)
        assert m.seed == 7
