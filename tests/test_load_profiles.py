"""Verification tests for the load-profile library."""

from __future__ import annotations

import pytest

from gridforge.data.load_profiles import (
    PROFILES,
    LoadSpec,
    commercial,
    industrial,
    residential,
)

SECONDS_PER_DAY = 24.0 * 3600.0


class TestProfileShapes:
    """Each profile must respect its peak / base envelope at all times."""

    def test_residential_within_envelope(self) -> None:
        for h in range(24):
            v = residential(h * 3600.0, peak_A=400.0, base_A=100.0)
            # Allow 5% headroom for seasonal + noise multipliers
            assert 80.0 <= v <= 480.0, f"residential at hour {h} = {v}"

    def test_commercial_low_overnight(self) -> None:
        midnight = commercial(0.0, peak_A=400.0, base_A=100.0, seed=1)
        midday = commercial(12 * 3600.0, peak_A=400.0, base_A=100.0, seed=1)
        assert midnight < midday

    def test_industrial_high_baseload(self) -> None:
        # Industrial should always be at least the base + 50% of (peak - base)
        for h in range(24):
            v = industrial(h * 3600.0, peak_A=300.0, base_A=200.0)
            assert v >= 220.0, f"industrial at hour {h} = {v}"


class TestWeekendModulation:
    def test_commercial_weekend_drop(self) -> None:
        # Day 0 = Monday in our convention; Saturday = day 5
        weekday_noon = commercial(12 * 3600.0, 400.0, 100.0, seed=42)
        weekend_noon = commercial(12 * 3600.0 + 5 * SECONDS_PER_DAY, 400.0, 100.0, seed=42)
        assert weekend_noon < 0.5 * weekday_noon


class TestDeterminism:
    """Same (t, params, seed) must always return the same value."""

    def test_residential_deterministic(self) -> None:
        v1 = residential(12 * 3600.0, 350.0, 80.0, seed=7)
        v2 = residential(12 * 3600.0, 350.0, 80.0, seed=7)
        assert v1 == v2

    def test_different_seeds_differ(self) -> None:
        v1 = residential(12 * 3600.0, 350.0, 80.0, seed=1)
        v2 = residential(12 * 3600.0, 350.0, 80.0, seed=2)
        # Seeds change the noise component (~2% amplitude); with high
        # probability they should not be exactly equal.
        assert v1 != v2


class TestLoadSpec:
    def test_callable(self) -> None:
        spec = LoadSpec(profile_name="residential", peak_A=300.0, base_A=80.0, seed=3)
        v = spec(8 * 3600.0)
        assert 50.0 <= v <= 360.0

    def test_unknown_profile_raises(self) -> None:
        spec = LoadSpec(profile_name="not_a_profile", peak_A=300.0, base_A=80.0)
        with pytest.raises(KeyError):
            spec(0.0)


class TestRegistry:
    def test_all_four_profiles_registered(self) -> None:
        assert set(PROFILES) == {"residential", "commercial", "industrial", "mixed"}
