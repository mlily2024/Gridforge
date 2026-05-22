"""Tests for the soil moisture-to-resistivity coupling."""

from __future__ import annotations

import math

import numpy as np
import pytest

from gridforge.physics.soil_moisture import (
    CLAY,
    DEFAULT_SOIL,
    KNOWN_SOIL_TYPES,
    LOAM,
    SANDY,
    SoilType,
    theta_array_to_rho_t,
    theta_to_rho_t,
)


# ---------------------------------------------------------------------------
# Endpoint behaviour
# ---------------------------------------------------------------------------


class TestEndpoints:
    """rho_T must pass through (theta=0, rho_dry) and (theta=theta_sat, rho_sat)."""

    @pytest.mark.parametrize("soil", list(KNOWN_SOIL_TYPES.values()))
    def test_zero_moisture_returns_rho_dry(self, soil: SoilType) -> None:
        assert theta_to_rho_t(0.0, soil) == pytest.approx(soil.rho_dry_KmW, rel=1e-12)

    @pytest.mark.parametrize("soil", list(KNOWN_SOIL_TYPES.values()))
    def test_saturated_moisture_returns_rho_sat(self, soil: SoilType) -> None:
        assert theta_to_rho_t(soil.theta_sat, soil) == pytest.approx(
            soil.rho_sat_KmW, rel=1e-12
        )


# ---------------------------------------------------------------------------
# Monotonicity and physical realism
# ---------------------------------------------------------------------------


class TestMonotonicity:
    """rho_T should fall monotonically as theta rises."""

    @pytest.mark.parametrize("soil", list(KNOWN_SOIL_TYPES.values()))
    def test_strictly_decreasing(self, soil: SoilType) -> None:
        thetas = np.linspace(0.0, soil.theta_sat, 50)
        rhos = theta_array_to_rho_t(thetas, soil)
        diffs = np.diff(rhos)
        assert np.all(diffs < 0.0), "rho_T must strictly decrease as theta increases"

    @pytest.mark.parametrize("soil", list(KNOWN_SOIL_TYPES.values()))
    def test_bounded_between_endpoints(self, soil: SoilType) -> None:
        thetas = np.linspace(0.0, soil.theta_sat, 50)
        rhos = theta_array_to_rho_t(thetas, soil)
        assert rhos.min() >= soil.rho_sat_KmW - 1e-12
        assert rhos.max() <= soil.rho_dry_KmW + 1e-12


class TestUKTypicalCalibration:
    """At UK-typical theta = 0.3 the loam should land near 1 K.m/W (engineering norm)."""

    def test_loam_at_uk_typical_moisture(self) -> None:
        rho = theta_to_rho_t(0.3, LOAM)
        # Engineering reference: cable engineers commonly assume ~1.0 K.m/W
        # for "standard" buried-cable backfill at typical moisture. The
        # exponential interpolation gives ~0.88 here, well within the
        # operational +/- 30 % spread reported by Anders (1997) Ch. 7.
        assert 0.7 <= rho <= 1.2, f"expected ~1.0 K.m/W, got {rho:.3f}"

    def test_dry_summer_rho_exceeds_2_KmW(self) -> None:
        # A theta below ~0.05 corresponds to a UK dry-summer extreme;
        # rho_T must rise into the >2 K.m/W band documented in
        # Brakelmann (2004) and Anders (1997) Ch. 7.
        rho = theta_to_rho_t(0.05, LOAM)
        assert rho > 2.0, f"dry-summer loam should exceed 2 K.m/W, got {rho:.3f}"


# ---------------------------------------------------------------------------
# Input clamping (graceful, not strict)
# ---------------------------------------------------------------------------


class TestInputClamping:
    """Out-of-domain theta values clamp to the nearest endpoint."""

    def test_negative_theta_clamped_to_rho_dry(self) -> None:
        assert theta_to_rho_t(-0.1, LOAM) == pytest.approx(LOAM.rho_dry_KmW)

    def test_supersaturated_theta_clamped_to_rho_sat(self) -> None:
        assert theta_to_rho_t(2.0, LOAM) == pytest.approx(LOAM.rho_sat_KmW)

    def test_vector_clamping_works(self) -> None:
        thetas = np.array([-0.5, 0.0, 0.2, 0.45, 1.0])
        rhos = theta_array_to_rho_t(thetas, LOAM)
        # First and last must clamp to endpoints; middle values stay between
        assert rhos[0] == pytest.approx(LOAM.rho_dry_KmW)
        assert rhos[-1] == pytest.approx(LOAM.rho_sat_KmW)
        assert LOAM.rho_sat_KmW < rhos[2] < LOAM.rho_dry_KmW


# ---------------------------------------------------------------------------
# Soil-type lookup
# ---------------------------------------------------------------------------


class TestSoilLookup:
    """String-keyed soil lookup should resolve to the right SoilType."""

    @pytest.mark.parametrize("name", ["loam", "sandy", "clay"])
    def test_string_key_resolves(self, name: str) -> None:
        # Same theta, same answer via string OR via the SoilType instance.
        soil = KNOWN_SOIL_TYPES[name]
        assert theta_to_rho_t(0.2, name) == pytest.approx(
            theta_to_rho_t(0.2, soil), rel=1e-12
        )

    def test_unknown_string_raises(self) -> None:
        with pytest.raises(ValueError, match="unknown soil type"):
            theta_to_rho_t(0.2, "siltstone")

    def test_default_is_loam(self) -> None:
        # Caller passes no soil — should be LOAM.
        assert theta_to_rho_t(0.2) == pytest.approx(theta_to_rho_t(0.2, LOAM))
        assert DEFAULT_SOIL is LOAM


# ---------------------------------------------------------------------------
# Vector / scalar consistency
# ---------------------------------------------------------------------------


class TestVectorScalarConsistency:
    """Vectorised path must agree with scalar path element-for-element."""

    def test_random_grid_agreement(self) -> None:
        rng = np.random.default_rng(seed=42)
        thetas = rng.uniform(-0.1, 0.6, size=200)
        vec = theta_array_to_rho_t(thetas, LOAM)
        scalar = np.array([theta_to_rho_t(float(t), LOAM) for t in thetas])
        assert np.allclose(vec, scalar, atol=1e-12)


# ---------------------------------------------------------------------------
# Cross-soil sanity
# ---------------------------------------------------------------------------


class TestCrossSoil:
    """Sanity check on the relative ordering of soils at low moisture."""

    def test_dry_soil_ordering(self) -> None:
        # Near-dry: rho_dry dominates the answer, so the ordering follows
        # the rho_dry constants directly: sandy > loam > clay.
        # (Above ~theta=0.05 the curves can cross because sandy saturates
        # faster — smaller theta_sat — and that crossover is physical.)
        theta = 0.03
        rho_sandy = theta_to_rho_t(theta, SANDY)
        rho_loam = theta_to_rho_t(theta, LOAM)
        rho_clay = theta_to_rho_t(theta, CLAY)
        assert rho_sandy > rho_loam > rho_clay, (
            f"at theta=0.03 expected sandy ({rho_sandy:.2f}) > "
            f"loam ({rho_loam:.2f}) > clay ({rho_clay:.2f})"
        )

    def test_all_finite_across_full_range(self) -> None:
        """Smoke test: no NaN / inf for any reasonable input."""
        thetas = np.linspace(-0.5, 2.0, 100)
        for soil in KNOWN_SOIL_TYPES.values():
            rhos = theta_array_to_rho_t(thetas, soil)
            assert np.all(np.isfinite(rhos)), f"{soil.name}: non-finite output"
