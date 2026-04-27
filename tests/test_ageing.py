"""Verification tests for Crine 2005 dielectric ageing kinetics."""

from __future__ import annotations

import numpy as np
import pytest

from gridforge.physics.ageing import (
    SECONDS_PER_YEAR,
    CrineParameters,
    cumulative_damage,
    damage_rate,
    life_at_constant_stress,
    remaining_useful_life,
    remaining_useful_life_years,
)


class TestCalibration:
    """Defaults must reproduce L_ref = 40 years at (E_ref, T_ref) by construction."""

    def test_life_at_reference_stress_is_calibration_value(self) -> None:
        params = CrineParameters()
        L = life_at_constant_stress(params.E_ref_V_per_m, params.T_ref_C, params)
        assert L == pytest.approx(params.L_ref_years * SECONDS_PER_YEAR, rel=1e-9)

    def test_life_at_default_xlpe_design_point_is_40_years(self) -> None:
        L = life_at_constant_stress(4.0e6, 90.0)
        assert L / SECONDS_PER_YEAR == pytest.approx(40.0, rel=1e-9)


class TestVoltageEnduranceLaw:
    """Field-dependence: doubling field must reduce life by 2^n."""

    def test_field_inverse_power_law(self) -> None:
        params = CrineParameters(n_voltage_endurance=11.0)
        L_ref = life_at_constant_stress(params.E_ref_V_per_m, params.T_ref_C, params)
        L_2x = life_at_constant_stress(2.0 * params.E_ref_V_per_m, params.T_ref_C, params)
        ratio = L_ref / L_2x
        assert ratio == pytest.approx(2.0**11.0, rel=1e-9)

    def test_lower_field_extends_life(self) -> None:
        params = CrineParameters()
        L_ref = life_at_constant_stress(params.E_ref_V_per_m, params.T_ref_C, params)
        L_low = life_at_constant_stress(0.5 * params.E_ref_V_per_m, params.T_ref_C, params)
        assert L_low > L_ref


class TestArrheniusLaw:
    """Temperature-dependence: lower T extends life; rule of thumb roughly
    factor 2 per 10 K reduction (Montsinger's rule generalised)."""

    def test_lower_temperature_extends_life(self) -> None:
        params = CrineParameters()
        L_ref = life_at_constant_stress(params.E_ref_V_per_m, params.T_ref_C, params)
        L_cold = life_at_constant_stress(params.E_ref_V_per_m, params.T_ref_C - 20.0, params)
        assert L_cold > L_ref

    def test_higher_temperature_shortens_life(self) -> None:
        params = CrineParameters()
        L_ref = life_at_constant_stress(params.E_ref_V_per_m, params.T_ref_C, params)
        L_hot = life_at_constant_stress(params.E_ref_V_per_m, params.T_ref_C + 10.0, params)
        assert L_hot < L_ref

    def test_montsinger_factor_in_expected_range(self) -> None:
        """For Phi ~ 1.1 eV at 90 C, a 10 K reduction should roughly halve
        the rate (life roughly doubles). Wide tolerance — material dependent."""
        params = CrineParameters()
        L_90 = life_at_constant_stress(params.E_ref_V_per_m, 90.0, params)
        L_80 = life_at_constant_stress(params.E_ref_V_per_m, 80.0, params)
        ratio = L_80 / L_90
        assert 1.5 < ratio < 4.0


class TestDamageRate:
    """damage_rate is the reciprocal of life at the same stress."""

    def test_rate_is_reciprocal_of_life(self) -> None:
        params = CrineParameters()
        L = life_at_constant_stress(params.E_ref_V_per_m, params.T_ref_C, params)
        r = damage_rate(params.E_ref_V_per_m, params.T_ref_C, params)
        assert r == pytest.approx(1.0 / L, rel=1e-9)

    def test_zero_field_rejected(self) -> None:
        with pytest.raises(ValueError):
            damage_rate(0.0, 90.0)


class TestCumulativeDamage:
    def test_constant_stress_gives_linear_damage(self) -> None:
        params = CrineParameters()
        L = life_at_constant_stress(params.E_ref_V_per_m, params.T_ref_C, params)
        # Run for half life under constant stress: D should be ~ 0.5
        N = 50
        times = np.linspace(0.0, 0.5 * L, N)
        E = np.full(N, params.E_ref_V_per_m)
        T = np.full(N, params.T_ref_C)
        D = cumulative_damage(times, E, T, params)
        assert D[0] == 0.0
        assert D[-1] == pytest.approx(0.5, rel=1e-6)

    def test_damage_monotone_non_decreasing(self) -> None:
        params = CrineParameters()
        N = 100
        times = np.linspace(0.0, 1.0e9, N)
        rng = np.random.default_rng(42)
        E = params.E_ref_V_per_m * rng.uniform(0.5, 1.5, N)
        T = 70.0 + rng.uniform(-10.0, 20.0, N)
        D = cumulative_damage(times, E, T, params)
        assert np.all(np.diff(D) >= -1e-12)

    def test_mismatched_input_lengths_rejected(self) -> None:
        params = CrineParameters()
        with pytest.raises(ValueError):
            cumulative_damage(
                np.array([0.0, 1.0, 2.0]),
                np.array([1.0e6, 2.0e6]),
                np.array([20.0, 30.0, 40.0]),
                params,
            )


class TestRemainingUsefulLife:
    def test_pristine_cable_has_full_life(self) -> None:
        rul_yr = remaining_useful_life_years(0.0, 4.0e6, 90.0)
        assert rul_yr == pytest.approx(40.0, rel=1e-9)

    def test_half_damaged_cable_has_half_life(self) -> None:
        rul_yr = remaining_useful_life_years(0.5, 4.0e6, 90.0)
        assert rul_yr == pytest.approx(20.0, rel=1e-9)

    def test_fully_damaged_cable_has_zero_life(self) -> None:
        rul = remaining_useful_life(1.0, 4.0e6, 90.0)
        assert rul == 0.0

    def test_lower_forward_stress_extends_rul(self) -> None:
        rul_high = remaining_useful_life_years(0.5, 4.0e6, 90.0)
        rul_low = remaining_useful_life_years(0.5, 4.0e6, 70.0)
        assert rul_low > rul_high

    def test_invalid_damage_rejected(self) -> None:
        with pytest.raises(ValueError):
            remaining_useful_life(-0.1, 4.0e6, 90.0)
        with pytest.raises(ValueError):
            remaining_useful_life(1.5, 4.0e6, 90.0)
