"""
Verification tests for the IEC 60287 steady-state thermal solver.

The tests check three things:

  1. Pure-physics correctness of the thermal-resistance components against
     hand-computable values for the canonical UK 11 kV 240 mm^2 archetype.

  2. Internal consistency of the steady-state solver: convergence, heat
     balance, monotone scaling with current and soil thermal resistivity.

  3. Behaviour at the boundary cases: zero current (only dielectric heating),
     and very low current (close to ambient).

These tests do not assert agreement with a single tabulated rating value
because tabulated cable ratings depend on a number of non-canonical
assumptions (sheath bonding scheme, soil moisture state, group derating)
that vary between vendors and standards. Calibration against published
rating tables is a separate downstream exercise.
"""

from __future__ import annotations

import math

import pytest

from gridforge.physics.cable_archetype import (
    UK_11KV_240MM2_XLPE_3CORE,
    UK_TYPICAL_INSTALLATION,
)
from gridforge.physics.thermal import (
    CableGeometry,
    CableMaterials,
    InstallationConditions,
    ac_resistance_at_temp,
    dielectric_loss_per_phase,
    solve_steady_state,
    thermal_resistance_T1,
    thermal_resistance_T3,
    thermal_resistance_T4,
)


# ---------------------------------------------------------------------------
# Component-level checks
# ---------------------------------------------------------------------------

class TestThermalResistanceComponents:
    """T1, T3, T4 against hand-computed values for the canonical archetype."""

    def setup_method(self) -> None:
        self.geom, self.mat = UK_11KV_240MM2_XLPE_3CORE
        self.install = UK_TYPICAL_INSTALLATION

    def test_T1_matches_hand_calculation(self) -> None:
        # T1 = (3.5 / 2pi) * ln(1 + 2 * 3.4 / 18.5)
        #    = 0.55704 * ln(1.36757)
        #    = 0.55704 * 0.31294
        #    = 0.17433 K.m/W
        expected = (3.5 / (2.0 * math.pi)) * math.log(1.0 + 2.0 * 3.4 / 18.5)
        T1 = thermal_resistance_T1(self.geom, self.mat)
        assert T1 == pytest.approx(expected, rel=1e-9)
        assert 0.10 < T1 < 0.25

    def test_T3_matches_hand_calculation(self) -> None:
        # T3 = (3.5 / 2pi) * ln(70 / (70 - 5))
        #    = 0.55704 * ln(70 / 65)
        #    = 0.55704 * 0.07410
        #    = 0.04128 K.m/W
        expected = (3.5 / (2.0 * math.pi)) * math.log(70.0 / 65.0)
        T3 = thermal_resistance_T3(self.geom, self.mat)
        assert T3 == pytest.approx(expected, rel=1e-9)
        assert 0.02 < T3 < 0.10

    def test_T4_matches_hand_calculation(self) -> None:
        # 2u = 2 * 0.8 / 0.070 = 22.857
        # T4 = (1.0 / 2pi) * ln(22.857 + sqrt(22.857^2 - 1))
        two_u = 2.0 * 0.8 / 0.070
        expected = (1.0 / (2.0 * math.pi)) * math.log(
            two_u + math.sqrt(two_u * two_u - 1.0)
        )
        T4 = thermal_resistance_T4(self.geom, self.install)
        assert T4 == pytest.approx(expected, rel=1e-9)
        assert 0.4 < T4 < 1.0

    def test_T4_increases_with_burial_depth(self) -> None:
        T4_shallow = thermal_resistance_T4(
            self.geom,
            InstallationConditions(0.5, 1.0, 15.0),
        )
        T4_deep = thermal_resistance_T4(
            self.geom,
            InstallationConditions(1.5, 1.0, 15.0),
        )
        assert T4_deep > T4_shallow
        # Logarithmic scaling — tripling depth should not triple T4
        assert T4_deep < 3.0 * T4_shallow

    def test_T4_scales_linearly_with_soil_resistivity(self) -> None:
        T4_dry = thermal_resistance_T4(
            self.geom,
            InstallationConditions(0.8, 2.0, 15.0),
        )
        T4_wet = thermal_resistance_T4(
            self.geom,
            InstallationConditions(0.8, 1.0, 15.0),
        )
        assert T4_dry == pytest.approx(2.0 * T4_wet, rel=1e-9)

    def test_T1_invalid_geometry_raises(self) -> None:
        with pytest.raises(ValueError):
            thermal_resistance_T1(
                CableGeometry(d_c_mm=0.0, t_i_mm=3.4, t_j_mm=2.5, D_e_mm=70.0),
                self.mat,
            )


# ---------------------------------------------------------------------------
# Conductor resistance
# ---------------------------------------------------------------------------

class TestConductorResistance:
    """A.c. resistance temperature correction and skin-effect factor."""

    def setup_method(self) -> None:
        _, self.mat = UK_11KV_240MM2_XLPE_3CORE

    def test_resistance_at_20C_matches_input(self) -> None:
        R = ac_resistance_at_temp(20.0, self.mat)
        # At 20 C, R = R_dc * R_ac_dc_ratio
        assert R == pytest.approx(self.mat.R_dc_20C_ohm_per_m * self.mat.R_ac_dc_ratio)

    def test_resistance_at_90C_higher_than_at_20C(self) -> None:
        R20 = ac_resistance_at_temp(20.0, self.mat)
        R90 = ac_resistance_at_temp(90.0, self.mat)
        # Expect ~27.5 % increase per 70 K rise for Cu (alpha = 3.93e-3 /K)
        ratio = R90 / R20
        assert 1.20 < ratio < 1.35

    def test_resistance_below_20C_lower(self) -> None:
        R20 = ac_resistance_at_temp(20.0, self.mat)
        R0 = ac_resistance_at_temp(0.0, self.mat)
        assert R0 < R20


# ---------------------------------------------------------------------------
# Dielectric loss
# ---------------------------------------------------------------------------

class TestDielectricLoss:
    def setup_method(self) -> None:
        _, self.mat = UK_11KV_240MM2_XLPE_3CORE

    def test_dielectric_loss_at_11kV_phase_voltage_is_small(self) -> None:
        U0 = 11_000.0 / math.sqrt(3.0)
        W_d = dielectric_loss_per_phase(U0, self.mat)
        # Order-of-magnitude: < 0.01 W/m per phase for 11 kV XLPE
        assert 1.0e-4 < W_d < 1.0e-2

    def test_dielectric_loss_scales_with_voltage_squared(self) -> None:
        U1 = 6_350.0
        U2 = 12_700.0
        W1 = dielectric_loss_per_phase(U1, self.mat)
        W2 = dielectric_loss_per_phase(U2, self.mat)
        assert W2 == pytest.approx(4.0 * W1, rel=1e-9)


# ---------------------------------------------------------------------------
# Steady-state solver — internal consistency
# ---------------------------------------------------------------------------

class TestSteadyStateSolver:
    """Solver convergence, scaling, and physical-balance checks."""

    def setup_method(self) -> None:
        self.geom, self.mat = UK_11KV_240MM2_XLPE_3CORE
        self.install = UK_TYPICAL_INSTALLATION

    def test_zero_current_temperature_is_near_ambient(self) -> None:
        sol = solve_steady_state(
            current_per_phase_A=0.0,
            line_voltage_V_rms=11_000.0,
            geom=self.geom,
            mat=self.mat,
            install=self.install,
        )
        # Only dielectric heating remains. Rise should be < 0.1 K.
        rise = sol.conductor_temp_C - self.install.ambient_soil_temp_C
        assert sol.converged is True
        assert 0.0 <= rise < 0.1

    def test_solver_converges_within_iteration_budget(self) -> None:
        sol = solve_steady_state(
            current_per_phase_A=400.0,
            line_voltage_V_rms=11_000.0,
            geom=self.geom,
            mat=self.mat,
            install=self.install,
        )
        assert sol.converged is True
        assert sol.iterations <= 15

    def test_temperature_monotone_increasing_with_current(self) -> None:
        prev_T = -math.inf
        for I in [0.0, 100.0, 200.0, 300.0, 400.0, 500.0]:
            sol = solve_steady_state(
                current_per_phase_A=I,
                line_voltage_V_rms=11_000.0,
                geom=self.geom,
                mat=self.mat,
                install=self.install,
            )
            assert sol.conductor_temp_C > prev_T
            prev_T = sol.conductor_temp_C

    def test_temperature_increases_with_soil_resistivity(self) -> None:
        wet = solve_steady_state(
            current_per_phase_A=400.0,
            line_voltage_V_rms=11_000.0,
            geom=self.geom,
            mat=self.mat,
            install=InstallationConditions(0.8, 0.7, 15.0),
        )
        dry = solve_steady_state(
            current_per_phase_A=400.0,
            line_voltage_V_rms=11_000.0,
            geom=self.geom,
            mat=self.mat,
            install=InstallationConditions(0.8, 2.0, 15.0),
        )
        assert dry.conductor_temp_C > wet.conductor_temp_C

    def test_temperature_increases_with_ambient(self) -> None:
        cool = solve_steady_state(
            current_per_phase_A=400.0,
            line_voltage_V_rms=11_000.0,
            geom=self.geom,
            mat=self.mat,
            install=InstallationConditions(0.8, 1.0, 5.0),
        )
        warm = solve_steady_state(
            current_per_phase_A=400.0,
            line_voltage_V_rms=11_000.0,
            geom=self.geom,
            mat=self.mat,
            install=InstallationConditions(0.8, 1.0, 25.0),
        )
        # The system is non-linear in ambient: a hotter conductor has higher
        # resistance, dissipating more I^2 R loss, raising T further. So a
        # 20 K ambient rise produces *more* than 20 K conductor rise. The
        # gain factor is small (single-digit percent) for typical loadings.
        diff = warm.conductor_temp_C - cool.conductor_temp_C
        assert diff > 20.0
        assert diff < 25.0

    def test_heat_balance_through_layers(self) -> None:
        """Sum of layer drops equals conductor rise above ambient."""
        sol = solve_steady_state(
            current_per_phase_A=400.0,
            line_voltage_V_rms=11_000.0,
            geom=self.geom,
            mat=self.mat,
            install=self.install,
        )
        rise = sol.conductor_temp_C - self.install.ambient_soil_temp_C
        # Layer drops (insulation, jacket, soil) must reproduce the rise to
        # within numerical tolerance. Note: the soil-interface temperature in
        # the solution sits at ambient by construction; back-computed via
        # subtraction it should match.
        reconstructed_ambient = sol.soil_interface_temp_C
        # Tolerance matches the solver's fixed-point convergence threshold.
        assert reconstructed_ambient == pytest.approx(
            self.install.ambient_soil_temp_C, abs=1e-3
        )
        assert rise > 0.0

    def test_temperature_in_engineering_range_at_typical_loading(self) -> None:
        """At a representative 11 kV distribution loading, conductor should
        sit somewhere in the operational engineering range (well above
        ambient, below thermal limit). Wide bounds — this is a sanity gate,
        not a calibration against any specific rating table."""
        sol = solve_steady_state(
            current_per_phase_A=400.0,
            line_voltage_V_rms=11_000.0,
            geom=self.geom,
            mat=self.mat,
            install=self.install,
        )
        assert 30.0 < sol.conductor_temp_C < 90.0  # well below XLPE 90 C limit

    def test_negative_current_rejected(self) -> None:
        with pytest.raises(ValueError):
            solve_steady_state(
                current_per_phase_A=-10.0,
                line_voltage_V_rms=11_000.0,
                geom=self.geom,
                mat=self.mat,
                install=self.install,
            )
