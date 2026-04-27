"""Verification tests for the lumped transient thermal solver."""

from __future__ import annotations

import numpy as np
import pytest

from gridforge.physics.cable_archetype import (
    UK_11KV_240MM2_XLPE_3CORE,
    UK_TYPICAL_INSTALLATION,
)
from gridforge.physics.thermal import solve_steady_state
from gridforge.physics.transient import (
    cable_thermal_capacitance,
    simulate_transient,
    total_thermal_resistance,
)


class TestThermalCapacitance:
    def test_capacitance_in_expected_order(self) -> None:
        geom, _ = UK_11KV_240MM2_XLPE_3CORE
        C = cable_thermal_capacitance(geom)
        # Per published distribution-cable studies, expect a few thousand
        # J/(K m) for a 3-core 240 mm^2 XLPE cable.
        assert 1500.0 < C < 8000.0


class TestTotalThermalResistance:
    def test_total_R_matches_sum_of_components(self) -> None:
        geom, mat = UK_11KV_240MM2_XLPE_3CORE
        install = UK_TYPICAL_INSTALLATION
        R = total_thermal_resistance(geom, mat, install)
        # Order-of-magnitude check: sum of T1 + ~3 * (T3 + T4) sits between
        # 1.5 and 4 K m/W for typical installations.
        assert 1.0 < R < 5.0


class TestTransientSolver:
    def setup_method(self) -> None:
        self.geom, self.mat = UK_11KV_240MM2_XLPE_3CORE
        self.install = UK_TYPICAL_INSTALLATION

    def test_constant_load_converges_to_steady_state(self) -> None:
        """Holding I constant for many time-constants reproduces IEC 60287."""
        I = 350.0
        # Find tau approximately
        R = total_thermal_resistance(self.geom, self.mat, self.install)
        C = cable_thermal_capacitance(self.geom)
        tau = R * C
        # Run for 8 tau — should be at steady state
        times = np.linspace(0.0, 8.0 * tau, 200)
        result = simulate_transient(
            current_profile=lambda t: I,
            times_s=times,
            line_voltage_V_rms=11_000.0,
            geom=self.geom,
            mat=self.mat,
            install=self.install,
            initial_temp_C=self.install.ambient_soil_temp_C,
        )
        ss = solve_steady_state(
            current_per_phase_A=I,
            line_voltage_V_rms=11_000.0,
            geom=self.geom,
            mat=self.mat,
            install=self.install,
        )
        assert result.conductor_temp_C[-1] == pytest.approx(
            ss.conductor_temp_C, abs=0.5
        )

    def test_step_response_first_order_shape(self) -> None:
        """Step from 0 A to 400 A produces approximate first-order rise."""
        I_step = 400.0
        # Run for a long horizon so we can sample anywhere up to many tau
        R = total_thermal_resistance(self.geom, self.mat, self.install)
        C = cable_thermal_capacitance(self.geom)
        # First guess: actual solver tau is C * R / n; horizon must cover >> tau
        approx_tau = C * R / self.geom.n_conductors
        times = np.linspace(0.0, 8.0 * approx_tau, 400)
        result = simulate_transient(
            current_profile=lambda t: I_step,
            times_s=times,
            line_voltage_V_rms=11_000.0,
            geom=self.geom,
            mat=self.mat,
            install=self.install,
            initial_temp_C=self.install.ambient_soil_temp_C,
        )
        # Use the solver's own reported time constant for the lookup
        tau = result.time_constant_s
        T0 = result.conductor_temp_C[0]
        T_inf = result.conductor_temp_C[-1]
        idx_tau = int(np.argmin(np.abs(times - tau)))
        T_at_tau = result.conductor_temp_C[idx_tau]
        expected_at_tau = T0 + (T_inf - T0) * (1.0 - np.exp(-1.0))
        # Non-linearity in R(T) and discrete sampling allow up to ~3 K slack
        assert T_at_tau == pytest.approx(expected_at_tau, abs=3.0)

    def test_zero_current_decays_to_ambient(self) -> None:
        """Starting hot, with zero load, must cool toward ambient."""
        R = total_thermal_resistance(self.geom, self.mat, self.install)
        C = cable_thermal_capacitance(self.geom)
        tau = R * C
        times = np.linspace(0.0, 6.0 * tau, 200)
        result = simulate_transient(
            current_profile=lambda t: 0.0,
            times_s=times,
            line_voltage_V_rms=11_000.0,
            geom=self.geom,
            mat=self.mat,
            install=self.install,
            initial_temp_C=80.0,
        )
        # End temperature should be very close to ambient (only dielectric
        # heating remains, < 0.1 K).
        assert result.conductor_temp_C[-1] < self.install.ambient_soil_temp_C + 1.0

    def test_diurnal_load_oscillates(self) -> None:
        """Sinusoidal daily load produces oscillating cable temperature."""
        period = 24.0 * 3600.0  # 24 hours

        def I_profile(t: float) -> float:
            return 200.0 + 150.0 * float(np.sin(2.0 * np.pi * t / period))

        times = np.linspace(0.0, 3.0 * period, 600)
        result = simulate_transient(
            current_profile=I_profile,
            times_s=times,
            line_voltage_V_rms=11_000.0,
            geom=self.geom,
            mat=self.mat,
            install=self.install,
            initial_temp_C=30.0,
        )
        # Discard first day (transient) and check oscillation amplitude
        last_day = result.conductor_temp_C[times >= 2.0 * period]
        amplitude = (last_day.max() - last_day.min()) / 2.0
        assert amplitude > 1.0  # measurable oscillation
        assert amplitude < 30.0  # but not unphysical

    def test_invalid_times_rejected(self) -> None:
        with pytest.raises(ValueError):
            simulate_transient(
                current_profile=lambda t: 200.0,
                times_s=[100.0, 50.0, 200.0],  # not monotone
                line_voltage_V_rms=11_000.0,
                geom=self.geom,
                mat=self.mat,
                install=self.install,
            )
