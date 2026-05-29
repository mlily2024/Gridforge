"""Smoke tests for the four UK 11 kV archetypes.

Each archetype must:
  - resolve via the ARCHETYPES registry
  - produce a converged steady-state solve at a representative loading
  - keep the conductor in the engineering range (above ambient, below the
    XLPE thermal limit at moderate loading)
"""

from __future__ import annotations

import pytest

from gridforge.physics.cable_archetype import (
    ARCHETYPES,
    UK_11KV_95MM2_CU_XLPE_3CORE,
    UK_11KV_240MM2_CU_PILC_3CORE,
    UK_11KV_240MM2_XLPE_3CORE,
    UK_11KV_300MM2_CU_XLPE_1CORE,
    UK_TYPICAL_INSTALLATION,
    archetype_by_name,
)
from gridforge.physics.thermal import solve_steady_state

REPRESENTATIVE_CURRENTS = {
    "11kV_240mm2_Cu_XLPE_3c": 350.0,
    "11kV_95mm2_Cu_XLPE_3c": 180.0,  # smaller cable, lower nameplate
    "11kV_300mm2_Cu_XLPE_1c": 250.0,  # single-core sees one-phase loss
    "11kV_240mm2_Cu_PILC_3c": 300.0,  # PILC slightly derated vs XLPE
}


class TestRegistry:
    def test_all_four_archetypes_registered(self) -> None:
        expected = {
            "11kV_240mm2_Cu_XLPE_3c",
            "11kV_95mm2_Cu_XLPE_3c",
            "11kV_300mm2_Cu_XLPE_1c",
            "11kV_240mm2_Cu_PILC_3c",
        }
        assert set(ARCHETYPES) == expected

    def test_archetype_by_name_returns_tuple(self) -> None:
        for name in ARCHETYPES:
            geom, mat = archetype_by_name(name)
            assert geom is not None
            assert mat is not None

    def test_unknown_name_raises(self) -> None:
        with pytest.raises(KeyError):
            archetype_by_name("not_an_archetype")


class TestEachArchetype:
    def _check(self, geom, mat, current_A: float, name: str) -> None:
        sol = solve_steady_state(
            current_per_phase_A=current_A,
            line_voltage_V_rms=11_000.0,
            geom=geom,
            mat=mat,
            install=UK_TYPICAL_INSTALLATION,
        )
        assert sol.converged is True, f"{name} did not converge"
        # Conductor must be above ambient and below the XLPE thermal limit
        # under representative loading.
        assert sol.conductor_temp_C > UK_TYPICAL_INSTALLATION.ambient_soil_temp_C, name
        assert sol.conductor_temp_C < 95.0, f"{name} too hot: {sol.conductor_temp_C}"

    def test_240mm2_XLPE_3c(self) -> None:
        geom, mat = UK_11KV_240MM2_XLPE_3CORE
        self._check(geom, mat, REPRESENTATIVE_CURRENTS["11kV_240mm2_Cu_XLPE_3c"], "240mm2_XLPE_3c")

    def test_95mm2_XLPE_3c(self) -> None:
        geom, mat = UK_11KV_95MM2_CU_XLPE_3CORE
        self._check(geom, mat, REPRESENTATIVE_CURRENTS["11kV_95mm2_Cu_XLPE_3c"], "95mm2_XLPE_3c")

    def test_300mm2_XLPE_1c(self) -> None:
        geom, mat = UK_11KV_300MM2_CU_XLPE_1CORE
        self._check(geom, mat, REPRESENTATIVE_CURRENTS["11kV_300mm2_Cu_XLPE_1c"], "300mm2_XLPE_1c")

    def test_240mm2_PILC_3c(self) -> None:
        geom, mat = UK_11KV_240MM2_CU_PILC_3CORE
        self._check(geom, mat, REPRESENTATIVE_CURRENTS["11kV_240mm2_Cu_PILC_3c"], "240mm2_PILC_3c")


class TestArchetypeDifferentiation:
    """Different archetypes at the same current should produce different
    conductor temperatures — proves the parameterisation works."""

    def test_smaller_cable_runs_hotter_at_same_current(self) -> None:
        I = 200.0
        big_geom, big_mat = UK_11KV_240MM2_XLPE_3CORE
        small_geom, small_mat = UK_11KV_95MM2_CU_XLPE_3CORE
        big = solve_steady_state(I, 11_000.0, big_geom, big_mat, UK_TYPICAL_INSTALLATION)
        small = solve_steady_state(I, 11_000.0, small_geom, small_mat, UK_TYPICAL_INSTALLATION)
        assert small.conductor_temp_C > big.conductor_temp_C

    def test_PILC_warmer_than_XLPE_at_same_current(self) -> None:
        """Higher insulation thermal resistivity (paper 5.0 vs XLPE 3.5)
        means PILC runs hotter at the same loading."""
        I = 250.0
        xlpe_geom, xlpe_mat = UK_11KV_240MM2_XLPE_3CORE
        pilc_geom, pilc_mat = UK_11KV_240MM2_CU_PILC_3CORE
        xlpe = solve_steady_state(I, 11_000.0, xlpe_geom, xlpe_mat, UK_TYPICAL_INSTALLATION)
        pilc = solve_steady_state(I, 11_000.0, pilc_geom, pilc_mat, UK_TYPICAL_INSTALLATION)
        assert pilc.conductor_temp_C > xlpe.conductor_temp_C


class TestCableYearWithArchetypeName:
    """The CableYearSpec must resolve archetype_name via the registry."""

    def test_simulator_resolves_archetype_name(self) -> None:
        from gridforge.data.cable_year import CableYearSpec, simulate_cable_year
        from gridforge.data.load_profiles import LoadSpec
        from gridforge.data.weather import WeatherSpec

        spec = CableYearSpec(
            cable_id="test_archetype",
            duration_years=0.02,
            load=LoadSpec("residential", peak_A=180.0, base_A=40.0, seed=1),
            weather=WeatherSpec(seed=1),
            archetype_name="11kV_95mm2_Cu_XLPE_3c",
        )
        result = simulate_cable_year(spec)
        assert result.archetype_name == "11kV_95mm2_Cu_XLPE_3c"
        # Conductor must rise meaningfully above ambient at peak load, and
        # on average sit above ambient (small momentary lag dips are
        # allowed because the cable's 1-hour lag cannot track the weather
        # noise instantaneously).
        delta = result.conductor_C - result.ambient_C
        assert delta.max() > 1.0
        assert delta.mean() > 0.5

    def test_unknown_archetype_name_raises(self) -> None:
        from gridforge.data.cable_year import CableYearSpec, simulate_cable_year
        from gridforge.data.load_profiles import LoadSpec
        from gridforge.data.weather import WeatherSpec

        spec = CableYearSpec(
            cable_id="bad",
            duration_years=0.02,
            load=LoadSpec("residential", peak_A=200.0, base_A=50.0, seed=0),
            weather=WeatherSpec(seed=0),
            archetype_name="not_a_real_archetype",
        )
        with pytest.raises(KeyError):
            simulate_cable_year(spec)

    def test_no_archetype_name_uses_default(self) -> None:
        from gridforge.data.cable_year import CableYearSpec, simulate_cable_year
        from gridforge.data.load_profiles import LoadSpec
        from gridforge.data.weather import WeatherSpec

        spec = CableYearSpec(
            cable_id="default",
            duration_years=0.02,
            load=LoadSpec("residential", peak_A=300.0, base_A=80.0, seed=0),
            weather=WeatherSpec(seed=0),
        )
        result = simulate_cable_year(spec)
        assert result.archetype_name == "11kV_240mm2_Cu_XLPE_3c"
