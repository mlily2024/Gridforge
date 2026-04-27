"""Verification tests for the radial electric-field model."""

from __future__ import annotations

from math import log

import pytest

from gridforge.physics.cable_archetype import UK_11KV_240MM2_XLPE_3CORE
from gridforge.physics.electrical import (
    average_e_field,
    conductor_outer_radius_m,
    insulation_outer_radius_m,
    max_e_field,
    min_e_field,
    radial_e_field,
)


class TestRadialEField:
    def setup_method(self) -> None:
        self.geom, _ = UK_11KV_240MM2_XLPE_3CORE
        self.U0 = 11_000.0 / 3.0**0.5  # phase-to-ground

    def test_max_field_at_conductor_surface(self) -> None:
        E_max = max_e_field(self.U0, self.geom)
        E_min = min_e_field(self.U0, self.geom)
        assert E_max > E_min

    def test_field_matches_closed_form(self) -> None:
        # Hand calculation at conductor surface
        r_c = conductor_outer_radius_m(self.geom)
        r_s = insulation_outer_radius_m(self.geom)
        expected = self.U0 / (r_c * log(r_s / r_c))
        assert max_e_field(self.U0, self.geom) == pytest.approx(expected, rel=1e-12)

    def test_average_field_equals_voltage_over_thickness(self) -> None:
        E_avg = average_e_field(self.U0, self.geom)
        expected = self.U0 / (self.geom.t_i_mm * 1e-3)
        assert E_avg == pytest.approx(expected, rel=1e-12)

    def test_e_max_for_11kV_xlpe_in_design_range(self) -> None:
        """E_max for 11 kV 240 mm^2 XLPE should be ~ 2-5 MV/m."""
        E = max_e_field(self.U0, self.geom)
        assert 1.5e6 < E < 6.0e6

    def test_field_outside_insulation_raises(self) -> None:
        r_too_small = conductor_outer_radius_m(self.geom) * 0.9
        with pytest.raises(ValueError):
            radial_e_field(r_too_small, self.U0, self.geom)
        r_too_large = insulation_outer_radius_m(self.geom) * 1.1
        with pytest.raises(ValueError):
            radial_e_field(r_too_large, self.U0, self.geom)

    def test_field_inversely_proportional_to_radius(self) -> None:
        r_c = conductor_outer_radius_m(self.geom)
        r_s = insulation_outer_radius_m(self.geom)
        r_mid = 0.5 * (r_c + r_s)
        E_inner = radial_e_field(r_c, self.U0, self.geom)
        E_mid = radial_e_field(r_mid, self.U0, self.geom)
        E_outer = radial_e_field(r_s, self.U0, self.geom)
        # E ~ 1/r so the products E*r should be equal (within numerical error)
        assert E_inner * r_c == pytest.approx(E_mid * r_mid, rel=1e-12)
        assert E_inner * r_c == pytest.approx(E_outer * r_s, rel=1e-12)

    def test_field_scales_linearly_with_voltage(self) -> None:
        E1 = max_e_field(self.U0, self.geom)
        E2 = max_e_field(2.0 * self.U0, self.geom)
        assert E2 == pytest.approx(2.0 * E1, rel=1e-12)
