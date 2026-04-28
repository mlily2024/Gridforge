"""Verification tests for the combined PINN loss + physics residual."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from gridforge.physics.cable_archetype import (
    UK_11KV_240MM2_XLPE_3CORE,
    UK_TYPICAL_INSTALLATION,
)
from gridforge.physics.thermal import solve_steady_state
from gridforge.training.loss import (
    PhysicsConstants,
    adaptive_loss_weights,
    combined_loss,
    physics_residual,
    precompute_physics_constants,
)


class TestConstantsPrecompute:
    def test_returns_expected_shape(self) -> None:
        geom, mat = UK_11KV_240MM2_XLPE_3CORE
        c = precompute_physics_constants(geom, mat, UK_TYPICAL_INSTALLATION)
        assert c.T1_KmW > 0
        assert c.T3_KmW > 0
        assert c.T4_geometric_factor > 0
        assert c.W_d_W_per_m > 0

    def test_too_shallow_burial_rejected(self) -> None:
        geom, mat = UK_11KV_240MM2_XLPE_3CORE
        from gridforge.physics.thermal import InstallationConditions
        bad = InstallationConditions(
            burial_depth_m=0.01, soil_thermal_resistivity_KmW=1.0,
        )
        with pytest.raises(ValueError):
            precompute_physics_constants(geom, mat, bad)


class TestPhysicsResidual:
    def setup_method(self) -> None:
        geom, mat = UK_11KV_240MM2_XLPE_3CORE
        self.c = precompute_physics_constants(geom, mat, UK_TYPICAL_INSTALLATION)
        self.geom = geom
        self.mat = mat

    def test_residual_zero_at_oracle_solution(self) -> None:
        """If we plug the IEC oracle's T_c into the residual, it should be ~0."""
        I, amb, rho = 400.0, 15.0, 1.0
        from gridforge.physics.thermal import InstallationConditions
        install = InstallationConditions(
            burial_depth_m=0.8, soil_thermal_resistivity_KmW=rho,
            ambient_soil_temp_C=amb,
        )
        sol = solve_steady_state(I, 11_000.0, self.geom, self.mat, install)
        T_c = torch.tensor([[sol.conductor_temp_C]])
        x = torch.tensor([[I, amb, rho]])
        r = physics_residual(T_c, x, self.c)
        assert abs(float(r.item())) < 0.01  # within solver tolerance

    def test_residual_non_zero_at_wrong_T(self) -> None:
        I, amb, rho = 400.0, 15.0, 1.0
        x = torch.tensor([[I, amb, rho]])
        T_c = torch.tensor([[80.0]])  # arbitrarily wrong
        r = physics_residual(T_c, x, self.c)
        assert abs(float(r.item())) > 1.0

    def test_residual_differentiable_through_T_c(self) -> None:
        x = torch.tensor([[400.0, 15.0, 1.0]])
        T_c = torch.tensor([[60.0]], requires_grad=True)
        r = physics_residual(T_c, x, self.c)
        r.sum().backward()
        assert T_c.grad is not None
        assert torch.isfinite(T_c.grad).all()


class TestCombinedLoss:
    def setup_method(self) -> None:
        geom, mat = UK_11KV_240MM2_XLPE_3CORE
        self.c = precompute_physics_constants(geom, mat, UK_TYPICAL_INSTALLATION)

    def test_loss_components_returned(self) -> None:
        x = torch.tensor([[400.0, 15.0, 1.0], [300.0, 10.0, 1.5]])
        T_pred = torch.tensor([[60.0], [50.0]], requires_grad=True)
        T_target = torch.tensor([[60.0], [50.0]])
        out = combined_loss(T_pred, T_target, x, self.c, w_data=1.0, w_phys=1.0)
        assert "data" in out and "physics" in out and "total" in out

    def test_zero_data_loss_when_target_matches(self) -> None:
        x = torch.tensor([[400.0, 15.0, 1.0]])
        T_pred = torch.tensor([[55.0]], requires_grad=True)
        T_target = torch.tensor([[55.0]])
        out = combined_loss(T_pred, T_target, x, self.c)
        assert float(out["data"].item()) == pytest.approx(0.0, abs=1e-9)
        # Physics term may not be zero, but is finite
        assert torch.isfinite(out["physics"])
