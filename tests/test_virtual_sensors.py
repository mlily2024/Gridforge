"""Tests for the trained-PINN surrogate wrapper."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

import numpy as np

from gridforge.inference.virtual_sensors import TrainedPINNSurrogate
from gridforge.models.pinn import IECSurrogatePINN


class TestSurrogate:
    def setup_method(self) -> None:
        self.model = IECSurrogatePINN(n_hidden=2, hidden_size=16, n_freqs=2)
        self.surrogate = TrainedPINNSurrogate(self.model)

    def test_scalar_inputs(self) -> None:
        T = self.surrogate(I_A=400.0, ambient_C=15.0, soil_rho_t_KmW=1.0)
        assert T.shape == (1,)
        # Output is a real number above ambient at a valid input
        assert np.isfinite(T[0])

    def test_vector_input_for_one_axis(self) -> None:
        Is = np.linspace(50, 600, 8)
        T = self.surrogate(Is, 15.0, 1.0)
        assert T.shape == (8,)

    def test_all_vector_inputs(self) -> None:
        n = 5
        Is = np.linspace(100, 500, n)
        ambs = np.linspace(5, 20, n)
        rhos = np.linspace(0.7, 1.5, n)
        T = self.surrogate(Is, ambs, rhos)
        assert T.shape == (n,)

    def test_mismatched_lengths_rejected(self) -> None:
        with pytest.raises(ValueError):
            self.surrogate(np.array([100.0, 200.0]), np.array([10.0, 15.0, 20.0]), 1.0)

    def test_save_and_load_roundtrip(self, tmp_path) -> None:
        path = tmp_path / "pinn.pt"
        torch.save(self.model.state_dict(), path)
        # Pass a fresh model with the same architecture as `self.model`
        same_arch = IECSurrogatePINN(n_hidden=2, hidden_size=16, n_freqs=2)
        loaded = TrainedPINNSurrogate.load(str(path), model=same_arch)
        T_orig = self.surrogate(400.0, 15.0, 1.0)
        T_loaded = loaded(400.0, 15.0, 1.0)
        assert np.allclose(T_orig, T_loaded, atol=1e-6)
