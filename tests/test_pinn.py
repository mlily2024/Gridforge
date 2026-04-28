"""Architecture and forward-pass tests for the IEC surrogate PINN."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from gridforge.models.pinn import (
    IECSurrogatePINN,
    InputNormaliser,
    SinusoidalEncoding,
)


class TestInputNormaliser:
    def test_normalises_to_unit_range(self) -> None:
        norm = InputNormaliser(0.0, 700.0, 0.0, 25.0, 0.5, 2.5)
        x_min = torch.tensor([[0.0, 0.0, 0.5]])
        x_max = torch.tensor([[700.0, 25.0, 2.5]])
        x_mid = torch.tensor([[350.0, 12.5, 1.5]])
        out_min = norm.normalise(x_min)
        out_max = norm.normalise(x_max)
        out_mid = norm.normalise(x_mid)
        assert torch.allclose(out_min, torch.tensor([[-1.0, -1.0, -1.0]]), atol=1e-6)
        assert torch.allclose(out_max, torch.tensor([[1.0, 1.0, 1.0]]), atol=1e-6)
        assert torch.allclose(out_mid, torch.tensor([[0.0, 0.0, 0.0]]), atol=1e-6)


class TestSinusoidalEncoding:
    def test_output_dim(self) -> None:
        enc = SinusoidalEncoding(n_freqs=4)
        x = torch.zeros(2, 3)  # batch=2, dim=3
        y = enc(x)
        # Per-input: 1 + 2*4 = 9, three inputs → 27
        assert y.shape == (2, 9 * 3)

    def test_zero_input_consistent(self) -> None:
        enc = SinusoidalEncoding(n_freqs=2)
        x = torch.zeros(1, 3)
        y = enc(x)
        # cos(0) = 1, sin(0) = 0
        # sin/cos features for x=0 give [0, 1] for each frequency
        assert torch.all(torch.isfinite(y))


class TestPINN:
    def setup_method(self) -> None:
        self.model = IECSurrogatePINN()

    def test_forward_pass_shape(self) -> None:
        x = torch.tensor([[400.0, 15.0, 1.0]])
        out = self.model(x)
        assert out.shape == (1, 1)

    def test_forward_batch(self) -> None:
        x = torch.rand(8, 3) * torch.tensor([700.0, 25.0, 2.0])
        out = self.model(x)
        assert out.shape == (8, 1)

    def test_output_above_ambient(self) -> None:
        """Softplus on rise + ambient guarantees T_c >= ambient at construction."""
        x = torch.tensor([[100.0, 15.0, 1.0], [200.0, 5.0, 1.5], [50.0, 20.0, 0.8]])
        out = self.model(x)
        for i in range(x.shape[0]):
            ambient = float(x[i, 1])
            T_c = float(out[i, 0])
            assert T_c >= ambient - 1e-6, f"row {i}: T_c={T_c}, amb={ambient}"

    def test_parameter_count_in_expected_band(self) -> None:
        """Default 4 hidden x 64 with sinusoidal encoding (4 freqs) ~ 19k params."""
        n = self.model.n_parameters()
        assert 5_000 < n < 50_000

    def test_gradients_flow(self) -> None:
        x = torch.rand(4, 3, requires_grad=True) * torch.tensor([700.0, 25.0, 2.0])
        out = self.model(x)
        loss = out.sum()
        loss.backward()
        # All learnable parameters should have a non-None gradient
        for p in self.model.parameters():
            assert p.grad is not None
