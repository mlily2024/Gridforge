"""
Physics-informed neural network for the IEC 60287 steady-state surrogate.

Forward model:
  inputs  : (I_A, ambient_C, soil_rho_T_KmW)        — normalised to ~[-1, 1]
  output  : conductor temperature T_c [degC]

Architecture: small MLP with tanh activations and an optional sinusoidal
positional encoding on the inputs. Tanh is the standard PINN activation
because it has continuous higher-order derivatives that the physics-residual
loss path needs.

The network is designed to be trainable on CPU in under a minute. PyTorch
is an optional dependency — importing this module without torch installed
raises a clear ImportError.
"""

from __future__ import annotations

from dataclasses import dataclass

try:
    import torch
    import torch.nn as nn

    _TORCH_AVAILABLE = True
except ImportError:  # pragma: no cover — exercised only without torch
    torch = None  # type: ignore[assignment]
    nn = None  # type: ignore[assignment]
    _TORCH_AVAILABLE = False


def _require_torch() -> None:
    if not _TORCH_AVAILABLE:
        raise ImportError(
            "PyTorch is required for gridforge.models.pinn. "
            "Install with: pip install 'gridforge[ml]'"
        )


@dataclass(frozen=True)
class InputNormaliser:
    """Linear scaling to map physical inputs to roughly [-1, 1].

    The reference ranges are deliberately wide enough to cover the operating
    envelope of UK 11 kV distribution cables. Values outside the reference
    range are not extrapolated by the network — the caller should keep
    inputs inside, or extend the ranges and retrain.
    """

    I_min_A: float = 0.0
    I_max_A: float = 700.0
    ambient_min_C: float = 0.0
    ambient_max_C: float = 25.0
    rho_t_min_KmW: float = 0.5
    rho_t_max_KmW: float = 2.5

    def normalise(self, x: "torch.Tensor") -> "torch.Tensor":
        """x has shape (..., 3): columns I, ambient, rho_t."""
        I = (x[..., 0] - self.I_min_A) / (self.I_max_A - self.I_min_A) * 2.0 - 1.0
        amb = (x[..., 1] - self.ambient_min_C) / (
            self.ambient_max_C - self.ambient_min_C
        ) * 2.0 - 1.0
        rho = (x[..., 2] - self.rho_t_min_KmW) / (
            self.rho_t_max_KmW - self.rho_t_min_KmW
        ) * 2.0 - 1.0
        return torch.stack([I, amb, rho], dim=-1)


class SinusoidalEncoding:
    """Fourier-feature positional encoding.

    Augments each input scalar with a bank of sin(2^k pi x) and cos(2^k pi x)
    features for k in [0, n_freqs). Helps PINNs avoid spectral bias on smooth
    targets — see Tancik et al., NeurIPS 2020.
    """

    def __init__(self, n_freqs: int = 4) -> None:
        _require_torch()
        self.n_freqs = n_freqs

    @property
    def out_features_per_input(self) -> int:
        # original + sin/cos per frequency
        return 1 + 2 * self.n_freqs

    def __call__(self, x: "torch.Tensor") -> "torch.Tensor":
        # x shape (..., d). Output shape (..., d * (1 + 2 * n_freqs)).
        feats = [x]
        for k in range(self.n_freqs):
            scale = (2.0**k) * 3.141592653589793
            feats.append(torch.sin(scale * x))
            feats.append(torch.cos(scale * x))
        return torch.cat(feats, dim=-1)


class IECSurrogatePINN(nn.Module if _TORCH_AVAILABLE else object):
    """MLP that maps (I, ambient, rho_t) to conductor temperature.

    Forward output is in degC, not normalised. The conductor temperature is
    parameterised as `ambient + raw_output`, so the network only has to
    learn the temperature *rise* — which is non-negative and bounded.
    A softplus on the raw output enforces non-negativity.
    """

    def __init__(
        self,
        n_hidden: int = 4,
        hidden_size: int = 64,
        n_freqs: int = 4,
        normaliser: InputNormaliser | None = None,
    ) -> None:
        _require_torch()
        super().__init__()
        self.normaliser = normaliser if normaliser is not None else InputNormaliser()
        self.encoder = SinusoidalEncoding(n_freqs=n_freqs)

        in_features = 3 * self.encoder.out_features_per_input
        layers: list[nn.Module] = []
        prev = in_features
        for _ in range(n_hidden):
            layers.append(nn.Linear(prev, hidden_size))
            layers.append(nn.Tanh())
            prev = hidden_size
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: "torch.Tensor") -> "torch.Tensor":
        """Conductor temperature in degC. x has shape (B, 3): I, ambient, rho_t."""
        ambient = x[..., 1:2]
        x_norm = self.normaliser.normalise(x)
        x_encoded = self.encoder(x_norm)
        rise = nn.functional.softplus(self.net(x_encoded))
        return ambient + rise

    def n_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
