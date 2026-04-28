"""
Virtual-sensor wrapper around a trained PINN.

Exposes the network as a plain Python callable T_c(I, ambient, rho_t)
returning conductor temperature in degC, suitable for integration into
GridOptima's decision engine without the consumer needing to import
torch.

Compared to calling `solve_steady_state` directly:
  - Faster (single forward pass vs fixed-point iteration)
  - Vectorised across thousands of operating points in one call
  - Differentiable inputs (sensitivities for cost-benefit modelling)

Compared to a simple regression model:
  - Constrained by IEC 60287 physics during training, so out-of-sample
    behaviour respects energy balance rather than free-fitting noise
"""

from __future__ import annotations

from typing import Sequence, Union

try:
    import torch

    _TORCH_AVAILABLE = True
except ImportError:  # pragma: no cover
    torch = None  # type: ignore[assignment]
    _TORCH_AVAILABLE = False

import numpy as np


ArrayLike = Union[float, Sequence[float], np.ndarray]


def _require_torch() -> None:
    if not _TORCH_AVAILABLE:
        raise ImportError(
            "PyTorch is required for gridforge.inference.virtual_sensors. "
            "Install with: pip install 'gridforge[ml]'"
        )


class TrainedPINNSurrogate:
    """Wraps a trained `IECSurrogatePINN` as a numpy-friendly callable.

    Example:
        >>> surrogate = TrainedPINNSurrogate(model)
        >>> T_c = surrogate(I_A=400.0, ambient_C=15.0, soil_rho_t_KmW=1.0)

    Vectorised inputs are fine:
        >>> Is = np.linspace(50, 600, 12)
        >>> T_c = surrogate(Is, 15.0, 1.0)  # array of length 12
    """

    def __init__(self, model) -> None:
        _require_torch()
        self.model = model
        self.model.eval()

    def __call__(
        self,
        I_A: ArrayLike,
        ambient_C: ArrayLike,
        soil_rho_t_KmW: ArrayLike,
    ) -> np.ndarray:
        """Predict conductor temperature [degC] for one or more operating points."""
        I_arr = np.atleast_1d(np.asarray(I_A, dtype=np.float32))
        amb_arr = np.atleast_1d(np.asarray(ambient_C, dtype=np.float32))
        rho_arr = np.atleast_1d(np.asarray(soil_rho_t_KmW, dtype=np.float32))

        # Broadcast to a common length
        n = max(I_arr.size, amb_arr.size, rho_arr.size)
        if I_arr.size == 1:
            I_arr = np.full(n, float(I_arr[0]), dtype=np.float32)
        if amb_arr.size == 1:
            amb_arr = np.full(n, float(amb_arr[0]), dtype=np.float32)
        if rho_arr.size == 1:
            rho_arr = np.full(n, float(rho_arr[0]), dtype=np.float32)

        if not (I_arr.size == amb_arr.size == rho_arr.size):
            raise ValueError(
                "I_A, ambient_C, soil_rho_t_KmW must broadcast to a common length"
            )

        x = np.stack([I_arr, amb_arr, rho_arr], axis=-1)
        with torch.no_grad():
            T = self.model(torch.tensor(x))
        out = T.detach().cpu().numpy().reshape(-1)
        return out

    @classmethod
    def load(cls, path: str, model=None) -> "TrainedPINNSurrogate":
        """Load a state-dict-saved PINN from disk.

        Pass `model` with the same architecture as the saved state dict.
        If omitted, the default `IECSurrogatePINN()` architecture is used —
        which only works if the saved model was trained with defaults.
        """
        _require_torch()
        from ..models.pinn import IECSurrogatePINN

        state = torch.load(path, map_location="cpu", weights_only=True)
        if model is None:
            model = IECSurrogatePINN()
        model.load_state_dict(state)
        return cls(model)
