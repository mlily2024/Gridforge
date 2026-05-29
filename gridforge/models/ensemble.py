"""
Deep-ensemble wrapper for the IEC 60287 surrogate PINN.

A deep ensemble trains K independent PINNs (different weight initialisations
and different training samples) and treats the spread of their predictions as
*epistemic* uncertainty. For a query point the ensemble reports the mean
prediction together with a Gaussian confidence interval ``mean +/- z * sigma``,
where ``sigma`` is the inter-member standard deviation.

This is the simplest principled uncertainty-quantification approach for the
surrogate: it needs no change to the network architecture or the training loss
(see ADR-0008), parallelises trivially, and degrades gracefully -- a one-member
"ensemble" behaves exactly like the existing point-estimate surrogate.

PyTorch is an optional dependency; importing this module without torch is fine,
but constructing or loading an ensemble raises a clear ImportError.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional, Sequence, Union

try:
    import torch

    _TORCH_AVAILABLE = True
except ImportError:  # pragma: no cover -- exercised only without torch
    torch = None  # type: ignore[assignment]
    _TORCH_AVAILABLE = False

import numpy as np

from ..inference.virtual_sensors import TrainedPINNSurrogate

ArrayLike = Union[float, Sequence[float], np.ndarray]

_MEMBER_GLOB = "member_*.pt"
_MANIFEST_NAME = "manifest.json"


def _require_torch() -> None:
    if not _TORCH_AVAILABLE:
        raise ImportError(
            "PyTorch is required for gridforge.models.ensemble. "
            "Install with: pip install 'gridforge[ml]'"
        )


@dataclass(frozen=True)
class EnsemblePrediction:
    """Per-point ensemble output. All arrays share the query length n.

    mean   : ensemble mean conductor temperature [degC]
    std    : inter-member standard deviation (epistemic uncertainty) [degC]
    lower  : mean - z * std  (lower CI bound) [degC]
    upper  : mean + z * std  (upper CI bound) [degC]
    z      : the z-score used for the interval (1.96 -> ~95%)
    """

    mean: np.ndarray
    std: np.ndarray
    lower: np.ndarray
    upper: np.ndarray
    z: float


class DeepEnsemblePINN:
    """An ensemble of trained IEC-surrogate PINNs with uncertainty output."""

    def __init__(self, members: Sequence[TrainedPINNSurrogate]) -> None:
        _require_torch()
        members = list(members)
        if not members:
            raise ValueError("DeepEnsemblePINN needs at least one member")
        self.members = members

    @property
    def n_members(self) -> int:
        return len(self.members)

    def _stack(self, I_A, ambient_C, soil_rho_t_KmW) -> np.ndarray:
        """Predictions from every member, shape (n_members, n_points)."""
        preds = [m(I_A, ambient_C, soil_rho_t_KmW) for m in self.members]
        return np.stack(preds, axis=0)

    def predict(
        self,
        I_A: ArrayLike,
        ambient_C: ArrayLike,
        soil_rho_t_KmW: ArrayLike,
        z: float = 1.96,
    ) -> EnsemblePrediction:
        """Mean conductor temperature and a ``mean +/- z * sigma`` interval."""
        per_member = self._stack(I_A, ambient_C, soil_rho_t_KmW)
        mean = per_member.mean(axis=0)
        if self.n_members > 1:
            # Sample std (ddof=1): members are draws from the model posterior.
            std = per_member.std(axis=0, ddof=1)
        else:
            std = np.zeros_like(mean)
        return EnsemblePrediction(
            mean=mean,
            std=std,
            lower=mean - z * std,
            upper=mean + z * std,
            z=float(z),
        )

    def __call__(
        self, I_A: ArrayLike, ambient_C: ArrayLike, soil_rho_t_KmW: ArrayLike
    ) -> np.ndarray:
        """Ensemble mean only -- drop-in compatible with TrainedPINNSurrogate."""
        return self.predict(I_A, ambient_C, soil_rho_t_KmW).mean

    def save(self, directory: Union[str, Path]) -> None:
        """Write each member's state dict + a small manifest to ``directory``."""
        _require_torch()
        d = Path(directory)
        d.mkdir(parents=True, exist_ok=True)
        for i, m in enumerate(self.members):
            torch.save(m.model.state_dict(), d / f"member_{i:02d}.pt")
        (d / _MANIFEST_NAME).write_text(json.dumps({"n_members": self.n_members}), encoding="utf-8")

    @classmethod
    def load(
        cls,
        directory: Union[str, Path],
        model_factory: Optional[Callable[[], object]] = None,
    ) -> "DeepEnsemblePINN":
        """Load an ensemble previously written by ``save``.

        ``model_factory`` builds a fresh architecture instance for each member;
        it defaults to the standard ``IECSurrogatePINN()``.
        """
        _require_torch()
        from .pinn import IECSurrogatePINN

        d = Path(directory)
        paths = sorted(d.glob(_MEMBER_GLOB))
        if not paths:
            raise FileNotFoundError(f"no {_MEMBER_GLOB} files found in {d}")
        factory = model_factory or (lambda: IECSurrogatePINN())
        members = []
        for p in paths:
            model = factory()
            state = torch.load(p, map_location="cpu", weights_only=True)
            model.load_state_dict(state)
            members.append(TrainedPINNSurrogate(model))
        return cls(members)


__all__ = ["EnsemblePrediction", "DeepEnsemblePINN"]
