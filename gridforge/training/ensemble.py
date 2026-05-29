"""
Train a deep ensemble of IEC-surrogate PINNs.

Trains K independent members, each with a distinct weight-initialisation seed
*and* a distinct training-data seed, by reusing the single-model ``train_pinn``
loop. The spread across members is the ensemble's epistemic uncertainty (see
ADR-0008 and ``gridforge.models.ensemble.DeepEnsemblePINN``).

A ``coverage`` helper measures calibration: the fraction of held-out targets
that fall inside the predicted confidence interval, which should track the
nominal level (~95% for z=1.96) on well-calibrated data.

PyTorch is an optional dependency.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Optional

try:
    import torch

    _TORCH_AVAILABLE = True
except ImportError:  # pragma: no cover
    torch = None  # type: ignore[assignment]
    _TORCH_AVAILABLE = False

import numpy as np

from ..inference.virtual_sensors import TrainedPINNSurrogate
from ..models.ensemble import DeepEnsemblePINN
from ..models.pinn import IECSurrogatePINN
from .train import TrainingConfig, TrainingResult, train_pinn

# Prime stride between member seeds -- keeps each member's data sample and
# weight initialisation well separated.
_SEED_STRIDE = 7919


def _require_torch() -> None:
    if not _TORCH_AVAILABLE:
        raise ImportError(
            "PyTorch is required for gridforge.training.ensemble. "
            "Install with: pip install 'gridforge[ml]'"
        )


def train_ensemble(
    n_members: int = 5,
    cfg: Optional[TrainingConfig] = None,
    base_seed: Optional[int] = None,
    verbose: bool = False,
) -> tuple[DeepEnsemblePINN, list[TrainingResult]]:
    """Train ``n_members`` independent PINNs and wrap them in a DeepEnsemblePINN.

    Returns the ensemble and the per-member training results (for diagnostics).
    """
    _require_torch()
    if n_members < 1:
        raise ValueError("n_members must be >= 1")
    if cfg is None:
        cfg = TrainingConfig()
    base = cfg.seed if base_seed is None else base_seed

    members: list[TrainedPINNSurrogate] = []
    results: list[TrainingResult] = []
    for k in range(n_members):
        member_seed = base + k * _SEED_STRIDE
        # Distinct weight initialisation per member.
        torch.manual_seed(member_seed)
        model = IECSurrogatePINN()
        # Distinct training-data sample per member.
        member_cfg = replace(cfg, seed=member_seed)
        result = train_pinn(model, cfg=member_cfg, verbose=verbose)
        members.append(TrainedPINNSurrogate(model))
        results.append(result)

    return DeepEnsemblePINN(members), results


def coverage(
    ensemble: DeepEnsemblePINN,
    X: np.ndarray,
    y: np.ndarray,
    z: float = 1.96,
) -> float:
    """Empirical CI coverage: fraction of targets inside the ``mean +/- z*sigma`` band.

    ``X`` has shape (n, 3) with columns (I_A, ambient_C, soil_rho_t_KmW); ``y``
    has shape (n,) or (n, 1) of true conductor temperatures [degC].
    """
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    pred = ensemble.predict(X[:, 0], X[:, 1], X[:, 2], z=z)
    inside = (y >= pred.lower) & (y <= pred.upper)
    return float(np.mean(inside))


__all__ = ["train_ensemble", "coverage"]
