"""Smoke tests for the PINN training loop and the GF-003 RMSE target."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from gridforge.models.pinn import IECSurrogatePINN
from gridforge.training.train import (
    TrainingConfig,
    generate_training_data,
    train_pinn,
)


class TestTrainingDataGeneration:
    def test_shape_and_ranges(self) -> None:
        cfg = TrainingConfig(n_train=50, n_val=10, n_epochs=1)
        from gridforge.physics.cable_archetype import (
            UK_11KV_240MM2_XLPE_3CORE,
            UK_TYPICAL_INSTALLATION,
        )

        geom, mat = UK_11KV_240MM2_XLPE_3CORE
        X, y = generate_training_data(50, cfg, geom, mat, UK_TYPICAL_INSTALLATION, seed=0)
        assert X.shape == (50, 3)
        assert y.shape == (50, 1)
        # Inputs lie in the configured ranges
        assert (X[:, 0] >= cfg.I_range_A[0]).all() and (X[:, 0] <= cfg.I_range_A[1]).all()
        assert (X[:, 1] >= cfg.ambient_range_C[0]).all() and (
            X[:, 1] <= cfg.ambient_range_C[1]
        ).all()
        assert (X[:, 2] >= cfg.rho_t_range_KmW[0]).all() and (
            X[:, 2] <= cfg.rho_t_range_KmW[1]
        ).all()
        # Targets are positive (above 0 degC) and below the XLPE thermal limit
        assert (y > -10.0).all() and (y < 200.0).all()


class TestTrainingLoopBasics:
    """Quick training run with very small data + few epochs to check the loop runs."""

    def test_loop_runs_and_returns_result(self) -> None:
        cfg = TrainingConfig(
            n_train=200, n_val=50, n_epochs=20, batch_size=64, weight_update_every=10
        )
        model = IECSurrogatePINN(n_hidden=2, hidden_size=16, n_freqs=2)
        result = train_pinn(model, cfg, verbose=False)
        assert result.epochs_run == 20
        assert len(result.history["epoch"]) == 20
        # Loss must remain finite throughout
        assert all(__import__("math").isfinite(v) for v in result.history["loss_data"])
        # Best validation RMSE recorded must be finite and lower than start
        import math

        assert math.isfinite(result.best_val_rmse_C)
        assert result.history["val_rmse_C"][-1] < result.history["val_rmse_C"][0]


@pytest.mark.slow
class TestGF003Target:
    """Full(er) training run that should achieve RMSE <= 0.5 degC.

    Marked slow — runs only with `pytest -m slow`. The script
    `scripts/05_train_pinn_iec_oracle.py` is the canonical demo of the
    GF-003 target; this test exists so CI can verify the model still hits
    the target after future code changes.
    """

    def test_meets_rmse_target(self) -> None:
        cfg = TrainingConfig(n_train=2000, n_val=400, n_epochs=600, batch_size=128)
        model = IECSurrogatePINN(n_hidden=4, hidden_size=64, n_freqs=4)
        result = train_pinn(model, cfg, verbose=False)
        assert (
            result.best_val_rmse_C <= 0.5
        ), f"GF-003: PINN RMSE {result.best_val_rmse_C:.3f} degC > 0.5 target"
