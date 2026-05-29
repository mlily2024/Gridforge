"""Tests for the deep-ensemble PINN uncertainty wrapper (ADR-0008)."""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")  # whole module skipped without torch

from gridforge.models.ensemble import DeepEnsemblePINN  # noqa: E402
from gridforge.physics.cable_archetype import (  # noqa: E402
    UK_11KV_240MM2_XLPE_3CORE,
    UK_TYPICAL_INSTALLATION,
)
from gridforge.training.ensemble import coverage, train_ensemble  # noqa: E402
from gridforge.training.train import TrainingConfig, generate_training_data  # noqa: E402


@pytest.fixture(scope="module")
def small_ensemble():
    # Tiny, fast configuration -- enough to exercise the mechanics, not GF-003.
    cfg = TrainingConfig(n_train=300, n_val=100, n_epochs=120, batch_size=64, seed=7)
    ensemble, _results = train_ensemble(n_members=3, cfg=cfg, verbose=False)
    return ensemble, cfg


class TestDeepEnsemble:
    def test_member_count(self, small_ensemble):
        ensemble, _ = small_ensemble
        assert ensemble.n_members == 3

    def test_prediction_structure(self, small_ensemble):
        ensemble, _ = small_ensemble
        p = ensemble.predict(400.0, 15.0, 1.0)
        assert p.mean.shape == (1,)
        assert np.all(p.std >= 0.0)
        assert np.all(p.lower <= p.mean)
        assert np.all(p.mean <= p.upper)
        assert p.z == pytest.approx(1.96)

    def test_call_returns_mean(self, small_ensemble):
        ensemble, _ = small_ensemble
        assert np.allclose(
            ensemble(400.0, 15.0, 1.0), ensemble.predict(400.0, 15.0, 1.0).mean
        )

    def test_vectorised(self, small_ensemble):
        ensemble, _ = small_ensemble
        Is = np.linspace(100.0, 500.0, 8)
        p = ensemble.predict(Is, 15.0, 1.0)
        assert p.mean.shape == (8,)
        assert np.all(p.upper >= p.lower)

    def test_mean_above_ambient(self, small_ensemble):
        # T_c = ambient + softplus(...) by construction -> mean >= ambient.
        ensemble, _ = small_ensemble
        amb = 12.0
        p = ensemble.predict(np.linspace(100.0, 500.0, 6), amb, 1.0)
        assert np.all(p.mean >= amb - 1e-6)

    def test_beats_naive_baseline(self, small_ensemble):
        # The ensemble mean should beat the trivial "predict the average"
        # baseline -- i.e. it has learned real structure, independent of the
        # exact (small) training budget. Tight accuracy is GF-003's job.
        ensemble, cfg = small_ensemble
        geom, mat = UK_11KV_240MM2_XLPE_3CORE
        X, y = generate_training_data(
            60, cfg, geom, mat, UK_TYPICAL_INSTALLATION, seed=999
        )
        y = y.reshape(-1)
        p = ensemble.predict(X[:, 0], X[:, 1], X[:, 2])
        model_rmse = float(np.sqrt(np.mean((p.mean - y) ** 2)))
        naive_rmse = float(np.sqrt(np.mean((y.mean() - y) ** 2)))
        assert model_rmse < naive_rmse

    def test_coverage_in_unit_interval(self, small_ensemble):
        ensemble, cfg = small_ensemble
        geom, mat = UK_11KV_240MM2_XLPE_3CORE
        X, y = generate_training_data(
            80, cfg, geom, mat, UK_TYPICAL_INSTALLATION, seed=12345
        )
        cov = coverage(ensemble, X, y)
        assert 0.0 <= cov <= 1.0

    def test_save_load_roundtrip(self, small_ensemble, tmp_path):
        ensemble, _ = small_ensemble
        ensemble.save(tmp_path / "ens")
        loaded = DeepEnsemblePINN.load(tmp_path / "ens")
        assert loaded.n_members == ensemble.n_members
        assert np.allclose(
            ensemble(400.0, 15.0, 1.0), loaded(400.0, 15.0, 1.0), atol=1e-5
        )

    def test_single_member_zero_std(self):
        cfg = TrainingConfig(n_train=120, n_val=60, n_epochs=20, batch_size=64, seed=3)
        ensemble, _ = train_ensemble(n_members=1, cfg=cfg)
        p = ensemble.predict(300.0, 15.0, 1.0)
        assert np.all(p.std == 0.0)
        assert np.all(p.lower == p.mean)
        assert np.all(p.upper == p.mean)
