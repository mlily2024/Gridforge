"""
Train a deep ensemble of IEC-surrogate PINNs and save it for the uncertainty
endpoint (ADR-0008; GridOptima C1 `/api/gridforge/conductor-temperature-ci`).

Trains N independent members, reports per-member validation RMSE plus the
ensemble's calibration (95% CI coverage) on an independent held-out set, and
saves the member state dicts.

Outputs:
  scripts/output/09_ensemble/member_00.pt ... member_0(N-1).pt
  scripts/output/09_ensemble/manifest.json
"""

from __future__ import annotations

import sys
import time as _time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from gridforge.physics.cable_archetype import (
    UK_11KV_240MM2_XLPE_3CORE,
    UK_TYPICAL_INSTALLATION,
)
from gridforge.training.ensemble import coverage, train_ensemble
from gridforge.training.train import TrainingConfig, generate_training_data

N_MEMBERS = 5


def main() -> int:
    print()
    print("GridForge -- deep-ensemble PINN (uncertainty quantification)")
    print(f"Members: {N_MEMBERS}")
    print()

    cfg = TrainingConfig(seed=2026_05_29)
    t0 = _time.time()
    ensemble, results = train_ensemble(n_members=N_MEMBERS, cfg=cfg, verbose=False)
    elapsed = _time.time() - t0
    print(f"Trained {N_MEMBERS} members in {elapsed:.1f}s")
    for k, r in enumerate(results):
        print(f"  member {k}: best val RMSE {r.best_val_rmse_C:.4f} degC")

    # Calibration + accuracy on an independent held-out set.
    geom, mat = UK_11KV_240MM2_XLPE_3CORE
    X, y = generate_training_data(
        1000, cfg, geom, mat, UK_TYPICAL_INSTALLATION, seed=cfg.seed + 99
    )
    pred = ensemble.predict(X[:, 0], X[:, 1], X[:, 2])
    rmse = float(np.sqrt(np.mean((pred.mean - y.reshape(-1)) ** 2)))
    mean_sigma = float(np.mean(pred.std))
    cov = coverage(ensemble, X, y)
    print()
    print("=== Ensemble quality (n=1000 held-out) ===")
    print(f"  Mean RMSE              : {rmse:.4f} degC")
    print(f"  Mean sigma (epistemic) : {mean_sigma:.4f} degC")
    print(f"  95% CI coverage        : {cov * 100:.1f}% (nominal 95%)")

    out_dir = Path(__file__).resolve().parent / "output" / "09_ensemble"
    ensemble.save(out_dir)
    print()
    print(f"  Saved ensemble         : {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
