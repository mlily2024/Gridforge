"""
Train the IEC 60287 surrogate PINN and verify the GF-003 RMSE target.

GF-003 in the consolidated requirements:
  Validation against analytical IEC 60287 steady-state — RMSE <= 0.5 degC
  on conductor temperature for canonical loading conditions.

This script trains the PINN on a few thousand samples drawn from the IEC
60287 oracle, with the physics residual enforced through the same
governing equation, and reports validation RMSE plus the physics-residual
RMS.

Outputs:
  scripts/output/05_pinn_training/training_history.csv
  scripts/output/05_pinn_training/validation.csv
  scripts/output/05_pinn_training/training_curves.png
  scripts/output/05_pinn_training/validation_scatter.png
  scripts/output/05_pinn_training/pinn_state.pt
"""

from __future__ import annotations

import csv
import sys
import time as _time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch

from gridforge.inference.virtual_sensors import TrainedPINNSurrogate
from gridforge.models.pinn import IECSurrogatePINN
from gridforge.physics.cable_archetype import (
    UK_11KV_240MM2_XLPE_3CORE,
    UK_TYPICAL_INSTALLATION,
)
from gridforge.physics.thermal import InstallationConditions, solve_steady_state
from gridforge.training.train import (
    TrainingConfig,
    generate_training_data,
    train_pinn,
)


def main() -> int:
    print()
    print("GridForge — IEC 60287 surrogate PINN")
    print("Target (GF-003): val RMSE <= 0.5 degC on conductor temperature")
    print()

    cfg = TrainingConfig(
        n_train=4000,
        n_val=800,
        n_epochs=1500,
        batch_size=256,
        learning_rate=2.0e-3,
        physics_weight_init=1.0e-3,
        weight_update_every=50,
        seed=2026_04_28,
    )

    torch.manual_seed(cfg.seed)
    model = IECSurrogatePINN(n_hidden=4, hidden_size=64, n_freqs=4)
    print(f"Architecture     : {model.n_parameters():,} parameters "
          f"(4 hidden x 64, sinusoidal n_freqs=4)")
    print(f"Train / Val      : {cfg.n_train} / {cfg.n_val} samples")
    print(f"Epochs           : {cfg.n_epochs}")
    print(f"Batch size       : {cfg.batch_size}")
    print()

    t0 = _time.time()
    result = train_pinn(model, cfg, verbose=True)
    elapsed = _time.time() - t0
    print()
    print(f"Trained in {elapsed:.1f}s")
    print()
    print("=== GF-003 result ===")
    print(f"  Final val RMSE         : {result.final_val_rmse_C:.4f} degC")
    print(f"  Best val RMSE          : {result.best_val_rmse_C:.4f} degC "
          f"at epoch {result.best_epoch}")
    print(f"  Physics residual RMS   : {result.final_physics_residual_RMS:.4e}")
    target = 0.5
    pass_str = "PASS" if result.best_val_rmse_C <= target else "FAIL"
    print(f"  Target <= {target} degC      : {pass_str}")

    # Output artefacts
    out_dir = Path(__file__).resolve().parent / "output" / "05_pinn_training"
    out_dir.mkdir(parents=True, exist_ok=True)

    history_path = out_dir / "training_history.csv"
    keys = ["epoch", "loss_total", "loss_data", "loss_physics",
            "val_rmse_C", "w_phys"]
    with history_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(keys)
        for i in range(len(result.history["epoch"])):
            writer.writerow([result.history[k][i] for k in keys])
    print(f"  Saved history          : {history_path}")

    # Validation scatter — predicted vs oracle on a held-out grid
    geom, mat = UK_11KV_240MM2_XLPE_3CORE
    install = UK_TYPICAL_INSTALLATION
    surrogate = TrainedPINNSurrogate(model)

    rng = np.random.default_rng(cfg.seed + 7919)
    n_check = 1000
    Is = rng.uniform(*cfg.I_range_A, size=n_check)
    ambs = rng.uniform(*cfg.ambient_range_C, size=n_check)
    rhos = rng.uniform(*cfg.rho_t_range_KmW, size=n_check)
    oracle_T = np.empty(n_check)
    for i in range(n_check):
        sub = InstallationConditions(
            burial_depth_m=install.burial_depth_m,
            soil_thermal_resistivity_KmW=float(rhos[i]),
            ambient_soil_temp_C=float(ambs[i]),
        )
        sub_sol = solve_steady_state(
            current_per_phase_A=float(Is[i]),
            line_voltage_V_rms=cfg.line_voltage_V_rms,
            geom=geom, mat=mat, install=sub,
        )
        oracle_T[i] = sub_sol.conductor_temp_C
    pinn_T = surrogate(Is, ambs, rhos)
    err = pinn_T - oracle_T
    rmse = float(np.sqrt(np.mean(err * err)))
    print(f"  Independent val RMSE   : {rmse:.4f} degC (n={n_check})")

    val_path = out_dir / "validation.csv"
    with val_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["I_A", "ambient_C", "rho_t_KmW", "oracle_C", "pinn_C", "error_C"])
        for i in range(n_check):
            w.writerow([
                round(float(Is[i]), 2),
                round(float(ambs[i]), 3),
                round(float(rhos[i]), 4),
                round(float(oracle_T[i]), 4),
                round(float(pinn_T[i]), 4),
                round(float(err[i]), 4),
            ])
    print(f"  Saved validation       : {val_path}")

    # Save model state dict for downstream use
    state_path = out_dir / "pinn_state.pt"
    torch.save(model.state_dict(), state_path)
    print(f"  Saved model            : {state_path}")

    # Plots
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("  (matplotlib not available — skipping figures)")
        return 0 if result.best_val_rmse_C <= target else 1

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    epochs = result.history["epoch"]
    axes[0].plot(epochs, result.history["loss_data"], label="data")
    axes[0].plot(epochs, result.history["loss_physics"], label="physics")
    axes[0].set_yscale("log")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("MSE")
    axes[0].grid(True, alpha=0.3, linestyle=":")
    axes[0].legend()
    axes[0].set_title("Training loss components")

    axes[1].plot(epochs, result.history["val_rmse_C"])
    axes[1].axhline(target, color="r", linestyle="--", linewidth=0.8,
                     label=f"target {target} degC")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Val RMSE [degC]")
    axes[1].grid(True, alpha=0.3, linestyle=":")
    axes[1].legend()
    axes[1].set_title("Validation RMSE")
    fig.tight_layout()
    fig.savefig(out_dir / "training_curves.png", dpi=120)
    plt.close(fig)
    print(f"  Saved fig (curves)     : {out_dir/'training_curves.png'}")

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(oracle_T, pinn_T, s=4, alpha=0.4)
    lo = min(float(oracle_T.min()), float(pinn_T.min()))
    hi = max(float(oracle_T.max()), float(pinn_T.max()))
    ax.plot([lo, hi], [lo, hi], color="r", linestyle="--", linewidth=0.8)
    ax.set_xlabel("Oracle (IEC 60287) T_c [degC]")
    ax.set_ylabel("PINN T_c [degC]")
    ax.set_title(f"Validation scatter — RMSE {rmse:.3f} degC")
    ax.grid(True, alpha=0.3, linestyle=":")
    fig.tight_layout()
    fig.savefig(out_dir / "validation_scatter.png", dpi=120)
    plt.close(fig)
    print(f"  Saved fig (scatter)    : {out_dir/'validation_scatter.png'}")

    return 0 if result.best_val_rmse_C <= target else 1


if __name__ == "__main__":
    raise SystemExit(main())
