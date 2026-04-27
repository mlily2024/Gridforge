"""
Simulate the canonical UK 11 kV cable under a 24-hour residential load profile.

The profile is a calibrated approximation of UK domestic demand: trough at
04:00, morning peak at 08:00, evening peak at 18:00, ~30 % minimum to ~85 %
maximum of nameplate capacity over a 24-hour cycle.

Outputs:
  scripts/output/02_diurnal_load.csv  (time, I, T_conductor, loss)
  scripts/output/02_diurnal_load.png  (load and conductor temperature traces)
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gridforge.physics.cable_archetype import (
    UK_11KV_240MM2_XLPE_3CORE,
    UK_TYPICAL_INSTALLATION,
)
from gridforge.physics.transient import simulate_transient


def diurnal_current(t_s: float, peak_A: float = 350.0, base_A: float = 100.0) -> float:
    """UK-typical residential load profile, single 24-hour period."""
    hour_of_day = (t_s / 3600.0) % 24.0
    # Two-peak shape using superposed Gaussians around 08:00 and 18:00
    morning = np.exp(-((hour_of_day - 8.0) / 2.0) ** 2)
    evening = np.exp(-((hour_of_day - 18.0) / 2.5) ** 2) * 1.3
    shape = max(morning, evening)
    return float(base_A + (peak_A - base_A) * shape)


def main() -> int:
    geom, mat = UK_11KV_240MM2_XLPE_3CORE
    install = UK_TYPICAL_INSTALLATION

    # Simulate 4 days at 5-minute resolution; report final 2 days
    total_s = 4 * 24 * 3600
    dt_s = 5 * 60
    times = np.arange(0.0, total_s + dt_s, dt_s, dtype=float)

    result = simulate_transient(
        current_profile=lambda t: diurnal_current(t),
        times_s=times,
        line_voltage_V_rms=11_000.0,
        geom=geom,
        mat=mat,
        install=install,
        initial_temp_C=install.ambient_soil_temp_C,
    )

    print()
    print("Transient simulation: UK 11 kV 240 mm^2 XLPE under diurnal load")
    print(f"R_total      : {result.R_total_KmW:.4f} K.m/W")
    print(f"C_cable      : {result.C_c_J_per_K_m:.1f} J/(K.m)")
    print(f"Time const.  : {result.time_constant_s:.0f} s = {result.time_constant_s/60:.1f} min")
    print(f"Days         : {total_s/86400:.0f}")
    print(f"Samples      : {len(times)}")
    print(f"T_min        : {result.conductor_temp_C.min():.2f} degC")
    print(f"T_max        : {result.conductor_temp_C.max():.2f} degC")
    print(f"T_mean       : {result.conductor_temp_C.mean():.2f} degC")

    out_dir = Path(__file__).resolve().parent / "output"
    out_dir.mkdir(parents=True, exist_ok=True)

    csv_path = out_dir / "02_diurnal_load.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["time_h", "current_A", "conductor_C", "total_loss_W_per_m"])
        for t, T, L in zip(times, result.conductor_temp_C, result.total_loss_W_per_m):
            writer.writerow([round(t / 3600.0, 4), round(diurnal_current(float(t)), 2),
                             round(float(T), 4), round(float(L), 4)])
    print(f"Saved table:  {csv_path}")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("(matplotlib not installed — skipping figure)")
        return 0

    times_h = times / 3600.0
    currents = np.array([diurnal_current(float(t)) for t in times])

    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)

    axes[0].plot(times_h, currents, color="#1f77b4", linewidth=1.0)
    axes[0].set_ylabel("Phase current [A]")
    axes[0].grid(True, alpha=0.3, linestyle=":")
    axes[0].set_title("UK 11 kV 240 mm^2 XLPE — diurnal load response")

    axes[1].plot(times_h, result.conductor_temp_C, color="#d62728", linewidth=1.2,
                 label="Conductor")
    axes[1].axhline(install.ambient_soil_temp_C, color="grey", linestyle=":",
                    linewidth=0.7, label="Ambient soil")
    axes[1].axhline(90.0, color="r", linestyle="--", linewidth=0.6,
                    label="XLPE 90 degC limit")
    axes[1].set_xlabel("Time [hours]")
    axes[1].set_ylabel("Temperature [degC]")
    axes[1].grid(True, alpha=0.3, linestyle=":")
    axes[1].legend(loc="upper left", fontsize=8)
    fig.tight_layout()

    png_path = out_dir / "02_diurnal_load.png"
    fig.savefig(png_path, dpi=120)
    plt.close(fig)
    print(f"Saved figure: {png_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
