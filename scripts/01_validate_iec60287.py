"""
Validate the IEC 60287 steady-state solver on the canonical UK 11 kV
240 mm^2 XLPE 3-core archetype.

Produces:
  * A load-vs-conductor-temperature table to stdout
  * scripts/output/01_load_vs_temperature.csv
  * scripts/output/01_load_vs_temperature.png  (only if matplotlib is available)

Run from the repository root:

    python scripts/01_validate_iec60287.py
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path
from typing import List

# Allow running directly without `pip install -e .`
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gridforge.physics.cable_archetype import (
    UK_11KV_240MM2_XLPE_3CORE,
    UK_TYPICAL_INSTALLATION,
)
from gridforge.physics.thermal import solve_steady_state


def main() -> int:
    geom, mat = UK_11KV_240MM2_XLPE_3CORE
    install = UK_TYPICAL_INSTALLATION

    currents_A: List[float] = list(range(0, 651, 50))
    rows: List[dict] = []

    print()
    print("Validation of IEC 60287 steady-state solver")
    print("Cable:        UK 11 kV 240 mm^2 Cu XLPE 3-core, HDPE jacket")
    print(f"Burial depth: {install.burial_depth_m:.2f} m")
    print(f"Soil rho_T:   {install.soil_thermal_resistivity_KmW:.2f} K.m/W")
    print(f"Ambient soil: {install.ambient_soil_temp_C:.1f} degC")
    print()
    header = (
        f"{'I [A]':>6} | {'theta_c [degC]':>14} | {'theta_sh [degC]':>15} | "
        f"{'I^2 R [W/m]':>11} | {'Total [W/m]':>11} | {'iters':>5} | conv"
    )
    print(header)
    print("-" * len(header))

    for I in currents_A:
        sol = solve_steady_state(
            current_per_phase_A=float(I),
            line_voltage_V_rms=11_000.0,
            geom=geom,
            mat=mat,
            install=install,
        )
        print(
            f"{I:>6} | {sol.conductor_temp_C:>14.2f} | "
            f"{sol.sheath_temp_C:>15.2f} | "
            f"{sol.I2R_loss_W_per_m:>11.3f} | "
            f"{sol.total_loss_W_per_m:>11.3f} | "
            f"{sol.iterations:>5} | {'yes' if sol.converged else 'NO'}"
        )
        rows.append(
            {
                "current_A": I,
                "conductor_C": round(sol.conductor_temp_C, 4),
                "sheath_C": round(sol.sheath_temp_C, 4),
                "jacket_C": round(sol.jacket_temp_C, 4),
                "soil_iface_C": round(sol.soil_interface_temp_C, 4),
                "I2R_W_per_m": round(sol.I2R_loss_W_per_m, 6),
                "W_d_W_per_m": round(sol.W_dielectric_W_per_m, 6),
                "total_W_per_m": round(sol.total_loss_W_per_m, 6),
                "T1_KmW": round(sol.thermal_resistances.T1_KmW, 6),
                "T3_KmW": round(sol.thermal_resistances.T3_KmW, 6),
                "T4_KmW": round(sol.thermal_resistances.T4_KmW, 6),
                "iterations": sol.iterations,
                "converged": sol.converged,
            }
        )

    out_dir = Path(__file__).resolve().parent / "output"
    out_dir.mkdir(parents=True, exist_ok=True)

    csv_path = out_dir / "01_load_vs_temperature.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print()
    print(f"Saved table:  {csv_path}")

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("(matplotlib not installed — skipping figure)")
        return 0

    fig, ax = plt.subplots(figsize=(8, 5))
    Is = [r["current_A"] for r in rows]
    Tc = [r["conductor_C"] for r in rows]
    Ts = [r["sheath_C"] for r in rows]
    ax.plot(Is, Tc, "o-", label="Conductor")
    ax.plot(Is, Ts, "s--", label="Sheath / screen")
    ax.axhline(90.0, color="r", linestyle=":", linewidth=0.8, label="XLPE 90 degC limit")
    ax.set_xlabel("Phase current [A]")
    ax.set_ylabel("Temperature [degC]")
    ax.set_title("UK 11 kV 240 mm^2 XLPE 3-core — IEC 60287 steady-state")
    ax.grid(True, alpha=0.3, linestyle=":")
    ax.legend(loc="upper left")
    fig.tight_layout()

    png_path = out_dir / "01_load_vs_temperature.png"
    fig.savefig(png_path, dpi=120)
    plt.close(fig)
    print(f"Saved figure: {png_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
