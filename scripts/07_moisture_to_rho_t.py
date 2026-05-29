"""
Demo the soil-moisture-to-thermal-resistivity coupling.

Produces:
  * A printed table to stdout (theta vs rho_T for loam / sandy / clay)
  * scripts/output/07_moisture_to_rho_t.csv
  * scripts/output/07_moisture_to_rho_t.png   (if matplotlib available)

Run from the repository root:

    python scripts/07_moisture_to_rho_t.py
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np

# Allow running directly without `pip install -e .`
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gridforge.physics.soil_moisture import (
    CLAY,
    KNOWN_SOIL_TYPES,
    LOAM,
    SANDY,
    theta_array_to_rho_t,
)

THETA_GRID: np.ndarray = np.linspace(0.0, 0.5, 26)  # 0.0, 0.02, ..., 0.50


def main() -> int:
    out_dir = Path(__file__).resolve().parent / "output"
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "07_moisture_to_rho_t.csv"
    png_path = out_dir / "07_moisture_to_rho_t.png"

    # ---- Stdout banner -------------------------------------------------
    print()
    print("Soil moisture -> thermal resistivity coupling")
    print("Form: rho_T(theta) = rho_dry * (rho_sat/rho_dry) ** (theta/theta_sat)")
    print("Constants per soil type (rho_dry, rho_sat, theta_sat):")
    for name in ("loam", "sandy", "clay"):
        s = KNOWN_SOIL_TYPES[name]
        print(
            f"  {s.name:<6} rho_dry={s.rho_dry_KmW:.2f}  "
            f"rho_sat={s.rho_sat_KmW:.2f}  theta_sat={s.theta_sat:.2f}  K.m/W"
        )
    print()

    header = f"{'theta':>6} | {'loam':>9} | {'sandy':>9} | {'clay':>9}   (K.m/W)"
    print(header)
    print("-" * len(header))

    loam = theta_array_to_rho_t(THETA_GRID, LOAM)
    sandy = theta_array_to_rho_t(THETA_GRID, SANDY)
    clay = theta_array_to_rho_t(THETA_GRID, CLAY)

    for i, theta in enumerate(THETA_GRID):
        print(f"{theta:6.2f} | {loam[i]:9.3f} | {sandy[i]:9.3f} | {clay[i]:9.3f}")

    # ---- CSV -----------------------------------------------------------
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["theta", "rho_T_loam_KmW", "rho_T_sandy_KmW", "rho_T_clay_KmW"])
        for i, theta in enumerate(THETA_GRID):
            writer.writerow(
                [
                    f"{theta:.4f}",
                    f"{loam[i]:.6f}",
                    f"{sandy[i]:.6f}",
                    f"{clay[i]:.6f}",
                ]
            )
    print(f"\n[saved] {csv_path}")

    # ---- PNG (optional, only if matplotlib is installed) --------------
    try:
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(THETA_GRID, loam, label="loam (default)", linewidth=2.0)
        ax.plot(THETA_GRID, sandy, label="sandy", linestyle="--", linewidth=2.0)
        ax.plot(THETA_GRID, clay, label="clay", linestyle=":", linewidth=2.0)
        ax.axhline(1.0, color="grey", linewidth=0.6, linestyle="--", alpha=0.6)
        ax.text(0.49, 1.05, "IEC 60287 'damp' = 1.0", ha="right", fontsize=8, color="grey")
        ax.set_xlabel("Volumetric water content theta  [m3/m3]")
        ax.set_ylabel("Soil thermal resistivity rho_T  [K.m/W]")
        ax.set_title("Soil moisture to thermal resistivity coupling")
        ax.legend(loc="upper right")
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0.0, 0.5)
        ax.set_ylim(0.0, 4.0)
        fig.tight_layout()
        fig.savefig(png_path, dpi=140)
        plt.close(fig)
        print(f"[saved] {png_path}")
    except ImportError:
        print("[skip ] matplotlib not installed; skipping PNG")

    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
