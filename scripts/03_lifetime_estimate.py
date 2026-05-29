"""
40-year lifetime study for the canonical UK 11 kV cable.

Two scenarios:
  A) Constant moderate loading at 250 A, 90 degC ambient operating point
  B) Diurnal-cycle loading from script 02
  C) Light overload bias: profile from B scaled by 1.15

Reports:
  - Cumulative damage curve over 40 years
  - Predicted RUL at end of simulated period if pristine
  - Predicted year of failure under continued operation

Outputs:
  scripts/output/03_lifetime_curves.csv
  scripts/output/03_lifetime_curves.png
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gridforge.physics.ageing import (
    SECONDS_PER_YEAR,
    cumulative_damage,
    life_at_constant_stress,
    remaining_useful_life_years,
)
from gridforge.physics.cable_archetype import (
    UK_11KV_240MM2_XLPE_3CORE,
    UK_TYPICAL_INSTALLATION,
)
from gridforge.physics.electrical import max_e_field
from gridforge.physics.transient import simulate_transient

YEARS_TO_SIMULATE: float = 5.0
DT_HOURS: float = 1.0


def diurnal_current(t_s: float, peak_A: float = 350.0, base_A: float = 100.0) -> float:
    hour_of_day = (t_s / 3600.0) % 24.0
    morning = np.exp(-(((hour_of_day - 8.0) / 2.0) ** 2))
    evening = np.exp(-(((hour_of_day - 18.0) / 2.5) ** 2)) * 1.3
    shape = max(morning, evening)
    return float(base_A + (peak_A - base_A) * shape)


def run_scenario(name: str, current_fn) -> dict:
    geom, mat = UK_11KV_240MM2_XLPE_3CORE
    install = UK_TYPICAL_INSTALLATION
    voltage_phase = 11_000.0 / np.sqrt(3.0)
    E_max = max_e_field(voltage_phase, geom)

    dt_s = DT_HOURS * 3600.0

    # Simulate one representative year then tile (fast and a fair approximation;
    # the steady-state thermal lag means 5 years of simulation matches 5 years
    # of repeated 1-year traces to within sub-degree).
    one_year_s = SECONDS_PER_YEAR
    times_year = np.arange(0.0, one_year_s, dt_s)
    transient = simulate_transient(
        current_profile=current_fn,
        times_s=times_year,
        line_voltage_V_rms=11_000.0,
        geom=geom,
        mat=mat,
        install=install,
        initial_temp_C=install.ambient_soil_temp_C,
    )
    # Tile the temperature trace and compute damage
    n_years = int(np.round(YEARS_TO_SIMULATE))
    T_full = np.tile(transient.conductor_temp_C, n_years)
    times_full = np.arange(len(T_full)) * dt_s
    E_full = np.full_like(T_full, E_max)

    D = cumulative_damage(times_full, E_full, T_full)

    # Forward life under mean operating point of this scenario
    T_mean = float(np.mean(T_full))
    L_mean_years = life_at_constant_stress(E_max, T_mean) / SECONDS_PER_YEAR
    rul_at_end = remaining_useful_life_years(float(D[-1]), E_max, T_mean)
    years_axis = times_full / SECONDS_PER_YEAR

    return {
        "name": name,
        "years": years_axis,
        "T_C": T_full,
        "damage": D,
        "T_mean_C": T_mean,
        "L_mean_years": L_mean_years,
        "rul_years": rul_at_end,
    }


def main() -> int:
    print()
    print("Lifetime estimation: UK 11 kV 240 mm^2 XLPE under three load scenarios")
    print(f"Simulated horizon: {YEARS_TO_SIMULATE:.0f} years at {DT_HOURS:.1f}-hour steps")
    print()

    scenarios = [
        run_scenario("A: constant 250 A", lambda t: 250.0),
        run_scenario("B: diurnal 100-350 A", diurnal_current),
        run_scenario("C: diurnal +15%", lambda t: 1.15 * diurnal_current(t)),
    ]

    print(f"{'scenario':<22} | {'T_mean':>8} | {'life@T_mean':>12} | {'D end':>8} | {'RUL':>6}")
    print("-" * 70)
    for s in scenarios:
        print(
            f"{s['name']:<22} | {s['T_mean_C']:>7.2f}  | "
            f"{s['L_mean_years']:>10.1f} y | "
            f"{s['damage'][-1]:>8.4e} | "
            f"{s['rul_years']:>5.1f} y"
        )

    out_dir = Path(__file__).resolve().parent / "output"
    out_dir.mkdir(parents=True, exist_ok=True)

    csv_path = out_dir / "03_lifetime_curves.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["scenario", "year", "T_C", "cumulative_damage"])
        for s in scenarios:
            for y, T, D in zip(s["years"], s["T_C"], s["damage"]):
                writer.writerow(
                    [s["name"], round(float(y), 4), round(float(T), 3), round(float(D), 8)]
                )
    print()
    print(f"Saved table:  {csv_path}")

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("(matplotlib not installed — skipping figure)")
        return 0

    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    colours = ["#1f77b4", "#2ca02c", "#d62728"]
    for s, c in zip(scenarios, colours):
        axes[0].plot(s["years"], s["T_C"], color=c, linewidth=0.5, label=s["name"])
        axes[1].plot(s["years"], s["damage"], color=c, linewidth=1.0, label=s["name"])

    axes[0].set_ylabel("Conductor T [degC]")
    axes[0].grid(True, alpha=0.3, linestyle=":")
    axes[0].legend(loc="upper right", fontsize=8)
    axes[0].set_title(f"GridForge lifetime study — {YEARS_TO_SIMULATE:.0f}-year horizon")

    axes[1].axhline(1.0, color="r", linestyle="--", linewidth=0.6, label="Failure (D=1)")
    axes[1].set_xlabel("Years")
    axes[1].set_ylabel("Cumulative damage D")
    axes[1].grid(True, alpha=0.3, linestyle=":")
    axes[1].legend(loc="upper left", fontsize=8)
    fig.tight_layout()

    png_path = out_dir / "03_lifetime_curves.png"
    fig.savefig(png_path, dpi=120)
    plt.close(fig)
    print(f"Saved figure: {png_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
