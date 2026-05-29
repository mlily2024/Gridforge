"""
Generate a small synthetic dataset spanning all four UK 11 kV archetypes,
all four failure modes, multiple load profiles, and a 5-year horizon.

Spec grid (configurable below):

  archetypes  4   11 kV 240 mm^2 XLPE 3c, 95 mm^2 XLPE 3c, 300 mm^2 XLPE 1c,
                  240 mm^2 PILC 3c
  load profs  4   residential, commercial, industrial, mixed
  failure modes 4 healthy, water_ingress, thermal_ageing, accelerated_dielectric
  seeds       2   per (archetype, profile, mode) combination
  horizon     5 yr  enough for some failure-time outcomes to appear

Total: 4 * 4 * 4 * 2 = 128 cables nominally; capped at MAX_CABLES below to
keep runtime bounded for the demo. A production GF-004 run lifts the cap.

Run from the gridforge subpackage root:

    python scripts/04_generate_mini_dataset.py
"""

from __future__ import annotations

import sys
import time as _time
from itertools import product
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gridforge.data.cable_year import CableYearSpec
from gridforge.data.dataset import assemble_dataset
from gridforge.data.failure_modes import (
    AcceleratedDielectricMode,
    HealthyMode,
    ThermalAgeingMode,
    WaterIngressMode,
)
from gridforge.data.load_profiles import LoadSpec
from gridforge.data.weather import WeatherSpec
from gridforge.physics.cable_archetype import ARCHETYPES

MAX_CABLES = 64
DURATION_YEARS = 5.0


# Per-archetype representative load envelopes (peak_A, base_A) — order-of-
# magnitude scaled to the cable's nameplate rating. Smaller cables carry
# less, larger cables carry more.
LOAD_ENVELOPE = {
    "11kV_240mm2_Cu_XLPE_3c": (350.0, 80.0),
    "11kV_95mm2_Cu_XLPE_3c": (180.0, 40.0),
    "11kV_300mm2_Cu_XLPE_1c": (260.0, 60.0),
    "11kV_240mm2_Cu_PILC_3c": (300.0, 70.0),
}


def _failure_mode_for(name: str, seed: int):
    if name == "healthy":
        return HealthyMode(seed=seed)
    if name == "water_ingress":
        return WaterIngressMode(seed=seed)
    if name == "thermal_ageing":
        return ThermalAgeingMode(seed=seed)
    if name == "accelerated_dielectric":
        return AcceleratedDielectricMode(seed=seed)
    raise ValueError(f"unknown failure mode: {name}")


def _short(name: str) -> str:
    return name.split("_")[1] if "_" in name else name[:4]


def build_specs() -> list[CableYearSpec]:
    archetypes = list(ARCHETYPES.keys())
    profiles = ["residential", "commercial", "industrial", "mixed"]
    modes = ["healthy", "water_ingress", "thermal_ageing", "accelerated_dielectric"]
    seeds = [0]  # one seed per (archetype, profile, mode) — already 64 combos

    # Iteration order with archetype as the OUTERMOST loop guarantees that any
    # truncation by MAX_CABLES still covers complete archetype slices, but in
    # this configuration (4*4*4*1 = 64) MAX_CABLES caps cleanly at full grid.
    specs: list[CableYearSpec] = []
    cable_idx = 0
    for arch, prof, mode, seed in product(archetypes, profiles, modes, seeds):
        peak_A, base_A = LOAD_ENVELOPE[arch]
        cable_id = f"{_short(arch)}_{prof[:3]}_{mode[:3]}_s{seed}_{cable_idx:03d}"
        specs.append(
            CableYearSpec(
                cable_id=cable_id,
                duration_years=DURATION_YEARS,
                load=LoadSpec(prof, peak_A=peak_A, base_A=base_A, seed=cable_idx),
                weather=WeatherSpec(seed=cable_idx),
                failure_mode=_failure_mode_for(mode, cable_idx),
                archetype_name=arch,
            )
        )
        cable_idx += 1
        if len(specs) >= MAX_CABLES:
            return specs
    return specs


def progress(i: int, n: int, cable_id: str) -> None:
    print(f"  [{i+1:>3}/{n}] {cable_id}", flush=True)


def main() -> int:
    out_dir = Path(__file__).resolve().parent / "output" / "mini_dataset"
    if out_dir.exists():
        for p in sorted(out_dir.rglob("*"), reverse=True):
            if p.is_file():
                p.unlink()
            else:
                p.rmdir()
        out_dir.rmdir()

    specs = build_specs()
    print()
    print(
        f"GridForge mini dataset — {len(specs)} cables, "
        f"{DURATION_YEARS:.1f}-year hourly traces across 4 archetypes"
    )
    print(f"Output: {out_dir}")
    print()

    t0 = _time.time()
    summary = assemble_dataset(specs, out_dir, name="gridforge-mini", progress_callback=progress)
    elapsed = _time.time() - t0
    print()
    print(f"Done in {elapsed:.1f}s")
    print(f"Train / Val / Test : {summary.n_train} / {summary.n_val} / {summary.n_test}")
    print(f"Manifest           : {out_dir/'manifest.csv'}")
    print(f"Telemetry          : {out_dir/'telemetry'}/")
    print(f"Ground truth       : {out_dir/'ground_truth/failure_times.csv'}")
    print(f"Summary            : {out_dir/'dataset_summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
