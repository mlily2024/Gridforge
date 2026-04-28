"""
Dataset assembler — produces multi-cable-year synthetic datasets ready for
benchmark and PINN training.

Output layout (compatible with the Day-7 benchmark suite):

    out_dir/
      manifest.csv                  one row per cable with metadata
      telemetry/
        cable_<id>.csv              hourly time series per cable
      ground_truth/
        failure_times.csv           cable_id, failure_time_s, failure_year

Splits are deterministic: cable IDs are hashed and assigned to train / val /
test by hash modulo. Sealed test labels (failure-time ground truth for the
test split) are emitted as a separate file the user does not see during
benchmarking.

Parquet output is intentionally deferred: pandas + pyarrow is heavier than
the v0.0.x dependency budget. CSV output with optional gzip compression is
plenty for a few hundred cable-years and converts cleanly to Parquet later
with a one-line dataframe write.
"""

from __future__ import annotations

import csv
import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterable, List, Sequence

import numpy as np

from .cable_year import CableYearResult, CableYearSpec, simulate_cable_year


SPLIT_TRAIN: str = "train"
SPLIT_VAL: str = "val"
SPLIT_TEST: str = "test"
DEFAULT_SPLIT_RATIOS: tuple[float, float, float] = (0.7, 0.15, 0.15)


def assign_split(cable_id: str, ratios: tuple[float, float, float] = DEFAULT_SPLIT_RATIOS) -> str:
    """Deterministic split assignment by hash of cable ID."""
    if abs(sum(ratios) - 1.0) > 1e-9:
        raise ValueError(f"split ratios must sum to 1.0, got {sum(ratios)}")
    h = int(hashlib.sha256(cable_id.encode()).hexdigest()[:8], 16) / 0xFFFFFFFF
    if h < ratios[0]:
        return SPLIT_TRAIN
    if h < ratios[0] + ratios[1]:
        return SPLIT_VAL
    return SPLIT_TEST


@dataclass(frozen=True)
class DatasetSummary:
    """Descriptor of the produced dataset."""

    name: str
    n_cables: int
    n_train: int
    n_val: int
    n_test: int
    duration_years: float
    sample_period_s: float
    columns: List[str]
    output_dir: str


TELEMETRY_COLUMNS: tuple[str, ...] = (
    "time_h",
    "current_A",
    "ambient_C",
    "moisture",
    "conductor_C",
    "e_field_V_per_m",
    "pd_rate_relative",
    "cumulative_damage",
)


def write_telemetry(out_dir: Path, result: CableYearResult) -> Path:
    """Emit one hourly-resolution CSV per cable."""
    out_path = out_dir / f"cable_{result.cable_id}.csv"
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(TELEMETRY_COLUMNS)
        for i in range(len(result.times_s)):
            writer.writerow([
                round(float(result.times_s[i] / 3600.0), 4),
                round(float(result.current_A[i]), 3),
                round(float(result.ambient_C[i]), 3),
                round(float(result.moisture[i]), 4),
                round(float(result.conductor_C[i]), 3),
                round(float(result.e_field_V_per_m[i]), 1),
                round(float(result.pd_rate_relative[i]), 4),
                float(result.cumulative_damage[i]),
            ])
    return out_path


def assemble_dataset(
    specs: Sequence[CableYearSpec],
    output_dir: Path,
    name: str = "gridforge-mini",
    split_ratios: tuple[float, float, float] = DEFAULT_SPLIT_RATIOS,
    progress_callback=None,
) -> DatasetSummary:
    """Run all cable-year simulations and write the dataset to disk."""
    output_dir.mkdir(parents=True, exist_ok=True)
    telem_dir = output_dir / "telemetry"
    telem_dir.mkdir(exist_ok=True)
    truth_dir = output_dir / "ground_truth"
    truth_dir.mkdir(exist_ok=True)

    manifest_rows: List[dict] = []
    truth_rows: List[dict] = []

    for i, spec in enumerate(specs):
        if progress_callback is not None:
            progress_callback(i, len(specs), spec.cable_id)
        result = simulate_cable_year(spec)
        write_telemetry(telem_dir, result)

        split = assign_split(spec.cable_id, split_ratios)
        failure_year = (
            result.failure_time_s / (365.25 * 24 * 3600)
            if result.failure_time_s is not None
            else None
        )

        manifest_rows.append({
            "cable_id": spec.cable_id,
            "split": split,
            "archetype": result.archetype_name,
            "load_profile": spec.load.profile_name,
            "load_peak_A": spec.load.peak_A,
            "load_base_A": spec.load.base_A,
            "load_seed": spec.load.seed,
            "weather_seed": spec.weather.seed,
            "failure_mode": spec.failure_mode.name,
            "duration_years": spec.duration_years,
            "sample_period_s": spec.sample_period_s,
            "line_voltage_V_rms": spec.line_voltage_V_rms,
            "R_total_KmW": round(result.R_total_KmW, 6),
            "C_cable_J_per_K_m": round(result.C_cable_J_per_K_m, 2),
            "time_constant_s": round(result.time_constant_s, 1),
            "n_samples": len(result.times_s),
        })
        truth_rows.append({
            "cable_id": spec.cable_id,
            "split": split,
            "failure_time_s": result.failure_time_s,
            "failure_year": failure_year,
            "final_damage": float(result.cumulative_damage[-1]),
        })

    manifest_path = output_dir / "manifest.csv"
    with manifest_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(manifest_rows[0].keys()))
        writer.writeheader()
        writer.writerows(manifest_rows)

    truth_path = truth_dir / "failure_times.csv"
    with truth_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(truth_rows[0].keys()))
        writer.writeheader()
        writer.writerows(truth_rows)

    summary = DatasetSummary(
        name=name,
        n_cables=len(specs),
        n_train=sum(1 for r in manifest_rows if r["split"] == SPLIT_TRAIN),
        n_val=sum(1 for r in manifest_rows if r["split"] == SPLIT_VAL),
        n_test=sum(1 for r in manifest_rows if r["split"] == SPLIT_TEST),
        duration_years=specs[0].duration_years,
        sample_period_s=specs[0].sample_period_s,
        columns=list(TELEMETRY_COLUMNS),
        output_dir=str(output_dir),
    )

    summary_path = output_dir / "dataset_summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(asdict(summary), f, indent=2)

    return summary
