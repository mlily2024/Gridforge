"""
Stdlib-only loader for the GridForge mini synthetic dataset.

Reads the artefact layout produced by `gridforge.data.dataset.assemble_dataset`:

    dataset_dir/
      manifest.csv
      ground_truth/failure_times.csv
      telemetry/cable_<id>.csv

Returns a `DatasetView` exposing per-cable telemetry as numpy arrays plus
a manifest dict per cable. No pandas dependency — keeps the bench module
core-only.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import numpy as np


@dataclass(frozen=True)
class CableRecord:
    cable_id: str
    split: str
    archetype: str
    load_profile: str
    condition: str
    duration_years: float
    sample_period_s: float
    final_damage: float
    failure_time_s: float | None
    times_h: np.ndarray
    current_A: np.ndarray
    ambient_C: np.ndarray
    moisture: np.ndarray
    conductor_C: np.ndarray
    e_field_V_per_m: np.ndarray
    pd_rate_relative: np.ndarray
    cumulative_damage: np.ndarray
    manifest: dict = field(default_factory=dict)


@dataclass(frozen=True)
class DatasetView:
    """Loaded mini-dataset view for benchmark consumption."""

    name: str
    dataset_dir: Path
    cables: dict[str, CableRecord]

    @property
    def cable_ids(self) -> tuple[str, ...]:
        return tuple(self.cables.keys())

    def by_split(self, split: str) -> tuple[CableRecord, ...]:
        return tuple(c for c in self.cables.values() if c.split == split)

    def __len__(self) -> int:
        return len(self.cables)


def _read_csv_columns(path: Path) -> dict[str, list[str]]:
    """Read a CSV file into a dict mapping column name -> list of strings."""
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        cols: dict[str, list[str]] = {name: [] for name in (reader.fieldnames or [])}
        for row in reader:
            for k, v in row.items():
                cols[k].append(v)
    return cols


def load_mini_dataset(dataset_dir: Path | str) -> DatasetView:
    """Read the dataset artefacts produced by `assemble_dataset`."""
    dataset_dir = Path(dataset_dir)
    if not dataset_dir.exists():
        raise FileNotFoundError(f"dataset directory not found: {dataset_dir}")

    manifest_path = dataset_dir / "manifest.csv"
    truth_path = dataset_dir / "ground_truth" / "failure_times.csv"
    telem_dir = dataset_dir / "telemetry"
    for p in (manifest_path, truth_path, telem_dir):
        if not p.exists():
            raise FileNotFoundError(f"missing dataset artefact: {p}")

    manifest_cols = _read_csv_columns(manifest_path)
    truth_cols = _read_csv_columns(truth_path)

    n_cables = len(manifest_cols.get("cable_id", []))
    truth_by_id: dict[str, dict[str, str]] = {}
    for i, cid in enumerate(truth_cols.get("cable_id", [])):
        truth_by_id[cid] = {k: v[i] for k, v in truth_cols.items()}

    cables: dict[str, CableRecord] = {}
    for i in range(n_cables):
        cid = manifest_cols["cable_id"][i]
        manifest_row = {k: v[i] for k, v in manifest_cols.items()}
        truth_row = truth_by_id.get(cid, {})

        telemetry_path = telem_dir / f"cable_{cid}.csv"
        if not telemetry_path.exists():
            raise FileNotFoundError(f"missing telemetry for cable {cid}: {telemetry_path}")

        telem = _read_csv_columns(telemetry_path)
        cables[cid] = CableRecord(
            cable_id=cid,
            split=manifest_row.get("split", "train"),
            archetype=manifest_row.get("archetype", ""),
            load_profile=manifest_row.get("load_profile", ""),
            condition=manifest_row.get("condition", ""),
            duration_years=float(manifest_row.get("duration_years", "0") or 0),
            sample_period_s=float(manifest_row.get("sample_period_s", "0") or 0),
            final_damage=float(truth_row.get("final_damage", "0") or 0),
            failure_time_s=(
                float(truth_row["failure_time_s"])
                if truth_row.get("failure_time_s") not in (None, "", "None")
                else None
            ),
            times_h=np.array(telem.get("time_h", []), dtype=np.float64),
            current_A=np.array(telem.get("current_A", []), dtype=np.float64),
            ambient_C=np.array(telem.get("ambient_C", []), dtype=np.float64),
            moisture=np.array(telem.get("moisture", []), dtype=np.float64),
            conductor_C=np.array(telem.get("conductor_C", []), dtype=np.float64),
            e_field_V_per_m=np.array(telem.get("e_field_V_per_m", []), dtype=np.float64),
            pd_rate_relative=np.array(telem.get("pd_rate_relative", []), dtype=np.float64),
            cumulative_damage=np.array(telem.get("cumulative_damage", []), dtype=np.float64),
            manifest=manifest_row,
        )

    return DatasetView(
        name="gridforge-mini",
        dataset_dir=dataset_dir,
        cables=cables,
    )


def stack_features(records: Iterable[CableRecord]) -> tuple[np.ndarray, np.ndarray]:
    """Concatenate per-cable telemetry into a single (N, 3) feature matrix
    and matched (N,) target vector.

    Features: (current_A, ambient_C, soil_rho_t_KmW)
    Target  : conductor_C (the IEC oracle ground truth from the dataset)

    Soil rho_t is read from the cable's manifest entry; it's constant
    within a cable but varies between cables.
    """
    Xs, ys = [], []
    for r in records:
        rho_t = float(r.manifest.get("rho_t_KmW", "1.0") or 1.0)
        # Manifest doesn't currently include rho_t — substitute the default
        # archetype's value if missing.
        if "rho_t_KmW" not in r.manifest:
            rho_t = 1.0
        n = r.times_h.size
        Xs.append(
            np.stack(
                [
                    r.current_A,
                    r.ambient_C,
                    np.full(n, rho_t),
                ],
                axis=-1,
            )
        )
        ys.append(r.conductor_C.copy())
    if not Xs:
        return np.zeros((0, 3)), np.zeros(0)
    return np.concatenate(Xs, axis=0), np.concatenate(ys, axis=0)
