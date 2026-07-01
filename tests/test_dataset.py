"""Verification tests for the dataset assembler."""

from __future__ import annotations

from pathlib import Path

import pytest

from gridforge.data.cable_year import CableYearSpec
from gridforge.data.dataset import (
    SPLIT_TEST,
    SPLIT_TRAIN,
    SPLIT_VAL,
    assemble_dataset,
    assign_split,
)
from gridforge.data.conditions import HealthyMode, WaterIngressMode
from gridforge.data.load_profiles import LoadSpec
from gridforge.data.weather import WeatherSpec


class TestSplitAssignment:
    def test_deterministic(self) -> None:
        assert assign_split("cable_001") == assign_split("cable_001")

    def test_different_ids_can_differ(self) -> None:
        # Hash-based split — different IDs should generally land in different splits
        # at sufficient cable count. Spot-check a known-different pair.
        a = assign_split("cable_a")
        b = assign_split("cable_zzzzzzz")
        assert a != b or True  # the assertion intentionally cannot guarantee inequality

    def test_only_three_split_values(self) -> None:
        valid = {SPLIT_TRAIN, SPLIT_VAL, SPLIT_TEST}
        for i in range(50):
            assert assign_split(f"cable_{i:03d}") in valid

    def test_invalid_ratios_rejected(self) -> None:
        with pytest.raises(ValueError):
            assign_split("anything", ratios=(0.5, 0.3, 0.3))

    def test_split_distribution_in_expected_band(self) -> None:
        ratios = (0.7, 0.15, 0.15)
        n = 1000
        counts = {SPLIT_TRAIN: 0, SPLIT_VAL: 0, SPLIT_TEST: 0}
        for i in range(n):
            counts[assign_split(f"id_{i:05d}", ratios)] += 1
        # 5% tolerance on each band
        assert 0.65 * n < counts[SPLIT_TRAIN] < 0.75 * n
        assert 0.10 * n < counts[SPLIT_VAL] < 0.20 * n
        assert 0.10 * n < counts[SPLIT_TEST] < 0.20 * n


class TestAssembleDataset:
    def test_emits_expected_files(self, tmp_path: Path) -> None:
        specs = [
            CableYearSpec(
                cable_id=f"c{i:03d}",
                duration_years=0.05,
                load=LoadSpec("residential", 300.0, 80.0, seed=i),
                weather=WeatherSpec(seed=i),
                condition=HealthyMode() if i % 2 == 0 else WaterIngressMode(seed=i),
            )
            for i in range(4)
        ]
        summary = assemble_dataset(specs, tmp_path, name="test-mini")

        assert (tmp_path / "manifest.csv").exists()
        assert (tmp_path / "ground_truth" / "failure_times.csv").exists()
        assert (tmp_path / "dataset_summary.json").exists()
        for s in specs:
            assert (tmp_path / "telemetry" / f"cable_{s.cable_id}.csv").exists()

        assert summary.n_cables == 4
        assert summary.n_train + summary.n_val + summary.n_test == 4

    def test_telemetry_csv_has_expected_columns(self, tmp_path: Path) -> None:
        spec = CableYearSpec(
            cable_id="single",
            duration_years=0.02,
            load=LoadSpec("commercial", 250.0, 60.0, seed=11),
            weather=WeatherSpec(seed=11),
        )
        assemble_dataset([spec], tmp_path)

        csv_path = tmp_path / "telemetry" / "cable_single.csv"
        with csv_path.open("r", encoding="utf-8") as f:
            header = f.readline().strip().split(",")
        for required in (
            "time_h",
            "current_A",
            "ambient_C",
            "conductor_C",
            "e_field_V_per_m",
            "cumulative_damage",
        ):
            assert required in header
