"""Tests for the bench loader using a tiny in-memory dataset fixture."""

from __future__ import annotations

import pytest

from gridforge.bench.loader import load_mini_dataset, stack_features
from gridforge.data.cable_year import CableYearSpec
from gridforge.data.conditions import HealthyMode
from gridforge.data.dataset import assemble_dataset
from gridforge.data.load_profiles import LoadSpec
from gridforge.data.weather import WeatherSpec


@pytest.fixture(scope="module")
def tiny_dataset(tmp_path_factory):
    out_dir = tmp_path_factory.mktemp("bench_dataset")
    specs = [
        CableYearSpec(
            cable_id=f"smoke_{i:02d}",
            duration_years=0.05,
            load=LoadSpec("residential", peak_A=300.0, base_A=80.0, seed=i),
            weather=WeatherSpec(seed=i),
            condition=HealthyMode(seed=i),
        )
        for i in range(4)
    ]
    assemble_dataset(specs, out_dir, name="bench-test")
    return out_dir


class TestLoader:
    def test_loads_all_cables(self, tiny_dataset) -> None:
        view = load_mini_dataset(tiny_dataset)
        assert len(view) == 4
        assert all(cid.startswith("smoke_") for cid in view.cable_ids)

    def test_per_cable_arrays_have_expected_size(self, tiny_dataset) -> None:
        view = load_mini_dataset(tiny_dataset)
        for rec in view.cables.values():
            n = rec.times_h.size
            assert rec.current_A.size == n
            assert rec.ambient_C.size == n
            assert rec.conductor_C.size == n
            assert rec.cumulative_damage.size == n
            assert n > 0

    def test_split_grouping(self, tiny_dataset) -> None:
        view = load_mini_dataset(tiny_dataset)
        total = sum(len(view.by_split(s)) for s in ("train", "val", "test"))
        assert total == len(view)

    def test_stack_features(self, tiny_dataset) -> None:
        view = load_mini_dataset(tiny_dataset)
        X, y = stack_features(view.cables.values())
        assert X.shape[1] == 3
        assert X.shape[0] == y.shape[0]
        assert X.shape[0] > 0

    def test_missing_dir_raises(self, tmp_path) -> None:
        with pytest.raises(FileNotFoundError):
            load_mini_dataset(tmp_path / "does_not_exist")
