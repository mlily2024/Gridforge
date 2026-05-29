"""End-to-end smoke tests for the benchmark runner with the IEC oracle baseline."""

from __future__ import annotations

import math

import pytest

from gridforge.bench import (
    ALL_TASKS,
    T4_VIRTUAL_SENSOR,
    IECOracleBaseline,
    load_mini_dataset,
    run_benchmark,
)
from gridforge.data.cable_year import CableYearSpec
from gridforge.data.dataset import assemble_dataset
from gridforge.data.failure_modes import (
    AcceleratedDielectricMode,
    HealthyMode,
    ThermalAgeingMode,
)
from gridforge.data.load_profiles import LoadSpec
from gridforge.data.weather import WeatherSpec


@pytest.fixture(scope="module")
def runner_dataset(tmp_path_factory):
    out_dir = tmp_path_factory.mktemp("runner_dataset")
    # Mix failure modes to exercise T3 and T5 ground-truth construction
    modes = [HealthyMode, ThermalAgeingMode, AcceleratedDielectricMode]
    specs = []
    # 20 cables gives a reliable non-empty test split (15% of 20 ~ 3) under
    # the SHA-256-based deterministic assignment.
    for i in range(20):
        m_cls = modes[i % len(modes)]
        specs.append(
            CableYearSpec(
                cable_id=f"r{i:02d}",
                duration_years=0.05,
                load=LoadSpec("residential", peak_A=300.0, base_A=80.0, seed=i),
                weather=WeatherSpec(seed=i),
                failure_mode=m_cls(seed=i),
            )
        )
    assemble_dataset(specs, out_dir)
    return out_dir


class TestRunner:
    def test_runs_to_completion_with_oracle(self, runner_dataset) -> None:
        view = load_mini_dataset(runner_dataset)
        entries = run_benchmark(view, [IECOracleBaseline()], list(ALL_TASKS))
        # At least T4 should always produce an entry — the test split is
        # non-empty in any deterministic split assignment for 6 cables.
        task_names = {e.task for e in entries}
        # T4 is always evaluable
        assert T4_VIRTUAL_SENSOR.name in task_names

    def test_t4_oracle_achieves_near_zero_rmse(self, runner_dataset) -> None:
        view = load_mini_dataset(runner_dataset)
        entries = run_benchmark(view, [IECOracleBaseline()], [T4_VIRTUAL_SENSOR])
        e = next(e for e in entries if e.task == T4_VIRTUAL_SENSOR.name)
        # The dataset's conductor_C IS the IEC oracle's prediction. A small
        # numerical-tolerance residual is expected.
        assert e.headline_metric_value < 0.05

    def test_leaderboard_entry_shape(self, runner_dataset) -> None:
        view = load_mini_dataset(runner_dataset)
        entries = run_benchmark(view, [IECOracleBaseline()], [T4_VIRTUAL_SENSOR])
        assert len(entries) >= 1
        e = entries[0]
        assert e.baseline == "IEC_Oracle"
        assert e.headline_metric_name == "rmse"
        assert e.n_samples > 0
        assert math.isfinite(e.headline_metric_value)
