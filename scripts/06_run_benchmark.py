"""
Run the GridForge benchmark suite (GF-006 + GF-007) end-to-end.

  1. Loads (or regenerates) the mini synthetic dataset under
     scripts/output/mini_dataset/.
  2. Runs three reference baselines (IEC oracle, gradient-boosted trees,
     PINN) across the five sealed-test tasks (T1..T5).
  3. Prints a leaderboard table and writes it to
     scripts/output/06_benchmark/leaderboard.csv

The PINN baseline auto-loads the trained state from Day 4 if it exists;
otherwise it falls back to the IEC oracle path. The gradient-boosted
baseline uses sklearn's HistGradientBoostingRegressor when available;
otherwise it predicts the training-mean.
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gridforge.bench import (
    ALL_TASKS,
    GradientBoostedBaseline,
    IECOracleBaseline,
    PINNBaseline,
    load_mini_dataset,
    run_benchmark,
)


def main() -> int:
    repo_root = Path(__file__).resolve().parents[1]
    dataset_dir = repo_root / "scripts" / "output" / "mini_dataset"
    pinn_state = repo_root / "scripts" / "output" / "05_pinn_training" / "pinn_state.pt"

    if not dataset_dir.exists():
        print(f"Dataset not found: {dataset_dir}")
        print("Generate it first: python scripts/04_generate_mini_dataset.py")
        return 2

    print()
    print("Loading mini synthetic dataset...")
    view = load_mini_dataset(dataset_dir)
    n_train = len(view.by_split("train"))
    n_val = len(view.by_split("val"))
    n_test = len(view.by_split("test"))
    print(f"  Cables: {len(view)}  (train={n_train}, val={n_val}, test={n_test})")

    print()
    print("Building baselines...")
    baselines = [
        IECOracleBaseline(),
        GradientBoostedBaseline(),
        PINNBaseline(pinn_state_path=pinn_state if pinn_state.exists() else None),
    ]
    for b in baselines:
        print(f"  - {b.name}")

    print()
    print("Running benchmark across all 5 tasks...")
    entries = run_benchmark(view, baselines, list(ALL_TASKS))

    # ----- Print leaderboard table -----
    print()
    print(f"=== Leaderboard ({len(entries)} entries, {n_test} test cables) ===")
    print(
        f"{'Baseline':<22} {'Task':<22} {'Metric':<28} {'Score':>12} "
        f"{'n':>6}  Secondary"
    )
    print("-" * 110)
    for e in sorted(entries, key=lambda x: (x.task, x.baseline)):
        sec = " ".join(f"{k}={v:.4g}" for k, v in e.secondary_metrics.items())
        print(
            f"{e.baseline:<22} {e.task:<22} {e.headline_metric_name:<28} "
            f"{e.headline_metric_value:>12.4f} {e.n_samples:>6}  {sec}"
        )

    # ----- Per-task winners -----
    print()
    print("=== Per-task winners (headline metric) ===")
    by_task: dict[str, list] = {}
    for e in entries:
        by_task.setdefault(e.task, []).append(e)
    from gridforge.bench import (
        T1_FAILURE_60D,
        T2_RUL_REGRESSION,
        T3_ANOMALY,
        T4_VIRTUAL_SENSOR,
        T5_COUNTERFACTUAL,
    )
    higher_is_better_map = {
        T1_FAILURE_60D.name: T1_FAILURE_60D.higher_is_better,
        T2_RUL_REGRESSION.name: T2_RUL_REGRESSION.higher_is_better,
        T3_ANOMALY.name: T3_ANOMALY.higher_is_better,
        T4_VIRTUAL_SENSOR.name: T4_VIRTUAL_SENSOR.higher_is_better,
        T5_COUNTERFACTUAL.name: T5_COUNTERFACTUAL.higher_is_better,
    }
    for task_name, ents in sorted(by_task.items()):
        higher = higher_is_better_map.get(task_name, False)
        ents.sort(key=lambda x: x.headline_metric_value, reverse=higher)
        winner = ents[0]
        print(
            f"  {task_name:<22} winner: {winner.baseline:<22} "
            f"{winner.headline_metric_name}={winner.headline_metric_value:.4f}"
        )

    # ----- Save CSV -----
    out_dir = repo_root / "scripts" / "output" / "06_benchmark"
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "leaderboard.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "baseline", "task", "headline_metric", "score",
            "n_samples", "secondary_metrics",
        ])
        for e in entries:
            sec = ";".join(f"{k}={v}" for k, v in e.secondary_metrics.items())
            writer.writerow([
                e.baseline, e.task, e.headline_metric_name,
                e.headline_metric_value, e.n_samples, sec,
            ])
    print()
    print(f"Saved leaderboard: {csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
