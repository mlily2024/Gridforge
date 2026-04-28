"""GridForge benchmark suite — sealed-test tasks + reference baselines + leaderboard."""

from .tasks import (
    ALL_TASKS,
    T1_FAILURE_60D,
    T2_RUL_REGRESSION,
    T3_ANOMALY,
    T4_VIRTUAL_SENSOR,
    T5_COUNTERFACTUAL,
    Task,
)
from .metrics import (
    auc_pr,
    brier_score,
    mae,
    precision_at_recall,
    quantile_loss,
    rmse,
)
from .loader import (
    DatasetView,
    load_mini_dataset,
)
from .baselines import (
    Baseline,
    GradientBoostedBaseline,
    IECOracleBaseline,
    PINNBaseline,
)
from .runner import (
    LeaderboardEntry,
    run_benchmark,
)

__all__ = [
    # tasks
    "Task",
    "T1_FAILURE_60D",
    "T2_RUL_REGRESSION",
    "T3_ANOMALY",
    "T4_VIRTUAL_SENSOR",
    "T5_COUNTERFACTUAL",
    "ALL_TASKS",
    # metrics
    "brier_score",
    "auc_pr",
    "mae",
    "rmse",
    "precision_at_recall",
    "quantile_loss",
    # loader
    "DatasetView",
    "load_mini_dataset",
    # baselines
    "Baseline",
    "IECOracleBaseline",
    "GradientBoostedBaseline",
    "PINNBaseline",
    # runner
    "LeaderboardEntry",
    "run_benchmark",
]
