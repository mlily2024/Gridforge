"""
Benchmark runner — orchestrates predictions across baselines and tasks,
computes metrics, and assembles the leaderboard.

The runner is intentionally simple: it walks the cartesian product of
(baselines, tasks), calls each baseline's `predict(...)` once per task,
constructs ground-truth labels for the test split, and evaluates the
configured metric on the predictions.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np

from . import metrics as M
from .baselines import (
    DEFAULT_DAMAGE_THRESHOLD,
    HORIZON_HOURS,
    WINDOW_HOURS,
    Baseline,
)
from .loader import DatasetView
from .tasks import (
    T1_FAILURE_60D,
    T2_RUL_REGRESSION,
    T3_ANOMALY,
    T4_VIRTUAL_SENSOR,
    T5_COUNTERFACTUAL,
    Task,
)

SECONDS_PER_YEAR = 365.25 * 24.0 * 3600.0


@dataclass(frozen=True)
class LeaderboardEntry:
    baseline: str
    task: str
    headline_metric_name: str
    headline_metric_value: float
    secondary_metrics: dict = field(default_factory=dict)
    n_samples: int = 0
    note: str = ""


# ---------------------------------------------------------------------------
# Ground-truth construction per task
# ---------------------------------------------------------------------------


def _t1_labels(view: DatasetView, damage_threshold: float) -> tuple[dict, dict]:
    """Per-cable binary label: did damage cross threshold in the last
    HORIZON_HOURS of telemetry, given the WINDOW_HOURS preceding it?"""
    labels: dict[str, int] = {}
    valid: dict[str, bool] = {}
    for cid, rec in view.cables.items():
        n = rec.cumulative_damage.size
        if n < WINDOW_HOURS + HORIZON_HOURS:
            valid[cid] = False
            labels[cid] = 0
            continue
        d_window_end = rec.cumulative_damage[n - HORIZON_HOURS]
        d_final = rec.cumulative_damage[-1]
        # Crossed the threshold during the horizon?
        labels[cid] = int(d_final >= damage_threshold and d_window_end < damage_threshold)
        valid[cid] = True
    return labels, valid


def _t2_labels(view: DatasetView, damage_threshold: float) -> tuple[dict, dict]:
    """Per-cable RUL ground truth in years (time from t=0 to threshold).

    Cables that never reach the threshold are flagged invalid (right-censored).
    """
    labels: dict[str, float] = {}
    valid: dict[str, bool] = {}
    for cid, rec in view.cables.items():
        above = rec.cumulative_damage >= damage_threshold
        if not above.any():
            valid[cid] = False
            labels[cid] = float("nan")
            continue
        idx = int(np.argmax(above))
        rul_years = float(rec.times_h[idx]) * 3600.0 / SECONDS_PER_YEAR
        labels[cid] = rul_years
        valid[cid] = True
    return labels, valid


def _t3_labels(view: DatasetView) -> dict[str, np.ndarray]:
    """Per-hour binary anomaly label per cable.

    Positives: hours where pd_rate_relative > 1.5 (indicates a switching
    impulse) or where the cable is in a thermal_ageing failure mode (every
    hour during sustained overheat is "anomalous" in the engineering sense).
    """
    out: dict[str, np.ndarray] = {}
    for cid, rec in view.cables.items():
        n = rec.times_h.size
        labels = np.zeros(n, dtype=np.int32)
        # Impulse-driven positives
        labels[rec.pd_rate_relative > 1.5] = 1
        # Thermal_ageing positives (per cable)
        if rec.failure_mode == "thermal_ageing":
            labels[:] = 1
        out[cid] = labels
    return out


def _t5_labels(view: DatasetView) -> dict[str, np.ndarray]:
    """Per-cable 3-vector of ground-truth driver attribution.

    Synthetic ground truth for v0.0.x: assume 50% load, 30% ambient, 20%
    failure-mode driver as an industry-typical decomposition. The cable's
    own failure_mode tilts the third component (for non-healthy modes).
    """
    out: dict[str, np.ndarray] = {}
    for cid, rec in view.cables.items():
        if rec.failure_mode == "healthy":
            out[cid] = np.array([0.6, 0.4, 0.0])
        else:
            out[cid] = np.array([0.5, 0.3, 0.2])
    return out


# ---------------------------------------------------------------------------
# Per-task evaluators
# ---------------------------------------------------------------------------


def _evaluate_t1(
    view: DatasetView, predictions: dict, damage_threshold: float
) -> LeaderboardEntry | None:
    labels_dict, valid_dict = _t1_labels(view, damage_threshold)
    test_records = view.by_split("test")
    y_true, y_pred = [], []
    for rec in test_records:
        if not valid_dict.get(rec.cable_id, False):
            continue
        if rec.cable_id not in predictions:
            continue
        y_true.append(labels_dict[rec.cable_id])
        y_pred.append(float(predictions[rec.cable_id]))
    if not y_true:
        return None
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    brier = M.brier_score(y_true, y_pred)
    auc = M.auc_pr(y_true, y_pred)
    return LeaderboardEntry(
        baseline="",
        task=T1_FAILURE_60D.name,
        headline_metric_name=T1_FAILURE_60D.headline_metric,
        headline_metric_value=brier,
        secondary_metrics={"auc_pr": auc},
        n_samples=len(y_true),
    )


def _evaluate_t2(
    view: DatasetView, predictions: dict, damage_threshold: float
) -> LeaderboardEntry | None:
    labels_dict, valid_dict = _t2_labels(view, damage_threshold)
    test_records = view.by_split("test")
    y_true, y_pred = [], []
    for rec in test_records:
        if not valid_dict.get(rec.cable_id, False):
            continue
        pred = predictions.get(rec.cable_id)
        if pred is None or not np.isfinite(pred):
            continue
        y_true.append(labels_dict[rec.cable_id])
        y_pred.append(float(pred))
    if not y_true:
        return None
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    return LeaderboardEntry(
        baseline="",
        task=T2_RUL_REGRESSION.name,
        headline_metric_name=T2_RUL_REGRESSION.headline_metric,
        headline_metric_value=M.mae(y_true, y_pred),
        secondary_metrics={
            "rmse": M.rmse(y_true, y_pred),
            "quantile_50": M.quantile_loss(y_true, y_pred, q=0.5),
        },
        n_samples=len(y_true),
    )


def _evaluate_t3(view: DatasetView, predictions: dict) -> LeaderboardEntry | None:
    labels_dict = _t3_labels(view)
    test_records = view.by_split("test")
    y_true, y_score = [], []
    for rec in test_records:
        if rec.cable_id not in predictions:
            continue
        y_true.append(labels_dict[rec.cable_id])
        y_score.append(predictions[rec.cable_id])
    if not y_true:
        return None
    y_true_flat = np.concatenate(y_true)
    y_score_flat = np.concatenate(y_score)
    return LeaderboardEntry(
        baseline="",
        task=T3_ANOMALY.name,
        headline_metric_name=T3_ANOMALY.headline_metric,
        headline_metric_value=M.precision_at_recall(y_true_flat, y_score_flat, 0.9),
        secondary_metrics={"auc_pr": M.auc_pr(y_true_flat, y_score_flat)},
        n_samples=len(y_true_flat),
    )


def _evaluate_t4(view: DatasetView, predictions: dict) -> LeaderboardEntry | None:
    test_records = view.by_split("test")
    y_true, y_pred = [], []
    for rec in test_records:
        if rec.cable_id not in predictions:
            continue
        y_true.append(rec.conductor_C)
        y_pred.append(predictions[rec.cable_id])
    if not y_true:
        return None
    y_true_flat = np.concatenate(y_true)
    y_pred_flat = np.concatenate(y_pred)
    return LeaderboardEntry(
        baseline="",
        task=T4_VIRTUAL_SENSOR.name,
        headline_metric_name=T4_VIRTUAL_SENSOR.headline_metric,
        headline_metric_value=M.rmse(y_true_flat, y_pred_flat),
        secondary_metrics={"mae": M.mae(y_true_flat, y_pred_flat)},
        n_samples=len(y_true_flat),
    )


def _evaluate_t5(view: DatasetView, predictions: dict) -> LeaderboardEntry | None:
    labels_dict = _t5_labels(view)
    test_records = view.by_split("test")
    y_true, y_pred = [], []
    for rec in test_records:
        if rec.cable_id not in predictions:
            continue
        y_true.append(labels_dict[rec.cable_id])
        y_pred.append(predictions[rec.cable_id])
    if not y_true:
        return None
    y_true_flat = np.concatenate(y_true)
    y_pred_flat = np.concatenate(y_pred)
    return LeaderboardEntry(
        baseline="",
        task=T5_COUNTERFACTUAL.name,
        headline_metric_name=T5_COUNTERFACTUAL.headline_metric,
        headline_metric_value=M.mae(y_true_flat, y_pred_flat),
        secondary_metrics={"rmse": M.rmse(y_true_flat, y_pred_flat)},
        n_samples=len(y_true_flat),
    )


_EVALUATORS = {
    T1_FAILURE_60D.name: _evaluate_t1,
    T2_RUL_REGRESSION.name: _evaluate_t2,
    T3_ANOMALY.name: _evaluate_t3,
    T4_VIRTUAL_SENSOR.name: _evaluate_t4,
    T5_COUNTERFACTUAL.name: _evaluate_t5,
}


def run_benchmark(
    view: DatasetView,
    baselines: Sequence[Baseline],
    tasks: Sequence[Task],
    damage_threshold: float = DEFAULT_DAMAGE_THRESHOLD,
) -> list[LeaderboardEntry]:
    """Run all (baseline, task) pairs and return the populated leaderboard."""
    entries: list[LeaderboardEntry] = []
    for b in baselines:
        for t in tasks:
            preds = b.predict(view, t, damage_threshold=damage_threshold)
            evaluator = _EVALUATORS[t.name]
            if t.name in {T1_FAILURE_60D.name, T2_RUL_REGRESSION.name}:
                entry = evaluator(view, preds, damage_threshold)
            else:
                entry = evaluator(view, preds)
            if entry is None:
                continue
            entries.append(
                LeaderboardEntry(
                    baseline=b.name,
                    task=entry.task,
                    headline_metric_name=entry.headline_metric_name,
                    headline_metric_value=entry.headline_metric_value,
                    secondary_metrics=entry.secondary_metrics,
                    n_samples=entry.n_samples,
                    note=entry.note,
                )
            )
    return entries
