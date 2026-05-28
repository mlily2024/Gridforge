"""
Numpy-only evaluation metrics for the GridForge benchmark.

No sklearn dependency. The implementations are short, exact, and easy to
audit. They handle the common edge cases (empty inputs, all-positive or
all-negative labels, ties in ranking) deterministically.
"""

from __future__ import annotations

import numpy as np

# `np.trapezoid` is the NumPy >= 2.0 name for the trapezoidal integrator;
# `np.trapz` is the < 2.0 spelling (deprecated in 2.0). Resolve once so the
# package works on its declared numpy>=1.26 floor and on numpy 2.x alike.
_trapezoid = getattr(np, "trapezoid", None) or np.trapz


def _to_array(x) -> np.ndarray:
    return np.asarray(x, dtype=np.float64).ravel()


def brier_score(y_true: np.ndarray, y_proba: np.ndarray) -> float:
    """Mean squared error between predicted probabilities and 0/1 labels."""
    y = _to_array(y_true)
    p = _to_array(y_proba)
    if y.size == 0:
        return float("nan")
    if y.shape != p.shape:
        raise ValueError(f"shape mismatch: y_true {y.shape} vs y_proba {p.shape}")
    return float(np.mean((y - p) ** 2))


def auc_pr(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Area under the precision-recall curve via a discrete trapezoidal rule.

    Returns NaN if there are no positives.
    """
    y = _to_array(y_true)
    s = _to_array(y_score)
    if y.size == 0 or y.shape != s.shape:
        return float("nan")
    n_pos = int(np.sum(y == 1))
    if n_pos == 0:
        return float("nan")

    # Sort by score descending
    order = np.argsort(-s, kind="stable")
    y_sorted = y[order]

    tp_cum = np.cumsum(y_sorted == 1).astype(np.float64)
    fp_cum = np.cumsum(y_sorted == 0).astype(np.float64)
    precision = tp_cum / np.maximum(tp_cum + fp_cum, 1e-12)
    recall = tp_cum / float(n_pos)

    # Prepend (recall=0, precision=1) so the trapezoid rule starts cleanly
    recall = np.concatenate([[0.0], recall])
    precision = np.concatenate([[1.0], precision])
    return float(_trapezoid(precision, recall))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean absolute error."""
    y = _to_array(y_true)
    p = _to_array(y_pred)
    if y.size == 0:
        return float("nan")
    if y.shape != p.shape:
        raise ValueError(f"shape mismatch: y_true {y.shape} vs y_pred {p.shape}")
    return float(np.mean(np.abs(y - p)))


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Root mean squared error."""
    y = _to_array(y_true)
    p = _to_array(y_pred)
    if y.size == 0:
        return float("nan")
    if y.shape != p.shape:
        raise ValueError(f"shape mismatch: y_true {y.shape} vs y_pred {p.shape}")
    return float(np.sqrt(np.mean((y - p) ** 2)))


def precision_at_recall(
    y_true: np.ndarray,
    y_score: np.ndarray,
    target_recall: float = 0.9,
) -> float:
    """Highest precision attainable while keeping recall >= target_recall.

    Returns 0.0 if the target recall is unreachable, NaN if no positives.
    """
    y = _to_array(y_true)
    s = _to_array(y_score)
    if y.size == 0 or y.shape != s.shape:
        return float("nan")
    n_pos = int(np.sum(y == 1))
    if n_pos == 0:
        return float("nan")
    if not (0.0 < target_recall <= 1.0):
        raise ValueError("target_recall must be in (0, 1]")

    order = np.argsort(-s, kind="stable")
    y_sorted = y[order]
    tp_cum = np.cumsum(y_sorted == 1).astype(np.float64)
    fp_cum = np.cumsum(y_sorted == 0).astype(np.float64)
    precision = tp_cum / np.maximum(tp_cum + fp_cum, 1e-12)
    recall = tp_cum / float(n_pos)

    mask = recall >= target_recall
    if not mask.any():
        return 0.0
    return float(precision[mask].max())


def quantile_loss(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    q: float = 0.5,
) -> float:
    """Pinball loss at quantile q. Returns scalar mean over samples."""
    if not (0.0 < q < 1.0):
        raise ValueError("q must lie in (0, 1)")
    y = _to_array(y_true)
    p = _to_array(y_pred)
    if y.size == 0 or y.shape != p.shape:
        return float("nan")
    diff = y - p
    return float(np.mean(np.maximum(q * diff, (q - 1.0) * diff)))
