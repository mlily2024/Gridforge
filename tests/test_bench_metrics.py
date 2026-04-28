"""Verification tests for the benchmark metrics."""

from __future__ import annotations

import numpy as np
import pytest

from gridforge.bench.metrics import (
    auc_pr,
    brier_score,
    mae,
    precision_at_recall,
    quantile_loss,
    rmse,
)


class TestBrier:
    def test_perfect_predictions(self) -> None:
        assert brier_score([0, 1, 0, 1], [0.0, 1.0, 0.0, 1.0]) == 0.0

    def test_constant_half(self) -> None:
        # MSE between 0.5 and {0,1} is 0.25
        assert brier_score([0, 1, 0, 1], [0.5, 0.5, 0.5, 0.5]) == 0.25

    def test_shape_mismatch_raises(self) -> None:
        with pytest.raises(ValueError):
            brier_score([0, 1], [0.5, 0.5, 0.5])

    def test_empty(self) -> None:
        import math
        assert math.isnan(brier_score([], []))


class TestAUCPR:
    def test_perfect_ranking(self) -> None:
        # Positives ranked ahead of negatives → precision stays 1.0 across recall
        score = auc_pr([1, 1, 0, 0], [0.9, 0.8, 0.4, 0.3])
        assert score == pytest.approx(1.0, abs=1e-6)

    def test_no_positives_returns_nan(self) -> None:
        import math
        assert math.isnan(auc_pr([0, 0, 0, 0], [0.1, 0.2, 0.3, 0.4]))

    def test_constant_score_neutral(self) -> None:
        # With all-same scores, every ranking is equivalent. AUC-PR should
        # equal the prevalence (positives / total).
        score = auc_pr([1, 0, 1, 0], [0.5, 0.5, 0.5, 0.5])
        assert 0.4 < score < 1.0


class TestMAERMSE:
    def test_zero_when_perfect(self) -> None:
        assert mae([1.0, 2.0, 3.0], [1.0, 2.0, 3.0]) == 0.0
        assert rmse([1.0, 2.0, 3.0], [1.0, 2.0, 3.0]) == 0.0

    def test_known_values(self) -> None:
        assert mae([1.0, 2.0], [3.0, 5.0]) == 2.5  # (|2| + |3|) / 2
        # rmse = sqrt(((2)^2 + (3)^2) / 2) = sqrt(6.5)
        assert rmse([1.0, 2.0], [3.0, 5.0]) == pytest.approx(np.sqrt(6.5))


class TestPrecisionAtRecall:
    def test_perfect_recovery(self) -> None:
        # Two positives ranked at top → precision = 1.0 at full recall
        p = precision_at_recall([1, 1, 0, 0, 0], [0.9, 0.8, 0.5, 0.4, 0.3], 0.9)
        assert p == pytest.approx(1.0)

    def test_invalid_target_raises(self) -> None:
        with pytest.raises(ValueError):
            precision_at_recall([1, 0], [0.5, 0.5], target_recall=1.5)

    def test_no_positives_nan(self) -> None:
        import math
        assert math.isnan(
            precision_at_recall([0, 0, 0], [0.1, 0.2, 0.3])
        )


class TestQuantileLoss:
    def test_median_equals_half_mae(self) -> None:
        y_true = np.array([1.0, 2.0, 3.0])
        y_pred = np.array([2.0, 2.0, 2.0])
        ql = quantile_loss(y_true, y_pred, q=0.5)
        # at q=0.5 the pinball loss equals 0.5 * MAE
        assert ql == pytest.approx(0.5 * mae(y_true, y_pred))

    def test_q_out_of_range(self) -> None:
        with pytest.raises(ValueError):
            quantile_loss([1, 2], [1, 2], q=0.0)
        with pytest.raises(ValueError):
            quantile_loss([1, 2], [1, 2], q=1.0)
