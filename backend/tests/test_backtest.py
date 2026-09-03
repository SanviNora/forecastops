from __future__ import annotations

import math

from app.forecast.models import metrics_are_finite, run_training, score_seasonal_naive
from app.generate import generate_demand_frame


def test_gbm_smape_is_finite_and_baseline_is_computed():
    frame = generate_demand_frame()
    baseline = score_seasonal_naive(frame)
    assert metrics_are_finite(baseline)
    assert baseline.n > 0

    result = run_training(frame)
    assert metrics_are_finite(result.baseline_holdout)
    assert metrics_are_finite(result.gbm_holdout)
    assert metrics_are_finite(result.baseline_backtest)
    assert metrics_are_finite(result.gbm_backtest)
    assert math.isfinite(result.residual_std)
    assert result.backtest_points

    comparison = {
        "baseline_backtest_smape": result.baseline_backtest.smape,
        "gbm_backtest_smape": result.gbm_backtest.smape,
        "gbm_beats_baseline": result.gbm_beats_baseline,
    }
    assert "gbm_beats_baseline" in comparison
