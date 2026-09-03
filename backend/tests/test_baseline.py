from __future__ import annotations

import math

from app.forecast.models import seasonal_naive_value
from app.generate import generate_demand_frame


def test_seasonal_naive_uses_week_minus_52_when_present():
    history = {week: float(week) for week in range(1, 80)}
    assert seasonal_naive_value(history, 70, origin=69) == history[18]


def test_seasonal_naive_falls_back_to_week_minus_4_when_seasonal_lag_is_unobserved():
    history = {week: float(week * 10) for week in range(1, 30)}
    # Week 20 has no t-52 history. Origin 19 still contains t-4 = week 16.
    assert seasonal_naive_value(history, 20, origin=19) == history[16]


def test_seasonal_naive_does_not_read_past_origin():
    history = {week: float(week) for week in range(1, 80)}
    # Origin 16 hides week 18 (the seasonal lag of week 70) but still contains week 16,
    # which is t-4 for a forecast at week 20. Check that week 18 is not used for week 70
    # when the origin is 16: last legal observation is week 16.
    assert seasonal_naive_value(history, 70, origin=16) == history[16]


def test_seasonal_naive_is_defined_on_generated_holdout():
    frame = generate_demand_frame()
    series = (
        frame[(frame["sku_code"] == "DD-CLARITY-1") & (frame["dc_code"] == "DC-FW")]
        .set_index("week_index")["units"]
        .to_dict()
    )
    for week in range(89, 105):
        yhat = seasonal_naive_value(series, week, origin=88)
        assert math.isfinite(yhat)
        assert yhat == series[week - 52]
