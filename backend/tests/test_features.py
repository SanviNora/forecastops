from __future__ import annotations

import pandas as pd

from app.forecast.features import _assert_no_future_leak, add_point_in_time_features


def _series() -> pd.DataFrame:
    units = [10, 12, 11, 15, 14, 13, 16, 18, 17, 19, 20, 22, 21, 24, 23, 25]
    rows = []
    for i, value in enumerate(units, start=1):
        rows.append(
            {
                "sku_code": "DD-CLARITY-1",
                "dc_code": "DC-FW",
                "week_index": i,
                "week_start": pd.Timestamp("2024-01-01") + pd.Timedelta(weeks=i - 1),
                "units": float(value),
                "promo_flag": False,
            }
        )
    return pd.DataFrame(rows)


def test_lag_1_equals_previous_week_units():
    featured = add_point_in_time_features(_series())
    row_t = featured[featured["week_index"] == 10].iloc[0]
    prev = featured[featured["week_index"] == 9].iloc[0]
    assert row_t["lag_1"] == prev["units"]


def test_rolling_mean_excludes_current_week():
    featured = add_point_in_time_features(_series())
    row = featured[featured["week_index"] == 8].iloc[0]
    prior = featured.loc[featured["week_index"].isin([4, 5, 6, 7]), "units"]
    current = float(featured.loc[featured["week_index"] == 8, "units"].iloc[0])
    assert abs(float(row["rolling_mean_4"]) - float(prior.mean())) < 1e-9
    leaked_mean = (float(prior.sum()) + current) / 5.0
    assert abs(float(row["rolling_mean_4"]) - leaked_mean) > 1e-6


def test_assert_no_future_leak_passes_on_point_in_time_features():
    featured = add_point_in_time_features(_series())
    _assert_no_future_leak(featured)


def test_assert_no_future_leak_catches_a_shifted_lag():
    featured = add_point_in_time_features(_series())
    featured.loc[featured["week_index"] == 10, "lag_1"] = featured.loc[
        featured["week_index"] == 11, "units"
    ].values[0]
    try:
        _assert_no_future_leak(featured)
    except AssertionError:
        return
    raise AssertionError("leakage in lag_1 was not detected")
