from __future__ import annotations

import pandas as pd

from app.forecast.features import _assert_no_future_leak, add_point_in_time_features
from app.generate import generate_demand_frame


def test_train_features_never_contain_future_units():
    frame = generate_demand_frame()
    featured = add_point_in_time_features(frame)
    _assert_no_future_leak(featured)

    for _, grp in featured.groupby(["sku_code", "dc_code"]):
        by_week = grp.set_index("week_index")
        units = by_week["units"]
        for week_index, row in by_week.iterrows():
            for col in ("lag_1", "lag_4", "lag_13", "rolling_mean_4", "rolling_mean_13"):
                value = row[col]
                if pd.isna(value):
                    continue
                if col.startswith("lag_"):
                    lag = int(col.split("_")[1])
                    source = week_index - lag
                    assert source < week_index
                    assert abs(float(value) - float(units.loc[source])) < 1e-8
                else:
                    window = 4 if col.endswith("_4") else 13
                    weeks = [week_index - k for k in range(1, window + 1)]
                    assert all(w < week_index for w in weeks)
                    expected = float(units.loc[weeks].mean())
                    assert abs(float(value) - expected) < 1e-8
