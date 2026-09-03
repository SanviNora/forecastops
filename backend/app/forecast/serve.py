from __future__ import annotations

from datetime import timedelta

import numpy as np
import pandas as pd

from .features import (
    add_point_in_time_features,
    apply_category_codes,
    model_matrix,
)
from .models import GbmBundle, seasonal_naive_value


def _extend_calendar(history: pd.DataFrame, horizon: int) -> pd.DataFrame:
    last_week = int(history["week_index"].max())
    last_dates = (
        history.sort_values("week_index")
        .groupby(["sku_code", "dc_code"], as_index=False)
        .tail(1)[["sku_code", "dc_code", "week_start"]]
    )
    rows: list[dict] = []
    for step in range(1, horizon + 1):
        week_index = last_week + step
        for rec in last_dates.itertuples(index=False):
            week_start = pd.Timestamp(rec.week_start) + timedelta(weeks=step)
            rows.append(
                {
                    "sku_code": rec.sku_code,
                    "dc_code": rec.dc_code,
                    "week_index": week_index,
                    "week_start": week_start.date() if hasattr(week_start, "date") else week_start,
                    "units": np.nan,
                    "promo_flag": 0,
                }
            )
    return pd.DataFrame(rows)


def forecast_gbm(history: pd.DataFrame, bundle: GbmBundle, horizon: int = 8) -> pd.DataFrame:
    future = _extend_calendar(history, horizon)
    working = history[["sku_code", "dc_code", "week_index", "week_start", "units", "promo_flag"]].copy()
    records: list[dict] = []
    z = 1.96
    sigma = max(float(bundle.residual_std), 0.0)
    for week_index in sorted(future["week_index"].unique()):
        step = future[future["week_index"] == week_index].copy()
        combined = pd.concat([working, step], ignore_index=True)
        featured = apply_category_codes(add_point_in_time_features(combined), bundle.codes)
        current = featured[featured["week_index"] == week_index].dropna(
            subset=["lag_1", "lag_4", "lag_13", "rolling_mean_4", "rolling_mean_13"]
        )
        if current.empty:
            continue
        yhat = bundle.model.predict(model_matrix(current))
        for row, pred in zip(current.itertuples(index=False), yhat):
            lo = max(0.0, float(pred) - z * sigma)
            hi = float(pred) + z * sigma
            records.append(
                {
                    "sku_code": row.sku_code,
                    "dc_code": row.dc_code,
                    "week_index": int(row.week_index),
                    "week_start": pd.Timestamp(row.week_start).date().isoformat(),
                    "yhat": float(pred),
                    "yhat_lo": lo,
                    "yhat_hi": hi,
                }
            )
            working = pd.concat(
                [
                    working,
                    pd.DataFrame(
                        [
                            {
                                "sku_code": row.sku_code,
                                "dc_code": row.dc_code,
                                "week_index": week_index,
                                "week_start": row.week_start,
                                "units": float(pred),
                                "promo_flag": int(row.promo_flag),
                            }
                        ]
                    ),
                ],
                ignore_index=True,
            )
    return pd.DataFrame(records)


def forecast_seasonal_naive(history: pd.DataFrame, horizon: int = 8, residual_std: float = 0.0) -> pd.DataFrame:
    last_week = int(history["week_index"].max())
    future = _extend_calendar(history, horizon)
    series: dict[tuple[str, str], dict[int, float]] = {}
    for (sku, dc), grp in history.groupby(["sku_code", "dc_code"]):
        series[(sku, dc)] = {int(w): float(u) for w, u in zip(grp["week_index"], grp["units"])}
    z = 1.96
    records: list[dict] = []
    for row in future.itertuples(index=False):
        key = (row.sku_code, row.dc_code)
        yhat = seasonal_naive_value(series[key], int(row.week_index), origin=last_week)
        series[key][int(row.week_index)] = yhat
        records.append(
            {
                "sku_code": row.sku_code,
                "dc_code": row.dc_code,
                "week_index": int(row.week_index),
                "week_start": pd.Timestamp(row.week_start).date().isoformat(),
                "yhat": yhat,
                "yhat_lo": max(0.0, yhat - z * residual_std),
                "yhat_hi": yhat + z * residual_std,
            }
        )
    return pd.DataFrame(records)
