from __future__ import annotations

import pandas as pd

LAGS = (1, 4, 13)
ROLLING_WINDOWS = (4, 13)
FEATURE_COLUMNS = [
    "lag_1",
    "lag_4",
    "lag_13",
    "rolling_mean_4",
    "rolling_mean_13",
    "week_of_year",
    "sku_id",
    "dc_id",
    "promo_flag",
]


def add_point_in_time_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Build lags and rolling means that a forecaster could have known at week t.

    Rolling windows are computed on units.shift(1) so the current week's demand is
    never inside the average that is used to predict that same week.
    """
    required = {"sku_code", "dc_code", "week_index", "week_start", "units", "promo_flag"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"demand frame is missing columns: {sorted(missing)}")

    out = frame.copy()
    out["week_start"] = pd.to_datetime(out["week_start"])
    out = out.sort_values(["sku_code", "dc_code", "week_index"]).reset_index(drop=True)

    parts: list[pd.DataFrame] = []
    for _, grp in out.groupby(["sku_code", "dc_code"], sort=False):
        g = grp.copy()
        g["lag_1"] = g["units"].shift(1)
        g["lag_4"] = g["units"].shift(4)
        g["lag_13"] = g["units"].shift(13)
        shifted = g["units"].shift(1)
        g["rolling_mean_4"] = shifted.rolling(window=4, min_periods=4).mean()
        g["rolling_mean_13"] = shifted.rolling(window=13, min_periods=13).mean()
        parts.append(g)

    featured = pd.concat(parts, ignore_index=True)
    featured["week_of_year"] = featured["week_start"].dt.isocalendar().week.astype(int)
    featured["promo_flag"] = featured["promo_flag"].astype(int)
    return featured


def fit_category_codes(train: pd.DataFrame) -> dict[str, dict[str, int]]:
    sku_map = {code: i for i, code in enumerate(sorted(train["sku_code"].unique()))}
    dc_map = {code: i for i, code in enumerate(sorted(train["dc_code"].unique()))}
    return {"sku_code": sku_map, "dc_code": dc_map}


def apply_category_codes(frame: pd.DataFrame, codes: dict[str, dict[str, int]]) -> pd.DataFrame:
    out = frame.copy()
    out["sku_id"] = out["sku_code"].map(codes["sku_code"]).fillna(-1).astype(int)
    out["dc_id"] = out["dc_code"].map(codes["dc_code"]).fillna(-1).astype(int)
    return out


def model_matrix(frame: pd.DataFrame) -> pd.DataFrame:
    missing = [c for c in FEATURE_COLUMNS if c not in frame.columns]
    if missing:
        raise ValueError(f"featured frame is missing columns: {missing}")
    return frame[FEATURE_COLUMNS].astype(float)


def drop_incomplete_features(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.dropna(subset=["lag_1", "lag_4", "lag_13", "rolling_mean_4", "rolling_mean_13"])


def _assert_no_future_leak(frame: pd.DataFrame) -> None:
    needed = {"sku_code", "dc_code", "week_index", "units", "lag_1", "lag_4", "lag_13"}
    missing = needed - set(frame.columns)
    if missing:
        raise AssertionError(f"cannot check leakage without columns {sorted(missing)}")

    ordered = frame.sort_values(["sku_code", "dc_code", "week_index"])
    for (_, _), grp in ordered.groupby(["sku_code", "dc_code"]):
        by_week = grp.set_index("week_index")
        units = by_week["units"]
        for week_index, row in by_week.iterrows():
            for lag, col in ((1, "lag_1"), (4, "lag_4"), (13, "lag_13")):
                value = row[col]
                if pd.isna(value):
                    continue
                source_week = week_index - lag
                if source_week not in units.index:
                    raise AssertionError(
                        f"{col} present at week {week_index} but week {source_week} is missing"
                    )
                expected = float(units.loc[source_week])
                if abs(float(value) - expected) > 1e-8:
                    raise AssertionError(
                        f"{col} at week {week_index} is {value}, expected units[{source_week}]={expected}"
                    )
            if "rolling_mean_4" in by_week.columns and not pd.isna(row.get("rolling_mean_4")):
                window = [week_index - k for k in range(1, 5)]
                if all(w in units.index for w in window):
                    expected_mean = float(units.loc[window].mean())
                    if abs(float(row["rolling_mean_4"]) - expected_mean) > 1e-8:
                        raise AssertionError(
                            f"rolling_mean_4 at week {week_index} includes a value other than units[t-1..t-4]"
                        )
            if "rolling_mean_13" in by_week.columns and not pd.isna(row.get("rolling_mean_13")):
                window = [week_index - k for k in range(1, 14)]
                if all(w in units.index for w in window):
                    expected_mean = float(units.loc[window].mean())
                    if abs(float(row["rolling_mean_13"]) - expected_mean) > 1e-8:
                        raise AssertionError(
                            f"rolling_mean_13 at week {week_index} includes a value other than units[t-1..t-13]"
                        )
