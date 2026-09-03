from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor

from ..catalog import HOLDOUT_WEEKS, SEASONAL_PERIOD
from .features import (
    FEATURE_COLUMNS,
    add_point_in_time_features,
    apply_category_codes,
    drop_incomplete_features,
    fit_category_codes,
    model_matrix,
)

GBM_PARAMS = {
    "n_estimators": 80,
    "max_depth": 3,
    "learning_rate": 0.1,
    "subsample": 0.9,
    "random_state": 42,
}

BACKTEST_ORIGIN_STEP = 4
BACKTEST_HORIZON = 8
BACKTEST_MIN_ORIGIN = 40


@dataclass
class MetricSet:
    mae: float
    smape: float
    rmse: float
    n: int

    def as_dict(self) -> dict[str, float | int]:
        return {
            "mae": float(self.mae),
            "smape": float(self.smape),
            "rmse": float(self.rmse),
            "n": int(self.n),
        }


@dataclass
class GbmBundle:
    model: GradientBoostingRegressor
    codes: dict[str, dict[str, int]]
    feature_columns: list[str]
    residual_std: float
    holdout: MetricSet
    backtest: MetricSet


@dataclass
class BacktestPoint:
    origin: int
    horizon: int
    week_index: int
    sku_code: str
    dc_code: str
    y: float
    yhat: float
    algorithm: str


@dataclass
class TrainResult:
    baseline_holdout: MetricSet
    gbm_holdout: MetricSet
    baseline_backtest: MetricSet
    gbm_backtest: MetricSet
    gbm_beats_baseline: bool
    residual_std: float
    bundle: GbmBundle
    backtest_points: list[BacktestPoint] = field(default_factory=list)


def mae(y: np.ndarray, yhat: np.ndarray) -> float:
    if len(y) == 0:
        return float("nan")
    return float(np.mean(np.abs(y - yhat)))


def rmse(y: np.ndarray, yhat: np.ndarray) -> float:
    if len(y) == 0:
        return float("nan")
    return float(np.sqrt(np.mean((y - yhat) ** 2)))


def smape(y: np.ndarray, yhat: np.ndarray) -> float:
    """Symmetric MAPE on a 0-200 scale.

    smape = (100/n) * sum 2|y - yhat| / (|y| + |yhat|). Rows where both sides are
    zero are skipped so a stock-out week does not divide by zero.
    """
    if len(y) == 0:
        return float("nan")
    denom = np.abs(y) + np.abs(yhat)
    mask = denom > 0
    if not np.any(mask):
        return 0.0
    return float(np.mean(2.0 * np.abs(y[mask] - yhat[mask]) / denom[mask]) * 100.0)


def summarize(y: np.ndarray, yhat: np.ndarray) -> MetricSet:
    return MetricSet(mae=mae(y, yhat), smape=smape(y, yhat), rmse=rmse(y, yhat), n=int(len(y)))


def metrics_are_finite(metrics: MetricSet) -> bool:
    return all(math.isfinite(v) for v in (metrics.mae, metrics.smape, metrics.rmse))


def seasonal_naive_value(history: dict[int, float], week_index: int, origin: int | None = None) -> float:
    """yhat[t] = y[t-52] if that week is known, otherwise y[t-4].

    `origin` is the last week that may be read. Holdout scoring passes origin=t-1 so a
    forecast never peeks into the week it is scoring.
    """
    last_allowed = week_index - 1 if origin is None else origin
    seasonal = week_index - SEASONAL_PERIOD
    if seasonal >= 1 and seasonal <= last_allowed and seasonal in history:
        return float(history[seasonal])
    short = week_index - 4
    if short >= 1 and short <= last_allowed and short in history:
        return float(history[short])
    known = [w for w in history if w <= last_allowed]
    if not known:
        raise ValueError(f"seasonal naive is undefined at week {week_index}")
    return float(history[max(known)])


def _series_map(frame: pd.DataFrame) -> dict[tuple[str, str], dict[int, float]]:
    out: dict[tuple[str, str], dict[int, float]] = {}
    for (sku, dc), grp in frame.groupby(["sku_code", "dc_code"]):
        out[(sku, dc)] = {int(w): float(u) for w, u in zip(grp["week_index"], grp["units"])}
    return out


def score_seasonal_naive(frame: pd.DataFrame, holdout_weeks: int = HOLDOUT_WEEKS) -> MetricSet:
    last_week = int(frame["week_index"].max())
    cutoff = last_week - holdout_weeks
    ys: list[float] = []
    yhats: list[float] = []
    history = _series_map(frame)
    holdout = frame[frame["week_index"] > cutoff]
    for row in holdout.itertuples(index=False):
        series = history[(row.sku_code, row.dc_code)]
        ys.append(float(row.units))
        yhats.append(seasonal_naive_value(series, int(row.week_index), origin=cutoff))
    return summarize(np.array(ys), np.array(yhats))


def _prepare(frame: pd.DataFrame) -> pd.DataFrame:
    featured = add_point_in_time_features(frame)
    return drop_incomplete_features(featured)


def train_gbm(
    frame: pd.DataFrame,
    holdout_weeks: int = HOLDOUT_WEEKS,
    n_estimators: int | None = None,
) -> GbmBundle:
    prepared = _prepare(frame)
    last_week = int(prepared["week_index"].max())
    cutoff = last_week - holdout_weeks
    train = prepared[prepared["week_index"] <= cutoff]
    test = prepared[prepared["week_index"] > cutoff]
    if train.empty or test.empty:
        raise ValueError("not enough history to train a GBM with the requested holdout")

    codes = fit_category_codes(train)
    train_x = model_matrix(apply_category_codes(train, codes))
    test_x = model_matrix(apply_category_codes(test, codes))
    params = dict(GBM_PARAMS)
    if n_estimators is not None:
        params["n_estimators"] = n_estimators
    model = GradientBoostingRegressor(**params)
    model.fit(train_x, train["units"].to_numpy())
    yhat = model.predict(test_x)
    y = test["units"].to_numpy()
    holdout = summarize(y, yhat)
    residual_std = float(np.std(y - yhat, ddof=1)) if len(y) > 1 else float(np.std(y - yhat))
    return GbmBundle(
        model=model,
        codes=codes,
        feature_columns=list(FEATURE_COLUMNS),
        residual_std=residual_std,
        holdout=holdout,
        backtest=MetricSet(mae=float("nan"), smape=float("nan"), rmse=float("nan"), n=0),
    )


def _history_through(frame: pd.DataFrame, origin: int) -> pd.DataFrame:
    return frame[frame["week_index"] <= origin].copy()


def _future_calendar(frame: pd.DataFrame, origin: int, horizon: int) -> pd.DataFrame:
    future = frame[
        (frame["week_index"] > origin) & (frame["week_index"] <= origin + horizon)
    ][["sku_code", "dc_code", "week_index", "week_start", "promo_flag", "units"]].copy()
    return future


def _recursive_gbm_forecast(
    history: pd.DataFrame,
    future: pd.DataFrame,
    model: GradientBoostingRegressor,
    codes: dict[str, dict[str, int]],
) -> list[tuple[int, str, str, float, float]]:
    """Walk the horizon using predicted units as subsequent lags.

    Actuals after `history` are not read. promo_flag and the calendar are known in
    advance, which is why they are taken from `future` rather than imputed.
    """
    working = history[["sku_code", "dc_code", "week_index", "week_start", "units", "promo_flag"]].copy()
    rows: list[tuple[int, str, str, float, float]] = []
    for week_index in sorted(future["week_index"].unique()):
        step = future[future["week_index"] == week_index].copy()
        step["units"] = np.nan
        combined = pd.concat([working, step], ignore_index=True)
        featured = add_point_in_time_features(combined)
        featured = apply_category_codes(featured, codes)
        current = featured[featured["week_index"] == week_index]
        current = current.dropna(subset=["lag_1", "lag_4", "lag_13", "rolling_mean_4", "rolling_mean_13"])
        if current.empty:
            continue
        yhat = model.predict(model_matrix(current))
        current = current.copy()
        current["yhat"] = yhat
        for row, pred in zip(current.itertuples(index=False), yhat):
            rows.append((week_index, row.sku_code, row.dc_code, float(row.units) if not pd.isna(row.units) else float("nan"), float(pred)))
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
    actuals = {(r.sku_code, r.dc_code, int(r.week_index)): float(r.units) for r in future.itertuples(index=False)}
    scored = []
    for week_index, sku, dc, _placeholder, pred in rows:
        y = actuals.get((sku, dc, week_index))
        if y is None:
            continue
        scored.append((week_index, sku, dc, y, pred))
    return scored


def rolling_origin_backtest(
    frame: pd.DataFrame,
    *,
    algorithm: str,
    origin_step: int = BACKTEST_ORIGIN_STEP,
    horizon: int = BACKTEST_HORIZON,
    min_origin: int = BACKTEST_MIN_ORIGIN,
    n_estimators: int = 40,
) -> tuple[MetricSet, list[BacktestPoint]]:
    last_week = int(frame["week_index"].max())
    origins = list(range(min_origin, last_week - horizon + 1, origin_step))
    if not origins:
        raise ValueError("no rolling-origin cutoffs fit in this series")

    points: list[BacktestPoint] = []
    history = _series_map(frame)

    for origin in origins:
        if algorithm == "seasonal_naive":
            future = _future_calendar(frame, origin, horizon)
            for row in future.itertuples(index=False):
                yhat = seasonal_naive_value(
                    history[(row.sku_code, row.dc_code)],
                    int(row.week_index),
                    origin=origin,
                )
                points.append(
                    BacktestPoint(
                        origin=origin,
                        horizon=int(row.week_index) - origin,
                        week_index=int(row.week_index),
                        sku_code=row.sku_code,
                        dc_code=row.dc_code,
                        y=float(row.units),
                        yhat=yhat,
                        algorithm=algorithm,
                    )
                )
            continue

        train_hist = _history_through(frame, origin)
        bundle = train_gbm_no_holdout(train_hist, n_estimators=n_estimators)
        future = _future_calendar(frame, origin, horizon)
        scored = _recursive_gbm_forecast(train_hist, future, bundle.model, bundle.codes)
        for week_index, sku, dc, y, yhat in scored:
            points.append(
                BacktestPoint(
                    origin=origin,
                    horizon=week_index - origin,
                    week_index=week_index,
                    sku_code=sku,
                    dc_code=dc,
                    y=y,
                    yhat=yhat,
                    algorithm=algorithm,
                )
            )

    y = np.array([p.y for p in points], dtype=float)
    yhat = np.array([p.yhat for p in points], dtype=float)
    return summarize(y, yhat), points


def train_gbm_no_holdout(frame: pd.DataFrame, n_estimators: int | None = None) -> GbmBundle:
    prepared = _prepare(frame)
    codes = fit_category_codes(prepared)
    x = model_matrix(apply_category_codes(prepared, codes))
    params = dict(GBM_PARAMS)
    if n_estimators is not None:
        params["n_estimators"] = n_estimators
    model = GradientBoostingRegressor(**params)
    y = prepared["units"].to_numpy()
    model.fit(x, y)
    yhat = model.predict(x)
    residual_std = float(np.std(y - yhat, ddof=1)) if len(y) > 1 else 0.0
    fitted = summarize(y, yhat)
    return GbmBundle(
        model=model,
        codes=codes,
        feature_columns=list(FEATURE_COLUMNS),
        residual_std=residual_std,
        holdout=fitted,
        backtest=MetricSet(mae=float("nan"), smape=float("nan"), rmse=float("nan"), n=0),
    )


def run_training(frame: pd.DataFrame) -> TrainResult:
    if frame["week_index"].nunique() < HOLDOUT_WEEKS + 20:
        raise ValueError("need at least holdout plus warmup weeks of demand history")

    baseline_holdout = score_seasonal_naive(frame)
    gbm_holdout_bundle = train_gbm(frame, holdout_weeks=HOLDOUT_WEEKS)

    baseline_backtest, naive_points = rolling_origin_backtest(frame, algorithm="seasonal_naive")
    gbm_backtest, gbm_points = rolling_origin_backtest(frame, algorithm="gbm")

    serve_bundle = train_gbm_no_holdout(frame)
    serve_bundle.holdout = gbm_holdout_bundle.holdout
    serve_bundle.backtest = gbm_backtest
    serve_bundle.residual_std = gbm_holdout_bundle.residual_std

    beats = (
        math.isfinite(gbm_backtest.smape)
        and math.isfinite(baseline_backtest.smape)
        and gbm_backtest.smape < baseline_backtest.smape
    )
    return TrainResult(
        baseline_holdout=baseline_holdout,
        gbm_holdout=gbm_holdout_bundle.holdout,
        baseline_backtest=baseline_backtest,
        gbm_backtest=gbm_backtest,
        gbm_beats_baseline=beats,
        residual_std=gbm_holdout_bundle.residual_std,
        bundle=serve_bundle,
        backtest_points=naive_points + gbm_points,
    )


def point_to_dict(point: BacktestPoint) -> dict[str, Any]:
    return {
        "origin": int(point.origin),
        "horizon": int(point.horizon),
        "week_index": int(point.week_index),
        "sku_code": str(point.sku_code),
        "dc_code": str(point.dc_code),
        "y": float(point.y),
        "yhat": float(point.yhat),
        "algorithm": str(point.algorithm),
    }
