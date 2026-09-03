from __future__ import annotations

from datetime import date

import pandas as pd
from sqlalchemy import select
from sqlalchemy.orm import Session

from .catalog import DRIFT_WEEK, FAMILY_BY_SKU, N_WEEKS
from .forecast.drift import TRAILING_WEEKS, evaluate_drift
from .forecast.features import (
    add_point_in_time_features,
    apply_category_codes,
    drop_incomplete_features,
    model_matrix,
)
from .forecast.models import (
    GbmBundle,
    MetricSet,
    point_to_dict,
    run_training,
    seasonal_naive_value,
)
from .forecast.registry import (
    RegistryError,
    load_artifact,
    load_production,
    promote,
    register,
)
from .forecast.serve import forecast_gbm, forecast_seasonal_naive
from .generate import attach_family
from .models import DemandHistory, Sku, TrainingRun


def demand_frame(db: Session, sku_code: str | None = None) -> pd.DataFrame:
    stmt = select(DemandHistory, Sku.family).join(Sku, Sku.sku_code == DemandHistory.sku_code)
    if sku_code:
        stmt = stmt.where(DemandHistory.sku_code == sku_code)
    rows = db.execute(stmt).all()
    if not rows:
        raise ValueError("no demand history is loaded")
    records = []
    for history, family in rows:
        records.append(
            {
                "sku_code": history.sku_code,
                "dc_code": history.dc_code,
                "family": family,
                "week_index": history.week_index,
                "week_start": history.week_start,
                "units": history.units,
                "promo_flag": history.promo_flag,
            }
        )
    return pd.DataFrame(records)


def _metrics_payload(result: TrainResult, kind: str) -> dict:
    if kind == "baseline":
        holdout = result.baseline_holdout
        backtest = result.baseline_backtest
    else:
        holdout = result.gbm_holdout
        backtest = result.gbm_backtest
    return {
        "holdout": holdout.as_dict(),
        "backtest": backtest.as_dict(),
    }


def train_and_register(db: Session, *, sku_code: str | None, created_by: str) -> TrainingRun:
    frame = demand_frame(db, sku_code=sku_code)
    result = run_training(frame)

    baseline_metrics = _metrics_payload(result, "baseline")
    gbm_metrics = _metrics_payload(result, "gbm")

    naive_artifact = {
        "algorithm": "seasonal_naive",
        "residual_std": result.residual_std,
    }
    gbm_artifact = {
        "algorithm": "gbm",
        "model": result.bundle.model,
        "codes": result.bundle.codes,
        "feature_columns": result.bundle.feature_columns,
        "residual_std": result.residual_std,
    }

    baseline_row = register(
        db,
        name="demand-seasonal-naive",
        algorithm="seasonal_naive",
        metrics=baseline_metrics,
        artifact=naive_artifact,
    )
    gbm_row = register(
        db,
        name="demand-gbm",
        algorithm="gbm",
        metrics=gbm_metrics,
        artifact=gbm_artifact,
    )

    promote(db, gbm_row.id, "staging")
    production_name = gbm_row.name
    production_version = gbm_row.version
    if result.gbm_beats_baseline:
        promote(db, gbm_row.id, "production")
    else:
        promote(db, baseline_row.id, "staging")
        promote(db, baseline_row.id, "production")
        production_name = baseline_row.name
        production_version = baseline_row.version

    run = TrainingRun(
        sku_filter=sku_code,
        status="completed",
        baseline_metrics_json=baseline_metrics,
        gbm_metrics_json=gbm_metrics,
        gbm_beats_baseline=result.gbm_beats_baseline,
        production_name=production_name,
        production_version=production_version,
        backtest_points_json=[point_to_dict(p) for p in result.backtest_points],
        residual_std=result.residual_std,
        created_by=created_by,
    )
    db.add(run)
    db.commit()
    db.refresh(run)
    return run


def latest_training_run(db: Session) -> TrainingRun | None:
    return db.scalar(select(TrainingRun).order_by(TrainingRun.created_at.desc(), TrainingRun.id.desc()))


def production_forecast(db: Session, horizon: int = 8) -> pd.DataFrame:
    row = load_production(db)
    artifact = load_artifact(row)
    history = demand_frame(db)
    dummy = MetricSet(mae=0.0, smape=0.0, rmse=0.0, n=0)
    if artifact.get("algorithm") == "gbm":
        bundle = GbmBundle(
            model=artifact["model"],
            codes=artifact["codes"],
            feature_columns=artifact["feature_columns"],
            residual_std=float(artifact.get("residual_std") or 0.0),
            holdout=dummy,
            backtest=dummy,
        )
        return forecast_gbm(history, bundle, horizon=horizon)
    return forecast_seasonal_naive(
        history,
        horizon=horizon,
        residual_std=float(artifact.get("residual_std") or 0.0),
    )


def _one_step_residuals(frame: pd.DataFrame, artifact: dict) -> pd.DataFrame:
    last_week = int(frame["week_index"].max())
    trailing_start = last_week - TRAILING_WEEKS + 1
    window = frame[frame["week_index"] >= trailing_start].copy()
    if artifact.get("algorithm") != "gbm":
        history = {
            (sku, dc): {
                int(w): float(u)
                for w, u in zip(grp["week_index"], grp["units"])
            }
            for (sku, dc), grp in frame.groupby(["sku_code", "dc_code"])
        }
        rows = []
        for rec in window.itertuples(index=False):
            yhat = seasonal_naive_value(
                history[(rec.sku_code, rec.dc_code)],
                int(rec.week_index),
                origin=int(rec.week_index) - 1,
            )
            rows.append(
                {
                    "sku_code": rec.sku_code,
                    "family": rec.family if hasattr(rec, "family") else FAMILY_BY_SKU[rec.sku_code],
                    "residual": float(rec.units) - yhat,
                }
            )
        return pd.DataFrame(rows)

    featured = drop_incomplete_features(add_point_in_time_features(frame))
    featured = apply_category_codes(featured, artifact["codes"])
    scoped = featured[featured["week_index"] >= trailing_start]
    if scoped.empty:
        return pd.DataFrame(columns=["sku_code", "family", "residual"])
    yhat = artifact["model"].predict(model_matrix(scoped))
    out = scoped[["sku_code", "week_index"]].copy()
    out["family"] = out["sku_code"].map(FAMILY_BY_SKU)
    out["residual"] = scoped["units"].to_numpy() - yhat
    return out


def drift_report(db: Session) -> dict:
    frame = attach_family(demand_frame(db))
    backtest_mae: dict[str, float] = {}
    trailing_mae: dict[str, float] = {}
    run = latest_training_run(db)
    if run is not None:
        points = pd.DataFrame(run.backtest_points_json or [])
        if not points.empty:
            points["family"] = points["sku_code"].map(FAMILY_BY_SKU)
            prod_name = run.production_name
            algo = "gbm" if "gbm" in prod_name else "seasonal_naive"
            scoped = points[points["algorithm"] == algo] if "algorithm" in points.columns else points
            if not scoped.empty:
                scoped = scoped.copy()
                scoped["abs_err"] = (scoped["y"] - scoped["yhat"]).abs()
                backtest_mae = scoped.groupby("family")["abs_err"].mean().to_dict()
        try:
            row = load_production(db)
            artifact = load_artifact(row)
            residuals = _one_step_residuals(frame, artifact)
            if not residuals.empty:
                residuals["abs_err"] = residuals["residual"].abs()
                trailing_mae = residuals.groupby("family")["abs_err"].mean().to_dict()
        except RegistryError:
            pass

    reports = evaluate_drift(
        frame,
        backtest_mae_by_family=backtest_mae,
        residual_mae_by_family=trailing_mae,
        reference_end=DRIFT_WEEK - 1,
    )
    return {
        "reference_weeks": f"1-{DRIFT_WEEK - 1}",
        "recent_weeks": f"{DRIFT_WEEK}-{N_WEEKS}",
        "families": [r.__dict__ for r in reports],
    }


def demand_rows(
    db: Session,
    sku_code: str | None = None,
    dc_code: str | None = None,
) -> list[dict]:
    frame = attach_family(demand_frame(db, sku_code=sku_code))
    if dc_code:
        frame = frame[frame["dc_code"] == dc_code]
    frame = frame.sort_values(["sku_code", "dc_code", "week_index"])
    out = []
    for rec in frame.itertuples(index=False):
        week_start = rec.week_start
        if isinstance(week_start, date):
            week_value = week_start
        else:
            week_value = pd.Timestamp(week_start).date()
        out.append(
            {
                "sku_code": rec.sku_code,
                "dc_code": rec.dc_code,
                "family": rec.family,
                "week_index": int(rec.week_index),
                "week_start": week_value,
                "units": float(rec.units),
                "promo_flag": bool(rec.promo_flag),
            }
        )
    return out
