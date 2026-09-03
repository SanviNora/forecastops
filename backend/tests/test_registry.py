from __future__ import annotations

import pytest

from app.forecast.registry import RegistryError, load_production, promote, register


def _artifact() -> dict:
    return {"algorithm": "seasonal_naive", "residual_std": 1.0}


def test_cannot_promote_to_production_skipping_staging(db_session):
    row = register(
        db_session,
        name="demand-seasonal-naive",
        algorithm="seasonal_naive",
        metrics={"backtest": {"smape": 12.0}},
        artifact=_artifact(),
        artifact_name="naive-skip.joblib",
    )
    with pytest.raises(RegistryError, match="staging"):
        promote(db_session, row.id, "production")


def test_first_model_can_go_staging_then_production(db_session):
    row = register(
        db_session,
        name="demand-seasonal-naive",
        algorithm="seasonal_naive",
        metrics={"backtest": {"smape": 12.0}},
        artifact=_artifact(),
        artifact_name="naive-first.joblib",
    )
    staged = promote(db_session, row.id, "staging")
    assert staged.stage == "staging"
    production = promote(db_session, row.id, "production")
    assert production.stage == "production"
    loaded = load_production(db_session)
    assert loaded.id == row.id


def test_worse_staging_model_cannot_take_production(db_session):
    better = register(
        db_session,
        name="demand-gbm",
        algorithm="gbm",
        metrics={"backtest": {"smape": 8.0}},
        artifact=_artifact(),
        artifact_name="gbm-better.joblib",
    )
    worse = register(
        db_session,
        name="demand-seasonal-naive",
        algorithm="seasonal_naive",
        metrics={"backtest": {"smape": 18.0}},
        artifact=_artifact(),
        artifact_name="naive-worse.joblib",
    )
    promote(db_session, better.id, "staging")
    promote(db_session, worse.id, "staging")
    with pytest.raises(RegistryError, match="staging competitor"):
        promote(db_session, worse.id, "production")
    promote(db_session, better.id, "production")
    assert load_production(db_session).id == better.id
