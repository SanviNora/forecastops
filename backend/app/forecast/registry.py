from __future__ import annotations

import json
from pathlib import Path

import joblib
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from ..config import get_settings
from ..models import ModelVersion

VALID_STAGES = ("none", "staging", "production", "archived")
PRODUCTION = "production"
STAGING = "staging"


class RegistryError(ValueError):
    pass


def _next_version(db: Session, name: str) -> int:
    current = db.scalar(select(func.max(ModelVersion.version)).where(ModelVersion.name == name))
    return 1 if current is None else int(current) + 1


def _smape(metrics: dict) -> float:
    backtest = metrics.get("backtest") or {}
    value = backtest.get("smape")
    if value is None:
        raise RegistryError("model metrics are missing backtest.smape")
    return float(value)


def register(
    db: Session,
    *,
    name: str,
    algorithm: str,
    metrics: dict,
    artifact,
    artifact_name: str | None = None,
) -> ModelVersion:
    settings = get_settings()
    directory = Path(settings.artifact_dir)
    directory.mkdir(parents=True, exist_ok=True)
    version = _next_version(db, name)
    filename = artifact_name or f"{name}-v{version}.joblib"
    path = directory / filename
    joblib.dump(artifact, path)

    row = ModelVersion(
        name=name,
        version=version,
        algorithm=algorithm,
        metrics_json=json.loads(json.dumps(metrics)),
        artifact_path=str(path),
        stage="none",
    )
    db.add(row)
    db.commit()
    db.refresh(row)
    return row


def promote(db: Session, version_id: int, to_stage: str) -> ModelVersion:
    if to_stage not in VALID_STAGES:
        raise RegistryError(f"unknown stage '{to_stage}'")

    row = db.get(ModelVersion, version_id)
    if row is None:
        raise RegistryError(f"model version {version_id} does not exist")

    if to_stage == PRODUCTION and row.stage != STAGING:
        raise RegistryError("cannot promote to production without passing through staging")

    if to_stage == PRODUCTION:
        competitors = list(
            db.scalars(
                select(ModelVersion).where(
                    ModelVersion.stage == STAGING,
                    ModelVersion.id != row.id,
                )
            )
        )
        if competitors:
            best_competitor = min(competitors, key=lambda m: _smape(m.metrics_json))
            if _smape(row.metrics_json) >= _smape(best_competitor.metrics_json):
                raise RegistryError(
                    "cannot promote to production unless backtest sMAPE is below the "
                    "staging competitor"
                )
        incumbents = list(
            db.scalars(select(ModelVersion).where(ModelVersion.stage == PRODUCTION))
        )
        for incumbent in incumbents:
            incumbent.stage = "archived"

    row.stage = to_stage
    db.commit()
    db.refresh(row)
    return row


def load_production(db: Session) -> ModelVersion:
    row = db.scalar(select(ModelVersion).where(ModelVersion.stage == PRODUCTION))
    if row is None:
        raise RegistryError("no production model is registered")
    return row


def load_artifact(row: ModelVersion):
    return joblib.load(row.artifact_path)


def list_versions(db: Session) -> list[ModelVersion]:
    return list(
        db.scalars(select(ModelVersion).order_by(ModelVersion.created_at.desc(), ModelVersion.id.desc()))
    )
