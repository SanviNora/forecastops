from __future__ import annotations

from datetime import date, datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, EmailStr, Field


class ORMModel(BaseModel):
    model_config = ConfigDict(from_attributes=True)


class Token(BaseModel):
    access_token: str
    token_type: str = "bearer"
    role: str
    full_name: str


class UserOut(ORMModel):
    id: int
    email: EmailStr
    full_name: str
    role: str


class TrainRequest(BaseModel):
    sku_code: str | None = None


class MetricSetOut(BaseModel):
    mae: float
    smape: float
    rmse: float
    n: int


class TrainResponse(BaseModel):
    run_id: int
    gbm_beats_baseline: bool
    production_name: str
    production_version: int | None
    baseline: dict[str, Any]
    gbm: dict[str, Any]


class PromoteRequest(BaseModel):
    stage: str = Field(pattern="^(none|staging|production|archived)$")


class ModelVersionOut(ORMModel):
    id: int
    name: str
    version: int
    algorithm: str
    metrics_json: dict[str, Any]
    artifact_path: str
    stage: str
    created_at: datetime


class ForecastRow(BaseModel):
    sku_code: str
    dc_code: str
    week_index: int
    week_start: str
    yhat: float
    yhat_lo: float
    yhat_hi: float


class DemandRow(BaseModel):
    sku_code: str
    dc_code: str
    family: str
    week_index: int
    week_start: date
    units: float
    promo_flag: bool


class FamilyDriftOut(BaseModel):
    family: str
    psi: float
    status: str
    trailing_mae: float | None
    backtest_mae: float | None
    performance_drift: bool
    note: str


class DriftResponse(BaseModel):
    reference_weeks: str
    recent_weeks: str
    families: list[FamilyDriftOut]


class BacktestPointOut(BaseModel):
    origin: int
    horizon: int
    week_index: int
    sku_code: str
    dc_code: str
    y: float
    yhat: float
    algorithm: str
