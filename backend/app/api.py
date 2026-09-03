from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Query, status
from fastapi.security import OAuth2PasswordRequestForm
from sqlalchemy import select
from sqlalchemy.orm import Session

from .auth import create_access_token, get_current_user, require_role, verify_password
from .db import get_db
from .forecast.registry import RegistryError, list_versions, promote as promote_version
from .models import User
from .schemas import (
    BacktestPointOut,
    DemandRow,
    DriftResponse,
    FamilyDriftOut,
    ForecastRow,
    ModelVersionOut,
    PromoteRequest,
    Token,
    TrainRequest,
    TrainResponse,
    UserOut,
)
from .services import (
    demand_rows,
    drift_report,
    latest_training_run,
    production_forecast,
    train_and_register,
)

router = APIRouter(prefix="/api")

CurrentUser = Annotated[User, Depends(get_current_user)]
Db = Annotated[Session, Depends(get_db)]
Scientist = Annotated[User, Depends(require_role("scientist", "admin"))]


@router.post("/auth/token", response_model=Token, tags=["auth"])
def login(form: Annotated[OAuth2PasswordRequestForm, Depends()], db: Db) -> Token:
    user = db.scalar(select(User).where(User.email == form.username))
    if user is None or not verify_password(form.password, user.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="incorrect email or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return Token(
        access_token=create_access_token(user.email, user.role),
        role=user.role,
        full_name=user.full_name,
    )


@router.get("/auth/me", response_model=UserOut, tags=["auth"])
def me(user: CurrentUser) -> User:
    return user


@router.post("/train", response_model=TrainResponse, tags=["forecast"])
def train(payload: TrainRequest, db: Db, user: Scientist) -> TrainResponse:
    try:
        run = train_and_register(db, sku_code=payload.sku_code, created_by=user.email)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except RegistryError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    return TrainResponse(
        run_id=run.id,
        gbm_beats_baseline=run.gbm_beats_baseline,
        production_name=run.production_name,
        production_version=run.production_version,
        baseline=run.baseline_metrics_json,
        gbm=run.gbm_metrics_json,
    )


@router.get("/models", response_model=list[ModelVersionOut], tags=["registry"])
def models(db: Db, _user: CurrentUser) -> list[ModelVersionOut]:
    return list_versions(db)


@router.post("/models/{version_id}/promote", response_model=ModelVersionOut, tags=["registry"])
def promote_model(
    version_id: int,
    payload: PromoteRequest,
    db: Db,
    _user: Scientist,
) -> ModelVersionOut:
    try:
        return promote_version(db, version_id, payload.stage)
    except RegistryError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@router.get("/forecast", response_model=list[ForecastRow], tags=["forecast"])
def forecast(
    db: Db,
    _user: CurrentUser,
    horizon: int = Query(default=8, ge=1, le=26),
) -> list[ForecastRow]:
    try:
        frame = production_forecast(db, horizon=horizon)
    except RegistryError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    return [ForecastRow(**row) for row in frame.to_dict(orient="records")]


@router.get("/drift", response_model=DriftResponse, tags=["drift"])
def drift(db: Db, _user: CurrentUser) -> DriftResponse:
    payload = drift_report(db)
    return DriftResponse(
        reference_weeks=payload["reference_weeks"],
        recent_weeks=payload["recent_weeks"],
        families=[FamilyDriftOut(**row) for row in payload["families"]],
    )


@router.get("/backtest", response_model=list[BacktestPointOut], tags=["forecast"])
def backtest(db: Db, _user: CurrentUser) -> list[BacktestPointOut]:
    run = latest_training_run(db)
    if run is None:
        return []
    return [BacktestPointOut(**row) for row in (run.backtest_points_json or [])]


@router.get("/demand", response_model=list[DemandRow], tags=["demand"])
def demand(
    db: Db,
    _user: CurrentUser,
    sku: str | None = None,
    dc: str | None = None,
) -> list[DemandRow]:
    return [DemandRow(**row) for row in demand_rows(db, sku_code=sku, dc_code=dc)]
