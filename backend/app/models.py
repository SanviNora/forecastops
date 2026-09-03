from __future__ import annotations

from datetime import date, datetime, timezone

from sqlalchemy import (
    Boolean,
    CheckConstraint,
    Date,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    JSON,
    String,
    UniqueConstraint,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


class Base(DeclarativeBase):
    pass


class User(Base):
    __tablename__ = "users"

    id: Mapped[int] = mapped_column(primary_key=True)
    email: Mapped[str] = mapped_column(String(255), unique=True, index=True)
    hashed_password: Mapped[str] = mapped_column(String(255))
    full_name: Mapped[str] = mapped_column(String(255))
    role: Mapped[str] = mapped_column(String(32), default="scientist")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)

    __table_args__ = (
        CheckConstraint("role in ('scientist','ops','admin')", name="ck_user_role"),
    )


class Sku(Base):
    __tablename__ = "skus"

    sku_code: Mapped[str] = mapped_column(String(48), primary_key=True)
    name: Mapped[str] = mapped_column(String(160))
    family: Mapped[str] = mapped_column(String(48), index=True)
    base_demand: Mapped[float] = mapped_column(Float)


class DistributionCenter(Base):
    __tablename__ = "distribution_centers"

    dc_code: Mapped[str] = mapped_column(String(48), primary_key=True)
    name: Mapped[str] = mapped_column(String(160))
    demand_scale: Mapped[float] = mapped_column(Float, default=1.0)


class DemandHistory(Base):
    __tablename__ = "demand_history"
    __table_args__ = (
        UniqueConstraint("sku_code", "dc_code", "week_start", name="uq_demand_sku_dc_week"),
        CheckConstraint("units >= 0", name="ck_demand_units_non_negative"),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    sku_code: Mapped[str] = mapped_column(
        String(48), ForeignKey("skus.sku_code"), index=True
    )
    dc_code: Mapped[str] = mapped_column(
        String(48), ForeignKey("distribution_centers.dc_code"), index=True
    )
    week_start: Mapped[date] = mapped_column(Date, index=True)
    week_index: Mapped[int] = mapped_column(Integer, index=True)
    units: Mapped[float] = mapped_column(Float)
    promo_flag: Mapped[bool] = mapped_column(Boolean, default=False)

    sku: Mapped[Sku] = relationship()


class ModelVersion(Base):
    __tablename__ = "model_versions"
    __table_args__ = (
        UniqueConstraint("name", "version", name="uq_model_name_version"),
        CheckConstraint(
            "stage in ('none','staging','production','archived')",
            name="ck_model_stage",
        ),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(String(80), index=True)
    version: Mapped[int] = mapped_column(Integer)
    algorithm: Mapped[str] = mapped_column(String(64))
    metrics_json: Mapped[dict] = mapped_column(JSON, default=dict)
    artifact_path: Mapped[str] = mapped_column(String(500), default="")
    stage: Mapped[str] = mapped_column(String(32), default="none", index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class TrainingRun(Base):
    __tablename__ = "training_runs"

    id: Mapped[int] = mapped_column(primary_key=True)
    sku_filter: Mapped[str | None] = mapped_column(String(48), nullable=True)
    status: Mapped[str] = mapped_column(String(32), default="completed")
    baseline_metrics_json: Mapped[dict] = mapped_column(JSON, default=dict)
    gbm_metrics_json: Mapped[dict] = mapped_column(JSON, default=dict)
    gbm_beats_baseline: Mapped[bool] = mapped_column(Boolean, default=False)
    production_name: Mapped[str] = mapped_column(String(80), default="")
    production_version: Mapped[int | None] = mapped_column(Integer, nullable=True)
    backtest_points_json: Mapped[list] = mapped_column(JSON, default=list)
    residual_std: Mapped[float] = mapped_column(Float, default=0.0)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    created_by: Mapped[str] = mapped_column(String(255), default="")
