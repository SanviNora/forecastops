import os
import tempfile

os.environ["DATABASE_URL"] = "sqlite://"
os.environ["JWT_SECRET"] = "test-secret"
os.environ["ARTIFACT_DIR"] = tempfile.mkdtemp(prefix="forecastops-artifacts-")

import pytest
from fastapi.testclient import TestClient
from sqlalchemy.orm import sessionmaker

import app.db as db_module
from app.auth import hash_password
from app.catalog import DCS, SKUS
from app.db import engine, get_db
from app.generate import generate_demand_frame
from app.main import app
from app.models import Base, DemandHistory, DistributionCenter, Sku, User

db_module.SessionLocal = sessionmaker(bind=engine, autoflush=False, expire_on_commit=False)


def _seed(session) -> None:
    session.add(
        User(
            email="scientist@forecastops.example.com",
            hashed_password=hash_password("science-password"),
            full_name="Mina Chen",
            role="scientist",
        )
    )
    session.add(
        User(
            email="ops@forecastops.example.com",
            hashed_password=hash_password("ops-password"),
            full_name="Luis Ortega",
            role="ops",
        )
    )
    for sku in SKUS:
        session.add(
            Sku(
                sku_code=sku.sku_code,
                name=sku.name,
                family=sku.family,
                base_demand=sku.base_demand,
            )
        )
    for dc in DCS:
        session.add(
            DistributionCenter(
                dc_code=dc.dc_code,
                name=dc.name,
                demand_scale=dc.demand_scale,
            )
        )
    session.flush()
    frame = generate_demand_frame()
    session.add_all(
        [
            DemandHistory(
                sku_code=row.sku_code,
                dc_code=row.dc_code,
                week_start=row.week_start,
                week_index=int(row.week_index),
                units=float(row.units),
                promo_flag=bool(row.promo_flag),
            )
            for row in frame.itertuples(index=False)
        ]
    )
    session.commit()


@pytest.fixture()
def db_session():
    Base.metadata.create_all(bind=engine)
    session = db_module.SessionLocal()
    _seed(session)
    yield session
    session.close()
    Base.metadata.drop_all(bind=engine)


@pytest.fixture()
def client(db_session):
    def override_get_db():
        try:
            yield db_session
        finally:
            pass

    app.dependency_overrides[get_db] = override_get_db
    with TestClient(app) as test_client:
        yield test_client
    app.dependency_overrides.clear()


@pytest.fixture()
def auth_headers(client):
    response = client.post(
        "/api/auth/token",
        data={"username": "scientist@forecastops.example.com", "password": "science-password"},
    )
    assert response.status_code == 200
    token = response.json()["access_token"]
    return {"Authorization": f"Bearer {token}"}
