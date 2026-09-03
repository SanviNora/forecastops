"""Create the schema, users, SKU catalog, and 104 weeks of demand history."""

from __future__ import annotations

import sys

from sqlalchemy import delete, select

from app.auth import hash_password
from app.catalog import DCS, SKUS
from app.db import SessionLocal, engine
from app.generate import generate_demand_frame
from app.models import Base, DemandHistory, DistributionCenter, Sku, User

USERS = [
    ("scientist@forecastops.example.com", "Mina Chen", "scientist", "science-password"),
    ("ops@forecastops.example.com", "Luis Ortega", "ops", "ops-password"),
    ("admin@forecastops.example.com", "Site Admin", "admin", "admin-password"),
]


def main() -> int:
    Base.metadata.create_all(engine)
    frame = generate_demand_frame()

    with SessionLocal() as db:
        for email, name, role, password in USERS:
            if db.scalar(select(User).where(User.email == email)) is None:
                db.add(
                    User(
                        email=email,
                        full_name=name,
                        role=role,
                        hashed_password=hash_password(password),
                    )
                )

        db.execute(delete(DemandHistory))
        db.execute(delete(Sku))
        db.execute(delete(DistributionCenter))
        db.commit()

        for sku in SKUS:
            db.add(
                Sku(
                    sku_code=sku.sku_code,
                    name=sku.name,
                    family=sku.family,
                    base_demand=sku.base_demand,
                )
            )
        for dc in DCS:
            db.add(
                DistributionCenter(
                    dc_code=dc.dc_code,
                    name=dc.name,
                    demand_scale=dc.demand_scale,
                )
            )
        db.commit()

        records = [
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
        db.add_all(records)
        db.commit()

    print(f"loaded {len(frame)} demand rows across {len(SKUS)} skus and {len(DCS)} dcs")
    print("\nseeded users:")
    for email, _name, role, password in USERS:
        print(f"  {role:<10} {email:<42} {password}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
