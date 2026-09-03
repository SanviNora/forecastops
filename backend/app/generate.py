from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import pandas as pd

from .catalog import (
    DCS,
    DEMAND_SEED,
    DRIFT_FAMILY,
    DRIFT_LIFT,
    DRIFT_WEEK,
    N_WEEKS,
    PROMO_FAMILY,
    PROMO_LIFT,
    PROMO_WEEKS,
    SKUS,
    FAMILY_BY_SKU,
)

# ISO week 1 of 2024 starts on Monday 1 January, so week_index maps onto calendar weeks
# without an off-by-one when we later read week_of_year from week_start.
ANCHOR_MONDAY = date(2024, 1, 1)


def week_start_for(week_index: int) -> date:
    return ANCHOR_MONDAY + timedelta(weeks=week_index - 1)


def generate_demand_frame(seed: int = DEMAND_SEED, n_weeks: int = N_WEEKS) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows: list[dict] = []
    for sku in SKUS:
        for dc in DCS:
            noise = rng.normal(0.0, 0.06, size=n_weeks)
            for week_index in range(1, n_weeks + 1):
                # 5% seasonality; larger amplitude broke PSI on unshifted families.
                seasonal = 1.0 + 0.05 * np.sin(2.0 * np.pi * week_index / 52.0)
                promo = sku.family == PROMO_FAMILY and week_index in PROMO_WEEKS
                promo_mult = PROMO_LIFT if promo else 1.0
                drift_mult = (
                    DRIFT_LIFT
                    if sku.family == DRIFT_FAMILY and week_index >= DRIFT_WEEK
                    else 1.0
                )
                units = (
                    sku.base_demand
                    * dc.demand_scale
                    * seasonal
                    * promo_mult
                    * drift_mult
                    * (1.0 + noise[week_index - 1])
                )
                rows.append(
                    {
                        "sku_code": sku.sku_code,
                        "dc_code": dc.dc_code,
                        "family": sku.family,
                        "week_index": week_index,
                        "week_start": week_start_for(week_index),
                        "units": max(0.0, float(units)),
                        "promo_flag": bool(promo),
                    }
                )
    return pd.DataFrame(rows)


def attach_family(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    if "family" not in out.columns:
        out["family"] = out["sku_code"].map(FAMILY_BY_SKU)
    return out
