from __future__ import annotations

from dataclasses import dataclass

FAMILIES = ("Daily Disposable", "Monthly", "Toric", "IOL")

PROMO_WEEKS = (20, 72)
PROMO_FAMILY = "Daily Disposable"
PROMO_LIFT = 1.40

DRIFT_WEEK = 80
DRIFT_FAMILY = "Toric"
DRIFT_LIFT = 1.25

N_WEEKS = 104
HOLDOUT_WEEKS = 16
SEASONAL_PERIOD = 52
DEMAND_SEED = 42


@dataclass(frozen=True)
class SkuSpec:
    sku_code: str
    name: str
    family: str
    base_demand: float


@dataclass(frozen=True)
class DcSpec:
    dc_code: str
    name: str
    demand_scale: float


SKUS: tuple[SkuSpec, ...] = (
    SkuSpec("DD-CLARITY-1", "Clarity 1-Day", "Daily Disposable", 1400),
    SkuSpec("DD-AQUA-1", "Aqua 1-Day", "Daily Disposable", 1100),
    SkuSpec("DD-ACTIVE-1", "Active 1-Day", "Daily Disposable", 900),
    SkuSpec("MO-COMFORT-30", "Comfort Monthly", "Monthly", 620),
    SkuSpec("MO-OXY-30", "Oxy Monthly", "Monthly", 540),
    SkuSpec("MO-NIGHT-30", "Night Monthly", "Monthly", 410),
    SkuSpec("TO-ASTIG-1", "Astig 1 Toric", "Toric", 380),
    SkuSpec("TO-ASTIG-2", "Astig 2 Toric", "Toric", 310),
    SkuSpec("TO-ASTIG-3", "Astig 3 Toric", "Toric", 250),
    SkuSpec("IO-MONO-1", "Mono IOL", "IOL", 95),
    SkuSpec("IO-EDOF-1", "EDOF IOL", "IOL", 70),
    SkuSpec("IO-TRIFOCAL-1", "Trifocal IOL", "IOL", 55),
)

DCS: tuple[DcSpec, ...] = (
    DcSpec("DC-FW", "Fort Worth", 1.00),
    DcSpec("DC-SG", "Singapore", 0.82),
    DcSpec("DC-EU", "Cork", 0.91),
)

SKU_BY_CODE = {s.sku_code: s for s in SKUS}
FAMILY_BY_SKU = {s.sku_code: s.family for s in SKUS}
