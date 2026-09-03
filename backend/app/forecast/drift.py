from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..catalog import DRIFT_WEEK, FAMILIES, FAMILY_BY_SKU

PSI_STABLE = 0.10
PSI_SHIFT = 0.25
PERFORMANCE_RATIO = 1.5
TRAILING_WEEKS = 8
N_BINS = 10
PSI_FLOOR = 1e-4


@dataclass
class FamilyDrift:
    family: str
    psi: float
    status: str
    trailing_mae: float | None
    backtest_mae: float | None
    performance_drift: bool
    note: str


def _bin_edges(reference: np.ndarray, n_bins: int = N_BINS) -> np.ndarray:
    quantiles = np.linspace(0.0, 1.0, n_bins + 1)
    edges = np.unique(np.quantile(reference, quantiles))
    if len(edges) < 3:
        lo = float(np.min(reference))
        hi = float(np.max(reference))
        if hi <= lo:
            hi = lo + 1.0
        edges = np.linspace(lo, hi, num=min(n_bins, 4) + 1)
    return edges


def population_stability_index(
    reference: np.ndarray,
    recent: np.ndarray,
    n_bins: int = N_BINS,
) -> float:
    reference = np.asarray(reference, dtype=float)
    recent = np.asarray(recent, dtype=float)
    if len(reference) == 0 or len(recent) == 0:
        return float("nan")
    edges = _bin_edges(reference, n_bins=n_bins)
    edges = edges.astype(float)
    edges[0] = min(edges[0], float(reference.min()), float(recent.min())) - 1e-9
    edges[-1] = max(edges[-1], float(reference.max()), float(recent.max())) + 1e-9
    expected_counts, _ = np.histogram(reference, bins=edges)
    actual_counts, _ = np.histogram(recent, bins=edges)
    expected = expected_counts.astype(float)
    actual = actual_counts.astype(float)
    expected = expected / expected.sum()
    actual = actual / actual.sum()
    expected = np.clip(expected, PSI_FLOOR, None)
    actual = np.clip(actual, PSI_FLOOR, None)
    return float(np.sum((actual - expected) * np.log(actual / expected)))


def psi_status(psi: float) -> str:
    if not np.isfinite(psi):
        return "unknown"
    if psi < PSI_STABLE:
        return "stable"
    if psi <= PSI_SHIFT:
        return "shift"
    return "drifted"


def family_psi(
    frame: pd.DataFrame,
    family: str,
    reference_end: int = DRIFT_WEEK - 1,
    recent_start: int | None = None,
) -> float:
    if recent_start is None:
        recent_start = reference_end + 1
    data = frame.copy()
    if "family" not in data.columns:
        data["family"] = data["sku_code"].map(FAMILY_BY_SKU)
    scoped = data[data["family"] == family]
    reference = scoped.loc[scoped["week_index"] <= reference_end, "units"].to_numpy()
    recent = scoped.loc[scoped["week_index"] >= recent_start, "units"].to_numpy()
    return population_stability_index(reference, recent)


def trailing_mae(residuals: np.ndarray) -> float:
    if len(residuals) == 0:
        return float("nan")
    return float(np.mean(np.abs(residuals)))


def performance_drift_flag(trailing: float, backtest_mae: float, ratio: float = PERFORMANCE_RATIO) -> bool:
    if not np.isfinite(trailing) or not np.isfinite(backtest_mae) or backtest_mae <= 0:
        return False
    return trailing > ratio * backtest_mae


def evaluate_drift(
    frame: pd.DataFrame,
    *,
    backtest_mae_by_family: dict[str, float] | None = None,
    residual_mae_by_family: dict[str, float] | None = None,
    reference_end: int = DRIFT_WEEK - 1,
) -> list[FamilyDrift]:
    backtest_mae_by_family = backtest_mae_by_family or {}
    residual_mae_by_family = residual_mae_by_family or {}
    reports: list[FamilyDrift] = []
    for family in FAMILIES:
        psi = family_psi(frame, family, reference_end=reference_end)
        status = psi_status(psi)
        trailing = residual_mae_by_family.get(family)
        backtest_mae = backtest_mae_by_family.get(family)
        perf = False
        if trailing is not None and backtest_mae is not None:
            perf = performance_drift_flag(trailing, backtest_mae)
        note = ""
        if family == "Toric" and psi > PSI_STABLE:
            note = (
                "Toric demand shifted at week 80. The generator applies a permanent +25% "
                "mean lift from that week, standing in for a new astigmatism campaign."
            )
        elif status == "stable":
            note = "Demand distribution is within the training-window bins."
        elif status == "shift":
            note = "Moderate distribution shift. Recheck promo calendar and mix before retraining."
        else:
            note = "Population has moved enough that the production model should be retrained."
        reports.append(
            FamilyDrift(
                family=family,
                psi=psi,
                status=status,
                trailing_mae=trailing,
                backtest_mae=backtest_mae,
                performance_drift=perf,
                note=note,
            )
        )
    return reports
