from __future__ import annotations

from app.forecast.drift import family_psi
from app.generate import generate_demand_frame


def test_toric_post_week_80_psi_exceeds_shift_threshold():
    frame = generate_demand_frame()
    psi = family_psi(frame, "Toric", reference_end=79, recent_start=80)
    assert psi > 0.1, f"Toric campaign lift should move PSI above 0.1, got {psi:.4f}"


def test_daily_disposable_without_mean_shift_stays_below_drifted():
    frame = generate_demand_frame()
    psi = family_psi(frame, "Daily Disposable", reference_end=79, recent_start=80)
    assert psi < 0.25, f"Daily Disposable should not look drifted, got {psi:.4f}"
