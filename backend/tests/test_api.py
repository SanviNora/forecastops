from __future__ import annotations


def test_health(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_train_list_models_forecast(client, auth_headers):
    trained = client.post("/api/train", json={}, headers=auth_headers)
    assert trained.status_code == 200, trained.text
    body = trained.json()
    assert "baseline" in body
    assert "gbm" in body
    assert "smape" in body["baseline"]["backtest"]
    assert "smape" in body["gbm"]["backtest"]

    listing = client.get("/api/models", headers=auth_headers)
    assert listing.status_code == 200
    models = listing.json()
    assert len(models) >= 2
    stages = {row["stage"] for row in models}
    assert "production" in stages

    forecast = client.get("/api/forecast?horizon=8", headers=auth_headers)
    assert forecast.status_code == 200
    rows = forecast.json()
    assert len(rows) > 0
    sample = rows[0]
    for key in ("sku_code", "dc_code", "week_start", "yhat", "yhat_lo", "yhat_hi"):
        assert key in sample
