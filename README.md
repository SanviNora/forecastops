# ForecastOps

[![CI](https://github.com/Akhilesh-Vangala/forecastops/actions/workflows/ci.yml/badge.svg)](https://github.com/Akhilesh-Vangala/forecastops/actions/workflows/ci.yml)

Weekly SKU demand forecasting for a 12-SKU, 3-DC lens network (104 weeks). Features are point-in-time (no future leakage). Models go through a Postgres registry. Drift is measured with PSI.

## Data

Fixed seed `42`:

- Per-SKU base × DC scale, plus yearly seasonality
- Daily Disposable promo weeks 20 and 72: +40% (`promo_flag` known in advance)
- From week 80, Toric mean demand is +25%

Table: `demand_history(sku_code, dc_code, week_start, units, promo_flag)`.

Lags and rolling means use only `units[s] < t`. `_assert_no_future_leak` checks that.

## Metrics and promotion

sMAPE is on a 0–200 scale. Holdout is the last 16 weeks (one-step). Rolling-origin backtest retrains every 4 weeks from origin 40 and scores an 8-week recursive horizon. Promotion uses **backtest sMAPE**, not the one-step holdout.

Stages: `none → staging → production`. A model can enter production only from staging, and only if its backtest sMAPE is strictly better than other staging candidates (or it is the first production model).

On this seed, GBM wins the 16-week one-step holdout and loses the 8-week recursive backtest, so seasonal naive is production.

| Split | Seasonal naive sMAPE | GBM sMAPE | Winner |
|---|---|---|---|
| 16-week one-step holdout | 10.77 | 5.77 | GBM |
| 8-week recursive backtest | 8.17 | 8.74 | naive (production) |

Toric PSI 1.26 (drifted) vs Daily Disposable 0.12 (shift). Thresholds: < 0.10 stable, 0.10–0.25 shift, > 0.25 drifted.

## Stack

- Python 3.13, FastAPI, SQLAlchemy 2, PostgreSQL, scikit-learn
- React 18, TypeScript, Vite
- Ports: UI 3008, API 8008, Postgres 5438

## Quick start

```bash
git clone https://github.com/Akhilesh-Vangala/forecastops.git
cd forecastops
cp .env.example .env
docker compose up --build
```

Open http://localhost:3008. Demo: `scientist@forecastops.example.com` / `science-password`.

```bash
python3.13 -m venv .venv
./.venv/bin/python -m pip install -r backend/requirements.txt
cd backend && ../.venv/bin/python -m pytest -q
```

Default JWT secret is `change-me-in-production`. Local use only.

## Tests

```bash
cd backend
pytest -q
```

17 tests covering leakage, backtest, registry gates, and PSI on the Toric shift.

## License

MIT. See [LICENSE](LICENSE).
