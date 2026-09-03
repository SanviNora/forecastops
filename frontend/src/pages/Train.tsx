import { useState } from "react";
import { TrainResponse, api } from "../api/client";

function fmt(value: number | undefined) {
  if (value === undefined || Number.isNaN(value)) return "—";
  return value.toFixed(3);
}

export default function Train() {
  const [result, setResult] = useState<TrainResponse | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");

  async function runTrain() {
    setBusy(true);
    setError("");
    try {
      const body = await api.train();
      setResult(body);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Training failed");
    } finally {
      setBusy(false);
    }
  }

  return (
    <div className="card">
      <h1>Train and compare</h1>
      <p className="lede">
        Fits a seasonal-naive baseline and a gradient boosting challenger on point-in-time lags. The GBM is promoted
        to staging automatically, then to production only if its rolling-origin sMAPE is lower.
      </p>
      <button onClick={runTrain} disabled={busy}>
        {busy ? "Training…" : "Train models"}
      </button>
      {error && <p className="error">{error}</p>}
      {result && (
        <>
          <p style={{ marginTop: "1rem" }}>
            {result.gbm_beats_baseline
              ? "GBM beat seasonal naive on rolling-origin sMAPE and is in production."
              : "GBM did not beat seasonal naive on rolling-origin sMAPE. Baseline is in production; the GBM remains in staging."}
          </p>
          <table style={{ marginTop: "1rem" }}>
            <thead>
              <tr>
                <th>Model</th>
                <th>Holdout MAE</th>
                <th>Holdout sMAPE</th>
                <th>Holdout RMSE</th>
                <th>Backtest MAE</th>
                <th>Backtest sMAPE</th>
                <th>Backtest RMSE</th>
              </tr>
            </thead>
            <tbody>
              <tr>
                <td>Seasonal naive</td>
                <td className="metric">{fmt(result.baseline.holdout.mae)}</td>
                <td className="metric">{fmt(result.baseline.holdout.smape)}</td>
                <td className="metric">{fmt(result.baseline.holdout.rmse)}</td>
                <td className="metric">{fmt(result.baseline.backtest.mae)}</td>
                <td className="metric">{fmt(result.baseline.backtest.smape)}</td>
                <td className="metric">{fmt(result.baseline.backtest.rmse)}</td>
              </tr>
              <tr>
                <td>GBM</td>
                <td className="metric">{fmt(result.gbm.holdout.mae)}</td>
                <td className="metric">{fmt(result.gbm.holdout.smape)}</td>
                <td className="metric">{fmt(result.gbm.holdout.rmse)}</td>
                <td className="metric">{fmt(result.gbm.backtest.mae)}</td>
                <td className="metric">{fmt(result.gbm.backtest.smape)}</td>
                <td className="metric">{fmt(result.gbm.backtest.rmse)}</td>
              </tr>
            </tbody>
          </table>
        </>
      )}
    </div>
  );
}
