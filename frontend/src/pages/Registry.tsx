import { useEffect, useState } from "react";
import { ModelVersion, api } from "../api/client";

function smape(row: ModelVersion) {
  const value = row.metrics_json.backtest?.smape;
  return value === undefined ? "—" : value.toFixed(3);
}

export default function Registry() {
  const [rows, setRows] = useState<ModelVersion[]>([]);
  const [error, setError] = useState("");
  const [busy, setBusy] = useState<number | null>(null);

  function reload() {
    api
      .models()
      .then(setRows)
      .catch((err: Error) => setError(err.message));
  }

  useEffect(() => {
    reload();
  }, []);

  async function promote(row: ModelVersion, stage: string) {
    setBusy(row.id);
    setError("");
    try {
      await api.promote(row.id, stage);
      reload();
    } catch (err) {
      setError(err instanceof Error ? err.message : "Promote failed");
    } finally {
      setBusy(null);
    }
  }

  return (
    <div className="card">
      <h1>Model registry</h1>
      <p className="lede">
        Local MLflow-style registry in Postgres. Production requires staging first, and a challenger may take
        production only if its backtest sMAPE is strictly below the other staging model.
      </p>
      {error && <p className="error">{error}</p>}
      <table>
        <thead>
          <tr>
            <th>Name</th>
            <th>Version</th>
            <th>Algorithm</th>
            <th>Backtest sMAPE</th>
            <th>Stage</th>
            <th></th>
          </tr>
        </thead>
        <tbody>
          {rows.map((row) => (
            <tr key={row.id}>
              <td>{row.name}</td>
              <td className="metric">{row.version}</td>
              <td>{row.algorithm}</td>
              <td className="metric">{smape(row)}</td>
              <td>
                <span className={`badge ${row.stage}`}>{row.stage}</span>
              </td>
              <td>
                <div className="row-actions">
                  {row.stage === "none" && (
                    <button className="secondary" disabled={busy === row.id} onClick={() => promote(row, "staging")}>
                      Promote to staging
                    </button>
                  )}
                  {row.stage === "staging" && (
                    <button disabled={busy === row.id} onClick={() => promote(row, "production")}>
                      Promote to production
                    </button>
                  )}
                </div>
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
