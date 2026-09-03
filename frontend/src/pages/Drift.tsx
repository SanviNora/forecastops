import { useEffect, useState } from "react";
import { DriftResponse, api } from "../api/client";

export default function Drift() {
  const [report, setReport] = useState<DriftResponse | null>(null);
  const [error, setError] = useState("");

  useEffect(() => {
    api
      .drift()
      .then(setReport)
      .catch((err: Error) => setError(err.message));
  }, []);

  return (
    <div>
      <div className="card">
        <h1>Drift dashboard</h1>
        <p className="lede">
          Population Stability Index compares weekly unit distributions in weeks {report?.reference_weeks ?? "1-79"}{" "}
          against {report?.recent_weeks ?? "80-104"}. PSI below 0.10 is stable, 0.10–0.25 is a shift, and above 0.25
          is drifted. Trailing 8-week MAE above 1.5× backtest MAE is flagged separately as performance drift.
        </p>
        {error && <p className="error">{error}</p>}
      </div>
      <div className="grid-3">
        {report?.families.map((family) => (
          <div key={family.family} className={`card psi-card ${family.status}`}>
            <h2>{family.family}</h2>
            <div className="metric-value metric">{family.psi.toFixed(3)}</div>
            <p>
              <span className={`badge ${family.status}`}>{family.status}</span>
              {family.performance_drift && (
                <span className="badge drifted" style={{ marginLeft: "0.4rem" }}>
                  performance
                </span>
              )}
            </p>
            <p className="lede">{family.note}</p>
            {family.family === "Toric" && (
              <p className="lede">
                The seeded astigmatism campaign lifts Toric mean demand 25% from week 80 onward. That location shift
                is what this PSI is designed to catch.
              </p>
            )}
          </div>
        ))}
      </div>
    </div>
  );
}
