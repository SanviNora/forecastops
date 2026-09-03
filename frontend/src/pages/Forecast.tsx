import { useEffect, useState } from "react";
import { ForecastRow, api } from "../api/client";

export default function Forecast() {
  const [rows, setRows] = useState<ForecastRow[]>([]);
  const [error, setError] = useState("");

  useEffect(() => {
    api
      .forecast(8)
      .then(setRows)
      .catch((err: Error) => setError(err.message));
  }, []);

  return (
    <div className="card">
      <h1>Production forecast</h1>
      <p className="lede">
        Next 8 weeks from the production model. Intervals are ±1.96 residual standard deviations measured on the
        16-week holdout.
      </p>
      {error && (
        <p className="error">
          {error}. Train a model first if the registry has no production version.
        </p>
      )}
      <table>
        <thead>
          <tr>
            <th>SKU</th>
            <th>DC</th>
            <th>Week</th>
            <th>Week start</th>
            <th>yhat</th>
            <th>yhat lo</th>
            <th>yhat hi</th>
          </tr>
        </thead>
        <tbody>
          {rows.map((row) => (
            <tr key={`${row.sku_code}-${row.dc_code}-${row.week_index}`}>
              <td>{row.sku_code}</td>
              <td>{row.dc_code}</td>
              <td className="metric">{row.week_index}</td>
              <td>{row.week_start}</td>
              <td className="metric">{row.yhat.toFixed(1)}</td>
              <td className="metric">{row.yhat_lo.toFixed(1)}</td>
              <td className="metric">{row.yhat_hi.toFixed(1)}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
