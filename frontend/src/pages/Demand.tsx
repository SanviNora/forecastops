import { useEffect, useMemo, useState } from "react";
import { DemandRow, api } from "../api/client";

function Sparkline({ values }: { values: number[] }) {
  if (values.length < 2) return null;
  const width = 140;
  const height = 32;
  const min = Math.min(...values);
  const max = Math.max(...values);
  const span = max - min || 1;
  const points = values
    .map((v, i) => {
      const x = (i / (values.length - 1)) * width;
      const y = height - ((v - min) / span) * (height - 2) - 1;
      return `${x.toFixed(1)},${y.toFixed(1)}`;
    })
    .join(" ");
  return (
    <svg className="spark" width={width} height={height} viewBox={`0 0 ${width} ${height}`} aria-hidden="true">
      <polyline fill="none" stroke="#3db8a0" strokeWidth="1.6" points={points} />
    </svg>
  );
}

export default function Demand() {
  const [rows, setRows] = useState<DemandRow[]>([]);
  const [sku, setSku] = useState("");
  const [dc, setDc] = useState("");
  const [error, setError] = useState("");

  useEffect(() => {
    api
      .demand()
      .then(setRows)
      .catch((err: Error) => setError(err.message));
  }, []);

  const skus = useMemo(() => Array.from(new Set(rows.map((r) => r.sku_code))).sort(), [rows]);
  const dcs = useMemo(() => Array.from(new Set(rows.map((r) => r.dc_code))).sort(), [rows]);

  const filtered = useMemo(
    () => rows.filter((r) => (!sku || r.sku_code === sku) && (!dc || r.dc_code === dc)),
    [rows, sku, dc],
  );

  const series = useMemo(() => {
    const grouped = new Map<string, DemandRow[]>();
    for (const row of filtered) {
      const key = `${row.sku_code}|${row.dc_code}`;
      const list = grouped.get(key) || [];
      list.push(row);
      grouped.set(key, list);
    }
    return Array.from(grouped.entries()).map(([, list]) => {
      const sorted = [...list].sort((a, b) => a.week_index - b.week_index);
      const latest = sorted[sorted.length - 1];
      return {
        key: `${latest.sku_code}|${latest.dc_code}`,
        sku_code: latest.sku_code,
        dc_code: latest.dc_code,
        family: latest.family,
        latest: latest.units,
        promo: sorted.filter((r) => r.promo_flag).length,
        values: sorted.map((r) => r.units),
      };
    });
  }, [filtered]);

  return (
    <div className="card">
      <h1>Demand explorer</h1>
      <p className="lede">
        Weekly units for 12 contact-lens SKUs across three DCs. Daily Disposable promos fire in weeks 20 and 72.
        Toric demand steps up 25% from week 80.
      </p>
      {error && <p className="error">{error}</p>}
      <div className="toolbar">
        <div className="field">
          <label htmlFor="sku">SKU</label>
          <select id="sku" value={sku} onChange={(e) => setSku(e.target.value)}>
            <option value="">All SKUs</option>
            {skus.map((code) => (
              <option key={code} value={code}>
                {code}
              </option>
            ))}
          </select>
        </div>
        <div className="field">
          <label htmlFor="dc">DC</label>
          <select id="dc" value={dc} onChange={(e) => setDc(e.target.value)}>
            <option value="">All DCs</option>
            {dcs.map((code) => (
              <option key={code} value={code}>
                {code}
              </option>
            ))}
          </select>
        </div>
      </div>
      <table>
        <thead>
          <tr>
            <th>SKU</th>
            <th>Family</th>
            <th>DC</th>
            <th>Latest week</th>
            <th>Promo weeks</th>
            <th>104-week path</th>
          </tr>
        </thead>
        <tbody>
          {series.map((row) => (
            <tr key={row.key}>
              <td>{row.sku_code}</td>
              <td>{row.family}</td>
              <td>{row.dc_code}</td>
              <td className="metric">{row.latest.toFixed(0)}</td>
              <td>{row.promo}</td>
              <td>
                <Sparkline values={row.values} />
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
