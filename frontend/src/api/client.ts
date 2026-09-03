const API_BASE = "/api";

export interface TokenResponse {
  access_token: string;
  role: string;
  full_name: string;
}

export interface MetricSet {
  mae: number;
  smape: number;
  rmse: number;
  n: number;
}

export interface TrainResponse {
  run_id: number;
  gbm_beats_baseline: boolean;
  production_name: string;
  production_version: number | null;
  baseline: { holdout: MetricSet; backtest: MetricSet };
  gbm: { holdout: MetricSet; backtest: MetricSet };
}

export interface ModelVersion {
  id: number;
  name: string;
  version: number;
  algorithm: string;
  metrics_json: { holdout?: MetricSet; backtest?: MetricSet };
  artifact_path: string;
  stage: string;
  created_at: string;
}

export interface ForecastRow {
  sku_code: string;
  dc_code: string;
  week_index: number;
  week_start: string;
  yhat: number;
  yhat_lo: number;
  yhat_hi: number;
}

export interface DemandRow {
  sku_code: string;
  dc_code: string;
  family: string;
  week_index: number;
  week_start: string;
  units: number;
  promo_flag: boolean;
}

export interface FamilyDrift {
  family: string;
  psi: number;
  status: string;
  trailing_mae: number | null;
  backtest_mae: number | null;
  performance_drift: boolean;
  note: string;
}

export interface DriftResponse {
  reference_weeks: string;
  recent_weeks: string;
  families: FamilyDrift[];
}

function authHeaders(): HeadersInit {
  const token = localStorage.getItem("token");
  return token ? { Authorization: `Bearer ${token}` } : {};
}

async function request<T>(path: string, options: RequestInit = {}): Promise<T> {
  const response = await fetch(`${API_BASE}${path}`, {
    ...options,
    headers: {
      ...authHeaders(),
      ...(options.headers || {}),
    },
  });
  if (response.status === 401) {
    localStorage.removeItem("token");
    window.location.href = "/login";
    throw new Error("Unauthorized");
  }
  if (!response.ok) {
    const detail = await response.text();
    throw new Error(detail || response.statusText);
  }
  return response.json();
}

export async function login(email: string, password: string): Promise<TokenResponse> {
  const body = new URLSearchParams({ username: email, password });
  const response = await fetch(`${API_BASE}/auth/token`, {
    method: "POST",
    headers: { "Content-Type": "application/x-www-form-urlencoded" },
    body,
  });
  if (!response.ok) throw new Error("Invalid credentials");
  return response.json();
}

export const api = {
  train: (sku_code?: string) =>
    request<TrainResponse>("/train", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ sku_code: sku_code || null }),
    }),
  models: () => request<ModelVersion[]>("/models"),
  promote: (id: number, stage: string) =>
    request<ModelVersion>(`/models/${id}/promote`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ stage }),
    }),
  forecast: (horizon = 8) => request<ForecastRow[]>(`/forecast?horizon=${horizon}`),
  drift: () => request<DriftResponse>("/drift"),
  demand: (sku?: string, dc?: string) => {
    const params = new URLSearchParams();
    if (sku) params.set("sku", sku);
    if (dc) params.set("dc", dc);
    const q = params.toString();
    return request<DemandRow[]>(`/demand${q ? `?${q}` : ""}`);
  },
};
