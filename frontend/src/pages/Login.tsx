import { FormEvent, useState } from "react";
import { useNavigate } from "react-router-dom";
import { login } from "../api/client";

export default function Login() {
  const navigate = useNavigate();
  const [email, setEmail] = useState("scientist@forecastops.example.com");
  const [password, setPassword] = useState("science-password");
  const [error, setError] = useState("");

  async function handleSubmit(e: FormEvent) {
    e.preventDefault();
    setError("");
    try {
      const token = await login(email, password);
      localStorage.setItem("token", token.access_token);
      localStorage.setItem("role", token.role);
      localStorage.setItem("full_name", token.full_name);
      navigate("/");
    } catch {
      setError("Invalid email or password.");
    }
  }

  return (
    <div className="login-page">
      <div className="card login-card">
        <h1>ForecastOps</h1>
        <p className="lede">SKU-level demand forecasting, registry, and drift for contact-lens supply.</p>
        <form onSubmit={handleSubmit}>
          <div style={{ marginBottom: "1rem" }}>
            <label htmlFor="email">Email</label>
            <input id="email" type="email" value={email} onChange={(e) => setEmail(e.target.value)} required />
          </div>
          <div style={{ marginBottom: "1rem" }}>
            <label htmlFor="password">Password</label>
            <input
              id="password"
              type="password"
              value={password}
              onChange={(e) => setPassword(e.target.value)}
              required
            />
          </div>
          {error && <p className="error">{error}</p>}
          <button type="submit" style={{ width: "100%" }}>
            Sign in
          </button>
        </form>
        <p className="muted" style={{ marginTop: "1rem", fontSize: "0.85rem" }}>
          Scientist demo is pre-filled. Ops: ops@forecastops.example.com / ops-password
        </p>
      </div>
    </div>
  );
}
