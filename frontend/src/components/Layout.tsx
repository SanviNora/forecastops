import { NavLink, Outlet, useNavigate } from "react-router-dom";

export default function Layout() {
  const navigate = useNavigate();
  const name = localStorage.getItem("full_name") || "";
  const role = localStorage.getItem("role") || "";

  function logout() {
    localStorage.removeItem("token");
    localStorage.removeItem("role");
    localStorage.removeItem("full_name");
    navigate("/login");
  }

  return (
    <>
      <nav className="nav">
        <span className="nav-brand">ForecastOps</span>
        <div className="nav-links">
          <NavLink to="/" end>
            Demand
          </NavLink>
          <NavLink to="/train">Train</NavLink>
          <NavLink to="/registry">Registry</NavLink>
          <NavLink to="/forecast">Forecast</NavLink>
          <NavLink to="/drift">Drift</NavLink>
        </div>
        <span className="muted">
          {name} · {role}
        </span>
        <button className="secondary" onClick={logout}>
          Log out
        </button>
      </nav>
      <main className="layout">
        <Outlet />
      </main>
    </>
  );
}
