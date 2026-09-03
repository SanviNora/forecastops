import { BrowserRouter, Navigate, Route, Routes } from "react-router-dom";
import Layout from "./components/Layout";
import Demand from "./pages/Demand";
import Drift from "./pages/Drift";
import Forecast from "./pages/Forecast";
import Login from "./pages/Login";
import Registry from "./pages/Registry";
import Train from "./pages/Train";

function PrivateRoute({ children }: { children: JSX.Element }) {
  const token = localStorage.getItem("token");
  return token ? children : <Navigate to="/login" replace />;
}

export default function App() {
  return (
    <BrowserRouter>
      <Routes>
        <Route path="/login" element={<Login />} />
        <Route
          path="/"
          element={
            <PrivateRoute>
              <Layout />
            </PrivateRoute>
          }
        >
          <Route index element={<Demand />} />
          <Route path="train" element={<Train />} />
          <Route path="registry" element={<Registry />} />
          <Route path="forecast" element={<Forecast />} />
          <Route path="drift" element={<Drift />} />
        </Route>
      </Routes>
    </BrowserRouter>
  );
}
