// frontend/src/App.jsx
// Main app shell with routing and layout

import { useState, useEffect } from "react";
import Dashboard from "./pages/Dashboard";
import ComparePage from "./pages/ComparePage";
import MapPage from "./pages/MapPage";
import AttentionPage from "./pages/AttentionPage";

const PAGES = ["dashboard", "map", "attention", "compare"];

export default function App() {
  const [page, setPage] = useState("dashboard");
  const [apiStatus, setApiStatus] = useState("checking");

  useEffect(() => {
    fetch(`${import.meta.env.VITE_API_URL || "http://localhost:8000"}/health`)
      .then((r) => r.json())
      .then((d) => setApiStatus(d.model_loaded ? "ready" : "loading"))
      .catch(() => setApiStatus("offline"));
  }, []);

  return (
    <div className="app-shell">
      <nav className="sidebar">
        <div className="logo">
          <span className="logo-icon">🚦</span>
          <span className="logo-text">Traffic GNN</span>
        </div>
        <div className="nav-links">
          {PAGES.map((p) => (
            <button
              key={p}
              className={`nav-item ${page === p ? "active" : ""}`}
              onClick={() => setPage(p)}
            >
              {NAV_ICONS[p]} {p.charAt(0).toUpperCase() + p.slice(1)}
            </button>
          ))}
        </div>
        <div className={`api-status status-${apiStatus}`}>
          <span className="status-dot" />
          API: {apiStatus}
        </div>
      </nav>

      <main className="content">
        {page === "dashboard" && <Dashboard />}
        {page === "map" && <MapPage />}
        {page === "attention" && <AttentionPage />}
        {page === "compare" && <ComparePage />}
      </main>
    </div>
  );
}

const NAV_ICONS = {
  dashboard: "📊",
  map: "🗺️",
  attention: "🔍",
  compare: "⚖️",
};
