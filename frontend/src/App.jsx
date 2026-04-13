import { useState, useEffect } from "react";
import Dashboard from "./pages/Dashboard";
import ComparePage from "./pages/ComparePage";
import MapPage from "./pages/MapPage";
import AttentionPage from "./pages/AttentionPage";

const NAV = [
  { id: "dashboard", icon: "▦",  label: "Dashboard"  },
  { id: "map",       icon: "◉",  label: "Live Map"   },
  { id: "attention", icon: "◎",  label: "Attention"  },
  { id: "compare",   icon: "⇄",  label: "Compare"    },
];

export default function App() {
  const [page, setPage]         = useState("dashboard");
  const [apiStatus, setStatus]  = useState("checking");

  useEffect(() => {
    fetch(`${import.meta.env.VITE_API_URL || "http://localhost:8000"}/health`)
      .then(r => r.json())
      .then(d => setStatus(d.model_loaded ? "ready" : "loading"))
      .catch(() => setStatus("offline"));
  }, []);

  return (
    <div className="app-shell">
      <nav className="sidebar">
        <div className="sidebar-logo">
          <div className="logo-mark">
            <div className="logo-icon">🚦</div>
            <span className="logo-name">TrafficGNN</span>
          </div>
          <div className="logo-sub">LA Freeway Intelligence</div>
        </div>

        <div className="sidebar-nav">
          <div className="nav-section-label">Navigation</div>
          {NAV.map(n => (
            <button
              key={n.id}
              className={`nav-item ${page === n.id ? "active" : ""}`}
              onClick={() => setPage(n.id)}
            >
              <span className="nav-icon">{n.icon}</span>
              {n.label}
            </button>
          ))}

          <div className="nav-section-label" style={{ marginTop: 16 }}>Dataset</div>
          <div style={{ padding: "6px 12px", fontSize: 11, color: "var(--muted)", lineHeight: 1.7 }}>
            <div style={{ color: "var(--text)", fontWeight: 500, marginBottom: 2 }}>METR-LA</div>
            <div>207 sensors</div>
            <div>4 months · 5-min intervals</div>
            <div>34,272 timesteps</div>
          </div>

          <div className="nav-section-label" style={{ marginTop: 8 }}>Model</div>
          <div style={{ padding: "6px 12px", fontSize: 11, color: "var(--muted)", lineHeight: 1.7 }}>
            <div style={{ color: "var(--text)", fontWeight: 500, marginBottom: 2 }}>ST-GNN</div>
            <div>GAT Spatial Encoder</div>
            <div>Transformer Temporal</div>
            <div>177,100 parameters</div>
          </div>
        </div>

        <div className="sidebar-footer">
          <div className="api-indicator">
            <span className={`api-dot ${apiStatus}`} />
            <span>API {apiStatus}</span>
          </div>
        </div>
      </nav>

      <main className="content">
        {page === "dashboard" && <Dashboard />}
        {page === "map"       && <MapPage />}
        {page === "attention" && <AttentionPage />}
        {page === "compare"   && <ComparePage />}
      </main>
    </div>
  );
}