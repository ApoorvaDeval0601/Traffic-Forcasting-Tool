// frontend/src/pages/Dashboard.jsx
// Main dashboard: live metrics + forecast chart + sensor selector

import { useState, useEffect } from "react";
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid,
  Tooltip, Legend, ResponsiveContainer,
} from "recharts";

const API = import.meta.env.VITE_API_URL || "http://localhost:8000";

const CONGESTION_COLORS = {
  free:     "#16a34a",
  moderate: "#ca8a04",
  heavy:    "#dc2626",
  severe:   "#7c3aed",
};

export default function Dashboard() {
  const [forecast, setForecast] = useState(null);
  const [selectedSensor, setSelectedSensor] = useState(0);
  const [loading, setLoading] = useState(true);
  const [metrics, setMetrics] = useState(null);

  useEffect(() => {
    fetchForecast();
    fetchMetrics();
    const interval = setInterval(fetchForecast, 60_000);
    return () => clearInterval(interval);
  }, []);

  async function fetchForecast() {
    try {
      const res = await fetch(`${API}/forecast`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ sensor_ids: null }),
      });
      const data = await res.json();
      setForecast(data);
    } catch (e) {
      console.error("Forecast fetch failed:", e);
    } finally {
      setLoading(false);
    }
  }

  async function fetchMetrics() {
    try {
      const res = await fetch(`${API}/model-comparison`);
      if (res.ok) setMetrics(await res.json());
    } catch (_) {}
  }

  const sensor = forecast?.sensors?.[selectedSensor];
  const chartData = sensor
    ? sensor.predicted_speeds.map((speed, i) => ({
        time: `+${(i + 1) * 5}m`,
        predicted: speed,
        threshold: 60,
      }))
    : [];

  // Congestion distribution
  const congestionCounts = forecast?.sensors?.reduce((acc, s) => {
    acc[s.congestion_level] = (acc[s.congestion_level] || 0) + 1;
    return acc;
  }, {}) || {};

  return (
    <div className="dashboard">
      <header className="page-header">
        <h1>Traffic Forecast Dashboard</h1>
        <p className="subtitle">METR-LA · 207 sensors · 1-hour horizon</p>
      </header>

      {/* Metric cards */}
      <div className="metric-grid">
        {Object.entries(congestionCounts).map(([level, count]) => (
          <div key={level} className="metric-card">
            <span className="metric-label">{level}</span>
            <span
              className="metric-value"
              style={{ color: CONGESTION_COLORS[level] }}
            >
              {count}
            </span>
            <span className="metric-sub">sensors</span>
          </div>
        ))}
        {metrics && (
          <div className="metric-card highlight">
            <span className="metric-label">Model MAE</span>
            <span className="metric-value">{metrics.transformer?.mae?.toFixed(3)}</span>
            <span className="metric-sub">mph (60-min)</span>
          </div>
        )}
      </div>

      {/* Sensor selector + chart */}
      <div className="chart-section">
        <div className="chart-controls">
          <label>Sensor</label>
          <select
            value={selectedSensor}
            onChange={(e) => setSelectedSensor(+e.target.value)}
          >
            {forecast?.sensors?.slice(0, 20).map((s, i) => (
              <option key={s.sensor_id} value={i}>
                {s.name || `Sensor ${s.sensor_id}`} — {s.congestion_level}
              </option>
            ))}
          </select>

          {sensor && (
            <div
              className="congestion-badge"
              style={{ background: CONGESTION_COLORS[sensor.congestion_level] }}
            >
              {sensor.current_speed} mph · {sensor.congestion_level}
            </div>
          )}
        </div>

        <div className="chart-container">
          {loading ? (
            <p className="loading-text">Loading forecast...</p>
          ) : (
            <ResponsiveContainer width="100%" height={280}>
              <LineChart data={chartData}>
                <CartesianGrid strokeDasharray="3 3" stroke="#e5e7eb" />
                <XAxis dataKey="time" tick={{ fontSize: 12 }} />
                <YAxis
                  domain={[0, 80]}
                  label={{ value: "Speed (mph)", angle: -90, position: "insideLeft", fontSize: 12 }}
                />
                <Tooltip
                  formatter={(val) => [`${val} mph`]}
                  labelFormatter={(l) => `Horizon: ${l}`}
                />
                <Legend />
                <Line
                  type="monotone"
                  dataKey="predicted"
                  stroke="#6366f1"
                  strokeWidth={2.5}
                  dot={{ r: 3 }}
                  name="Predicted speed"
                />
                <Line
                  type="monotone"
                  dataKey="threshold"
                  stroke="#16a34a"
                  strokeDasharray="5 5"
                  strokeWidth={1}
                  dot={false}
                  name="Free flow (60 mph)"
                />
              </LineChart>
            </ResponsiveContainer>
          )}
        </div>
      </div>

      {/* Model comparison if available */}
      {metrics && (
        <div className="comparison-table-section">
          <h2>Model Comparison (METR-LA)</h2>
          <table className="comparison-table">
            <thead>
              <tr>
                <th>Metric</th>
                <th>ST-GNN (Transformer)</th>
                <th>ST-GNN (LSTM)</th>
                <th>Improvement</th>
              </tr>
            </thead>
            <tbody>
              {[
                ["MAE 15min", "mae_15min"],
                ["MAE 30min", "mae_30min"],
                ["MAE 60min", "mae_60min"],
                ["RMSE 60min", "rmse_60min"],
              ].map(([label, key]) => {
                const t = metrics.transformer?.[key];
                const l = metrics.lstm?.[key];
                const imp = t && l ? (((l - t) / l) * 100).toFixed(1) : null;
                return (
                  <tr key={key}>
                    <td>{label}</td>
                    <td className="better">{t?.toFixed(3) ?? "—"}</td>
                    <td>{l?.toFixed(3) ?? "—"}</td>
                    <td className="improvement">
                      {imp ? `↑ ${imp}%` : "—"}
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}
