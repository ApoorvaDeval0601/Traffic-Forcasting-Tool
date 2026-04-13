import { useState, useEffect, useRef } from "react";
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from "recharts";

const API = import.meta.env.VITE_API_URL || "http://localhost:8000";
const COLORS = { free: "#10b981", moderate: "#f59e0b", heavy: "#ef4444", severe: "#a855f7" };

const CustomTooltip = ({ active, payload, label }) => {
  if (!active || !payload?.length) return null;
  return (
    <div style={{ background: "#1a2235", border: "1px solid rgba(255,255,255,0.1)", borderRadius: 8, padding: "10px 14px", fontSize: 12 }}>
      <div style={{ color: "#6b7a99", marginBottom: 6 }}>{label}</div>
      {payload.map(p => (
        <div key={p.name} style={{ color: p.color, marginBottom: 3 }}>
          {p.name}: <strong>{p.value} mph</strong>
        </div>
      ))}
    </div>
  );
};

export default function Dashboard() {
  const [frame, setFrame]         = useState(null);
  const [selected, setSelected]   = useState(0);
  const [benchmark, setBenchmark] = useState(null);
  const [isLive, setIsLive]       = useState(false);
  const wsRef = useRef(null);

  useEffect(() => {
    fetchFrame();
    fetch(`${API}/benchmark`).then(r => r.json()).then(setBenchmark).catch(() => {});
  }, []);

  async function fetchFrame() {
    try {
      const data = await fetch(`${API}/frame`).then(r => r.json());
      setFrame(data);
    } catch (e) { console.error(e); }
  }

  function toggleLive() {
    if (isLive) { wsRef.current?.close(); setIsLive(false); }
    else {
      const ws = new WebSocket("ws://localhost:8000/ws/live");
      ws.onmessage = e => {
  const m = JSON.parse(e.data);
  if (m.type === "frame") {
    setFrame(prev => prev ? {
      ...prev,
      frame_idx: m.frame_idx,
      congestion_counts: m.congestion_counts,
    } : prev);
  }
};
      wsRef.current = ws;
      setIsLive(true);
    }
  }

  const sensor = frame?.sensors?.[selected];
  const counts = frame?.congestion_counts || {};

  const chartData = sensor ? sensor.timestamps.map((t, i) => ({
    time:      t,
    Predicted: sensor.predicted_speeds[i],
    Actual:    sensor.actual_speeds?.[i] ?? null,
  })) : [];

  return (
    <div>
      <header className="page-header">
        <div style={{ display:"flex", justifyContent:"space-between", alignItems:"flex-start" }}>
          <div>
            <h1>Traffic Forecast Dashboard</h1>
            <p className="subtitle">
              Real-time predictions from ST-GNN · METR-LA · 1-hour horizon
              {frame && <span style={{ marginLeft:12, fontFamily:"var(--mono)", fontSize:11 }}>
                frame {frame.frame_idx}/{frame.total_frames}
              </span>}
            </p>
          </div>
          <div style={{ display:"flex", gap:8 }}>
            <button className="btn" onClick={fetchFrame}>↻ Next frame</button>
            <button className={`btn ${isLive ? "btn-danger" : "btn-success"}`} onClick={toggleLive}>
              {isLive ? "⏹ Stop live" : "▶ Go live"}
            </button>
          </div>
        </div>
      </header>

      {/* Congestion cards */}
      <div className="metric-grid">
        {[["free","Free flow"],["moderate","Moderate"],["heavy","Heavy"],["severe","Severe"]].map(([k, label]) => (
          <div key={k} className="metric-card">
            <div className="metric-label">{label}</div>
            <div className="metric-value" style={{ color: COLORS[k] }}>{counts[k] || 0}</div>
            <div className="metric-sub">sensors</div>
          </div>
        ))}
        {benchmark?.transformer && (
          <div className="metric-card highlight">
            <div className="metric-label">Transformer MAE</div>
            <div className="metric-value" style={{ color: "var(--accent2)", fontSize:20 }}>
              {benchmark.transformer.mae_60min?.toFixed(2)}
            </div>
            <div className="metric-sub">mph at 60 min</div>
          </div>
        )}
        {benchmark?.lstm && (
          <div className="metric-card">
            <div className="metric-label">LSTM MAE</div>
            <div className="metric-value" style={{ color: "var(--amber)", fontSize:20 }}>
              {benchmark.lstm.mae_60min?.toFixed(2)}
            </div>
            <div className="metric-sub">mph at 60 min</div>
          </div>
        )}
      </div>

      {/* Chart */}
      <div className="chart-section">
        <div className="chart-controls">
          <label>Sensor</label>
          <select value={selected} onChange={e => setSelected(+e.target.value)}>
            {frame?.sensors?.slice(0,30).map(s => (
              <option key={s.id} value={s.id}>
                Sensor {s.id} — {s.current_speed} mph — {s.congestion}
              </option>
            ))}
          </select>
          {sensor && (
            <span className="badge" style={{ background: COLORS[sensor.congestion] + "25", color: COLORS[sensor.congestion], border: `1px solid ${COLORS[sensor.congestion]}40` }}>
              {sensor.current_speed} mph · {sensor.congestion}
            </span>
          )}
          <span style={{ marginLeft:"auto", fontSize:11, color:"var(--muted)" }}>
            Blue = predicted &nbsp;·&nbsp; Green = actual
          </span>
        </div>

        <ResponsiveContainer width="100%" height={260}>
          <LineChart data={chartData}>
            <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.05)" />
            <XAxis dataKey="time" tick={{ fill: "#6b7a99", fontSize: 11 }} axisLine={false} tickLine={false} />
            <YAxis domain={[0,85]} tick={{ fill: "#6b7a99", fontSize: 11 }} axisLine={false} tickLine={false}
              label={{ value: "Speed (mph)", angle: -90, position: "insideLeft", fill: "#6b7a99", fontSize: 11 }} />
            <Tooltip content={<CustomTooltip />} />
            <Legend wrapperStyle={{ fontSize: 12, color: "#6b7a99" }} />
            <Line type="monotone" dataKey="Predicted" stroke="#6366f1" strokeWidth={2.5} dot={{ r:3, fill:"#6366f1" }} activeDot={{ r:5 }} />
            <Line type="monotone" dataKey="Actual" stroke="#10b981" strokeWidth={2} strokeDasharray="5 4" dot={{ r:3, fill:"#10b981" }} activeDot={{ r:5 }} />
          </LineChart>
        </ResponsiveContainer>
      </div>

      {/* Comparison table */}
      {benchmark?.transformer && (
        <div className="comparison-table-section">
          <h2>Model Comparison — METR-LA Test Set</h2>
          <table className="data-table">
            <thead>
              <tr>
                <th>Metric</th>
                <th>ST-GNN Transformer</th>
                <th>ST-GNN LSTM</th>
                <th>Winner</th>
              </tr>
            </thead>
            <tbody>
              {[["MAE 15 min","mae_15min"],["MAE 30 min","mae_30min"],["MAE 60 min","mae_60min"],["RMSE 60 min","rmse_60min"]].map(([label,key]) => {
                const t = benchmark.transformer[key];
                const l = benchmark.lstm[key];
                const tWins = t < l;
                return (
                  <tr key={key}>
                    <td style={{ color:"var(--muted)" }}>{label}</td>
                    <td className={tWins ? "better" : ""}>{t?.toFixed(3)} mph</td>
                    <td className={!tWins ? "better" : ""}>{l?.toFixed(3)} mph</td>
                    <td style={{ color: tWins ? "var(--accent2)" : "var(--amber)", fontWeight:500, fontSize:12 }}>
                      {tWins ? "Transformer" : "LSTM"}
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