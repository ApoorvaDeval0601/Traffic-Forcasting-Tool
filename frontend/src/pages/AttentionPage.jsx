// frontend/src/pages/AttentionPage.jsx
// Visualize which sensors attend to each other via GAT weights

import { useState, useEffect, useRef } from "react";

const API = import.meta.env.VITE_API_URL || "http://localhost:8000";

export default function AttentionPage() {
  const [sensorId, setSensorId] = useState(0);
  const [attention, setAttention] = useState(null);
  const [loading, setLoading] = useState(false);
  const canvasRef = useRef(null);

  async function fetchAttention() {
    setLoading(true);
    try {
      const res = await fetch(`${API}/attention/${sensorId}?top_k=10`);
      if (res.ok) setAttention(await res.json());
    } catch (e) {
      console.error(e);
    } finally {
      setLoading(false);
    }
  }

  useEffect(() => {
    if (attention) drawAttentionGraph();
  }, [attention]);

  function drawAttentionGraph() {
    const canvas = canvasRef.current;
    if (!canvas || !attention) return;
    const ctx = canvas.getContext("2d");
    const W = canvas.width;
    const H = canvas.height;
    ctx.clearRect(0, 0, W, H);

    const cx = W / 2;
    const cy = H / 2;
    const radius = Math.min(W, H) * 0.35;

    // Draw source node
    ctx.beginPath();
    ctx.arc(cx, cy, 24, 0, 2 * Math.PI);
    ctx.fillStyle = "#6366f1";
    ctx.fill();
    ctx.fillStyle = "#fff";
    ctx.font = "bold 11px sans-serif";
    ctx.textAlign = "center";
    ctx.textBaseline = "middle";
    ctx.fillText(`S${attention.source_sensor}`, cx, cy);

    // Draw neighbor nodes + edges
    const neighbors = attention.neighbors;
    const maxWeight = Math.max(...neighbors.map((n) => n.weight), 0.001);

    neighbors.forEach((n, i) => {
      const angle = (i / neighbors.length) * 2 * Math.PI - Math.PI / 2;
      const nx = cx + radius * Math.cos(angle);
      const ny = cy + radius * Math.sin(angle);
      const alpha = n.weight / maxWeight;

      // Edge
      ctx.beginPath();
      ctx.moveTo(cx, cy);
      ctx.lineTo(nx, ny);
      ctx.strokeStyle = `rgba(99, 102, 241, ${alpha.toFixed(2)})`;
      ctx.lineWidth = alpha * 5 + 0.5;
      ctx.stroke();

      // Node
      const nodeR = 12 + alpha * 12;
      ctx.beginPath();
      ctx.arc(nx, ny, nodeR, 0, 2 * Math.PI);
      ctx.fillStyle = `rgba(239, 68, 68, ${0.3 + alpha * 0.7})`;
      ctx.fill();
      ctx.strokeStyle = "#ef4444";
      ctx.lineWidth = 1;
      ctx.stroke();

      // Label
      ctx.fillStyle = "#1f2937";
      ctx.font = "10px sans-serif";
      ctx.textAlign = "center";
      ctx.textBaseline = "middle";
      ctx.fillText(`S${n.sensor_id}`, nx, ny);

      // Weight label
      ctx.fillStyle = "#6b7280";
      ctx.font = "9px sans-serif";
      ctx.fillText(n.weight.toFixed(3), nx, ny + nodeR + 10);
    });
  }

  return (
    <div className="attention-page">
      <header className="page-header">
        <h1>GAT Attention Weights</h1>
        <p className="subtitle">
          Which sensors most influence each node's predictions?
        </p>
      </header>

      <div className="attention-controls">
        <label>Source sensor ID</label>
        <input
          type="number"
          min="0"
          max="206"
          value={sensorId}
          onChange={(e) => setSensorId(+e.target.value)}
        />
        <button onClick={fetchAttention} disabled={loading}>
          {loading ? "Loading..." : "Visualize attention →"}
        </button>
      </div>

      <div className="attention-layout">
        <canvas
          ref={canvasRef}
          width={500}
          height={500}
          className="attention-canvas"
        />

        {attention && (
          <div className="attention-table">
            <h3>Top-{attention.top_k} neighbors</h3>
            <table>
              <thead>
                <tr>
                  <th>Sensor</th>
                  <th>Attention weight</th>
                  <th>Influence</th>
                </tr>
              </thead>
              <tbody>
                {attention.neighbors.map((n) => (
                  <tr key={n.sensor_id}>
                    <td>Sensor {n.sensor_id}</td>
                    <td>{n.weight.toFixed(4)}</td>
                    <td>
                      <div
                        className="weight-bar"
                        style={{
                          width: `${(n.weight / attention.neighbors[0].weight) * 100}%`,
                          background: "#6366f1",
                        }}
                      />
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>

            <div className="attention-insight">
              <strong>Insight:</strong> The source sensor attends most strongly
              to sensor {attention.neighbors[0]?.sensor_id}, suggesting strong
              spatial dependency along that road segment.
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
