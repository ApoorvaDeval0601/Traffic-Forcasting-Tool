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
      const data = await fetch(`${API}/attention/${sensorId}?top_k=10`).then(r => r.json());
      setAttention(data);
    } catch(e) { console.error(e); }
    setLoading(false);
  }

  useEffect(() => { if (attention) drawGraph(); }, [attention]);

  function drawGraph() {
    const canvas = canvasRef.current;
    if (!canvas || !attention) return;
    const ctx = canvas.getContext("2d");
    const W = canvas.width, H = canvas.height;
    ctx.clearRect(0,0,W,H);

    const cx = W/2, cy = H/2, R = Math.min(W,H)*0.36;
    const nbrs = attention.neighbors;
    const maxW = Math.max(...nbrs.map(n=>n.weight), 0.001);

    // Draw edges
    nbrs.forEach((n, i) => {
      const angle = (i/nbrs.length)*Math.PI*2 - Math.PI/2;
      const nx = cx + R*Math.cos(angle), ny = cy + R*Math.sin(angle);
      const alpha = n.weight/maxW;
      ctx.beginPath(); ctx.moveTo(cx,cy); ctx.lineTo(nx,ny);
      ctx.strokeStyle = `rgba(99,102,241,${(alpha*0.7+0.1).toFixed(2)})`;
      ctx.lineWidth = alpha*5 + 0.5;
      ctx.stroke();
    });

    // Draw neighbor nodes
    nbrs.forEach((n, i) => {
      const angle = (i/nbrs.length)*Math.PI*2 - Math.PI/2;
      const nx = cx + R*Math.cos(angle), ny = cy + R*Math.sin(angle);
      const alpha = n.weight/maxW;
      const nr = 10 + alpha*12;

      // Glow
      ctx.beginPath(); ctx.arc(nx,ny,nr+6,0,Math.PI*2);
      ctx.fillStyle = `rgba(239,68,68,${(alpha*0.2).toFixed(2)})`; ctx.fill();

      ctx.beginPath(); ctx.arc(nx,ny,nr,0,Math.PI*2);
      ctx.fillStyle = `rgba(239,68,68,${(0.3+alpha*0.7).toFixed(2)})`; ctx.fill();
      ctx.strokeStyle = `rgba(239,68,68,0.8)`; ctx.lineWidth = 1; ctx.stroke();

      ctx.fillStyle = "#f0f4ff"; ctx.font = `bold ${Math.max(9,11)}px Inter,sans-serif`;
      ctx.textAlign = "center"; ctx.textBaseline = "middle";
      ctx.fillText(`S${n.sensor_id}`, nx, ny);

      ctx.fillStyle = "rgba(107,122,153,0.9)"; ctx.font = "9px Inter,sans-serif";
      ctx.fillText(n.weight.toFixed(3), nx, ny + nr + 10);
    });

    // Source node
    ctx.beginPath(); ctx.arc(cx,cy,22,0,Math.PI*2);
    ctx.fillStyle = "rgba(99,102,241,0.15)"; ctx.fill();
    ctx.beginPath(); ctx.arc(cx,cy,18,0,Math.PI*2);
    ctx.fillStyle = "#6366f1"; ctx.fill();
    ctx.strokeStyle = "rgba(129,140,248,0.8)"; ctx.lineWidth = 2; ctx.stroke();
    ctx.fillStyle = "white"; ctx.font = "bold 12px Inter,sans-serif";
    ctx.textAlign = "center"; ctx.textBaseline = "middle";
    ctx.fillText(`S${attention.source_sensor}`, cx, cy-1);
    ctx.fillStyle = "rgba(129,140,248,0.7)"; ctx.font = "9px Inter,sans-serif";
    ctx.fillText("source", cx, cy+12);
  }

  return (
    <div>
      <header className="page-header">
        <h1>GAT Attention Weights</h1>
        <p className="subtitle">Which sensors influence each node's prediction — learned by the Graph Attention Network</p>
      </header>

      <div className="attention-controls">
        <label style={{ fontSize:11, color:"var(--muted)", textTransform:"uppercase", letterSpacing:"0.5px" }}>
          Source sensor ID
        </label>
        <input type="number" min="0" max="206" value={sensorId} onChange={e=>setSensorId(+e.target.value)} />
        <button className="btn btn-primary" onClick={fetchAttention} disabled={loading}>
          {loading ? "Loading..." : "Visualize →"}
        </button>
      </div>

      <div className="attention-layout">
        <canvas ref={canvasRef} width={480} height={480} className="attention-canvas" />

        {attention && (
          <div className="attention-table">
            <h3>Top-{attention.top_k} neighbors by attention weight</h3>
            <table className="data-table">
              <thead>
                <tr><th>Sensor</th><th>Weight</th><th>Influence</th></tr>
              </thead>
              <tbody>
                {attention.neighbors.map(n => (
                  <tr key={n.sensor_id}>
                    <td style={{ fontFamily:"var(--mono)" }}>S{n.sensor_id}</td>
                    <td style={{ fontFamily:"var(--mono)", color:"var(--accent2)" }}>{n.weight.toFixed(4)}</td>
                    <td style={{ width:120 }}>
                      <div style={{ background:"rgba(255,255,255,0.05)", borderRadius:3, height:5, overflow:"hidden" }}>
                        <div className="weight-bar" style={{
                          width:`${(n.weight/attention.neighbors[0].weight)*100}%`,
                          background:"var(--accent)",
                        }} />
                      </div>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
            <div className="attention-insight">
              <strong>Insight:</strong> Sensor {attention.source_sensor} attends most strongly to sensor {attention.neighbors[0]?.sensor_id} (weight {attention.neighbors[0]?.weight.toFixed(4)}), suggesting strong spatial dependency along that road segment.
            </div>
          </div>
        )}

        {!attention && (
          <div style={{ flex:1, display:"flex", alignItems:"center", justifyContent:"center", color:"var(--muted)", fontSize:13, border:"1px dashed var(--border)", borderRadius:12 }}>
            Enter a sensor ID and click Visualize to see attention weights
          </div>
        )}
      </div>
    </div>
  );
}