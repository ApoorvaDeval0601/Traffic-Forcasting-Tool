import { useState, useEffect } from "react";
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, RadarChart, PolarGrid, PolarAngleAxis, Radar } from "recharts";

const API = import.meta.env.VITE_API_URL || "http://localhost:8000";

const DarkTooltip = ({ active, payload, label }) => {
  if (!active || !payload?.length) return null;
  return (
    <div style={{ background:"#1a2235", border:"1px solid rgba(255,255,255,0.1)", borderRadius:8, padding:"10px 14px", fontSize:12 }}>
      <div style={{ color:"#6b7a99", marginBottom:6 }}>{label}</div>
      {payload.map(p => <div key={p.name} style={{ color:p.fill, marginBottom:2 }}>{p.name}: <strong>{p.value} mph</strong></div>)}
    </div>
  );
};

export default function ComparePage() {
  const [metrics, setMetrics]   = useState(null);
  const [active, setActive]     = useState("mae");
  const [loading, setLoading]   = useState(true);

  useEffect(() => {
    fetch(`${API}/benchmark`).then(r=>r.json()).then(d=>{setMetrics(d);setLoading(false);}).catch(()=>setLoading(false));
  }, []);

  if (loading) return <div className="loading-text">Loading benchmark results...</div>;
  if (!metrics || metrics.error) return (
    <div className="error-text">
      No benchmark data. Run:<br/>
      <code style={{fontFamily:"var(--mono)",fontSize:12}}>
        python evaluation/benchmark.py --transformer-ckpt checkpoints/stgnn_metr_la/best_model.pt --lstm-ckpt checkpoints/lstm_metr_la/best_model.pt --dataset metr-la
      </code>
    </div>
  );

  const t = metrics.transformer, l = metrics.lstm;
  const barData = ["15min","30min","60min"].map(h => ({
    horizon: h,
    Transformer: +t?.[`${active}_${h}`]?.toFixed(3),
    LSTM:        +l?.[`${active}_${h}`]?.toFixed(3),
  }));

  const maxM = Math.max(t?.mae_60min||0, l?.mae_60min||0);
  const radarData = [
    { s:"MAE 15m",  T: 1-(t?.mae_15min/maxM), L: 1-(l?.mae_15min/maxM) },
    { s:"MAE 30m",  T: 1-(t?.mae_30min/maxM), L: 1-(l?.mae_30min/maxM) },
    { s:"MAE 60m",  T: 1-(t?.mae_60min/maxM), L: 1-(l?.mae_60min/maxM) },
    { s:"RMSE",     T: 1-(t?.rmse_60min/20),  L: 1-(l?.rmse_60min/20) },
    { s:"Params",   T: 1-(t?.params/300000),  L: 1-(l?.params/300000) },
  ];

  const imp = l?.mae_60min && t?.mae_60min ? (((l.mae_60min-t.mae_60min)/l.mae_60min)*100).toFixed(1) : "0";
  const tWins = parseFloat(imp) > 0;

  return (
    <div>
      <header className="page-header">
        <h1>Model Comparison</h1>
        <p className="subtitle">ST-GNN Transformer vs LSTM temporal encoder · METR-LA test set · {(6832).toLocaleString()} samples</p>
      </header>

      <div className="compare-summary">
        <div className="summary-card transformer">
          <div className="model-badge transformer-badge">Transformer</div>
          {[["MAE 15 min",t?.mae_15min],["MAE 30 min",t?.mae_30min],["MAE 60 min",t?.mae_60min],["RMSE 60 min",t?.rmse_60min],["Parameters",t?.params?.toLocaleString()]].map(([k,v])=>(
            <div key={k} className="summary-stat">
              <span className="stat-label">{k}</span>
              <span className="stat-value">{typeof v==="string"?v:(v?.toFixed(3)+" mph")}</span>
            </div>
          ))}
        </div>

        <div className="improvement-badge">
          <span className="imp-arrow">{tWins?"↑":"↓"}</span>
          <span className="imp-value" style={{color:tWins?"var(--green)":"var(--amber)"}}>{Math.abs(parseFloat(imp))}%</span>
          <span className="imp-label">{tWins?"Transformer better":"LSTM better"}<br/>at 60 min</span>
        </div>

        <div className="summary-card lstm">
          <div className="model-badge lstm-badge">LSTM</div>
          {[["MAE 15 min",l?.mae_15min],["MAE 30 min",l?.mae_30min],["MAE 60 min",l?.mae_60min],["RMSE 60 min",l?.rmse_60min],["Parameters",l?.params?.toLocaleString()]].map(([k,v])=>(
            <div key={k} className="summary-stat">
              <span className="stat-label">{k}</span>
              <span className="stat-value">{typeof v==="string"?v:(v?.toFixed(3)+" mph")}</span>
            </div>
          ))}
        </div>
      </div>

      <div className="metric-selector">
        {["mae","rmse","mape"].map(m=>(
          <button key={m} className={`metric-btn ${active===m?"active":""}`} onClick={()=>setActive(m)}>
            {m.toUpperCase()}
          </button>
        ))}
      </div>

      <div className="charts-grid">
        <div className="chart-card">
          <h3>{active.toUpperCase()} by horizon (mph)</h3>
          <ResponsiveContainer width="100%" height={240}>
            <BarChart data={barData}>
              <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.05)" />
              <XAxis dataKey="horizon" tick={{fill:"#6b7a99",fontSize:11}} axisLine={false} tickLine={false} />
              <YAxis tick={{fill:"#6b7a99",fontSize:11}} axisLine={false} tickLine={false} />
              <Tooltip content={<DarkTooltip />} />
              <Legend wrapperStyle={{fontSize:12,color:"#6b7a99"}} />
              <Bar dataKey="Transformer" fill="#6366f1" radius={[4,4,0,0]} />
              <Bar dataKey="LSTM"        fill="#f59e0b" radius={[4,4,0,0]} />
            </BarChart>
          </ResponsiveContainer>
        </div>

        <div className="chart-card">
          <h3>Performance profile</h3>
          <ResponsiveContainer width="100%" height={240}>
            <RadarChart data={radarData}>
              <PolarGrid stroke="rgba(255,255,255,0.07)" />
              <PolarAngleAxis dataKey="s" tick={{fill:"#6b7a99",fontSize:10}} />
              <Radar name="Transformer" dataKey="T" stroke="#6366f1" fill="#6366f1" fillOpacity={0.25} />
              <Radar name="LSTM"        dataKey="L" stroke="#f59e0b" fill="#f59e0b" fillOpacity={0.15} />
              <Legend wrapperStyle={{fontSize:12,color:"#6b7a99"}} />
            </RadarChart>
          </ResponsiveContainer>
        </div>
      </div>

      <div style={{marginBottom:12}}>
        <h2 style={{fontSize:13,fontWeight:600,textTransform:"uppercase",letterSpacing:"0.5px",color:"var(--text)",marginBottom:12}}>Key Findings</h2>
        <div className="findings-grid">
          {[
            {icon:"📊",title:"Real test set results",body:`Evaluated on 6,832 held-out samples. Transformer avg MAE: ${t?.mae?.toFixed(3)} mph, LSTM: ${l?.mae?.toFixed(3)} mph across all horizons.`},
            {icon:"🕐",title:"Long-horizon behavior",body:`At 15 min both models are nearly identical. The gap widens at 60 min — ${tWins?"Transformer captures longer temporal dependencies via self-attention":"LSTM's recency bias suits this 12-step sequence length"}.`},
            {icon:"🔗",title:"Shared spatial encoder",body:"Both models use the same GAT spatial encoder. The ablation isolates the temporal component — showing the contribution of attention vs recurrence."},
            {icon:"⚡",title:"Parameter efficiency",body:`Transformer (${t?.params?.toLocaleString()}) vs LSTM (${l?.params?.toLocaleString()}). ${Math.abs(parseFloat(imp))}% accuracy difference with ${Math.round((t?.params-l?.params)/1000)}K more parameters.`},
          ].map((f,i)=>(
            <div key={i} className="finding-card">
              <span className="finding-icon">{f.icon}</span>
              <h4>{f.title}</h4>
              <p>{f.body}</p>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}