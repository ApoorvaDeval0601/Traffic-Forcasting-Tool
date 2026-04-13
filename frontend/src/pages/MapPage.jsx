// MapPage.jsx — Real OSM tile background + sensor overlay, no grey box
import { useState, useEffect, useRef } from "react";
import { LineChart, Line, XAxis, YAxis, Tooltip, ResponsiveContainer, ReferenceLine } from "recharts";

function SensorChart({ sensor }) {
  const data = sensor.predicted_speeds.map((pred, i) => ({
    t: `+${(i+1)*5}m`,
    Predicted: pred,
    Actual: sensor.actual_speeds?.[i] ?? null,
  }));
  const mae = sensor.actual_speeds
    ? (sensor.predicted_speeds.reduce((s,v,i)=>s+Math.abs(v-(sensor.actual_speeds[i]||0)),0)/12).toFixed(2)
    : null;
  return (
    <div style={{ marginTop:12 }}>
      <div style={{ display:"flex", justifyContent:"space-between", alignItems:"baseline", marginBottom:6 }}>
        <span style={{ fontSize:11, fontWeight:600, color:"var(--text)" }}>Next 60 minutes</span>
        {mae && <span style={{ fontSize:10, color:"var(--accent2)" }}>MAE {mae} mph</span>}
      </div>
      <ResponsiveContainer width="100%" height={110}>
        <LineChart data={data} margin={{ top:4, right:4, left:-28, bottom:0 }}>
          <XAxis dataKey="t" tick={{ fill:"#6b7a99", fontSize:9 }} axisLine={false} tickLine={false} interval={2} />
          <YAxis tick={{ fill:"#6b7a99", fontSize:9 }} axisLine={false} tickLine={false} domain={["auto","auto"]} />
          <Tooltip
            contentStyle={{ background:"#111827", border:"1px solid rgba(255,255,255,0.08)", borderRadius:6, fontSize:11 }}
            labelStyle={{ color:"#6b7a99" }}
            formatter={(v,n) => [`${v} mph`, n]}
          />
          <ReferenceLine y={60} stroke="rgba(255,255,255,0.1)" strokeDasharray="3 3" />
          <Line type="monotone" dataKey="Predicted" stroke="#6366f1" strokeWidth={2} dot={false} />
          <Line type="monotone" dataKey="Actual" stroke="#10b981" strokeWidth={1.5} dot={false} strokeDasharray="4 2" />
        </LineChart>
      </ResponsiveContainer>
      <div style={{ display:"flex", gap:12, marginTop:2 }}>
        <span style={{ fontSize:10, color:"var(--muted)", display:"flex", alignItems:"center", gap:4 }}>
          <span style={{ width:12, height:2, background:"#6366f1", display:"inline-block", borderRadius:1 }}/> Predicted
        </span>
        <span style={{ fontSize:10, color:"var(--muted)", display:"flex", alignItems:"center", gap:4 }}>
          <span style={{ width:12, height:2, background:"#10b981", display:"inline-block", borderRadius:1 }}/> Actual
        </span>
      </div>
    </div>
  );
}

const API = import.meta.env.VITE_API_URL || "http://localhost:8000";
const COLORS = { free: "#10b981", moderate: "#f59e0b", heavy: "#ef4444", severe: "#a855f7" };
const LABELS = { free: "Free flow >60mph", moderate: "Moderate 40-60mph", heavy: "Heavy 20-40mph", severe: "Severe <20mph" };

const BBOX = { minLat: 33.93, maxLat: 34.22, minLng: -118.52, maxLng: -118.10 };

function latToMerc(lat) {
  const rad = (lat * Math.PI) / 180;
  return Math.log(Math.tan(Math.PI / 4 + rad / 2));
}
const MERC_TOP    = latToMerc(BBOX.maxLat);
const MERC_BOTTOM = latToMerc(BBOX.minLat);

function geo(lat, lng, W, H) {
  const x = ((lng - BBOX.minLng) / (BBOX.maxLng - BBOX.minLng)) * W;
  const y = ((MERC_TOP - latToMerc(lat)) / (MERC_TOP - MERC_BOTTOM)) * H;
  return { x, y };
}

function lngToTileX(lng, z) { return Math.floor((lng + 180) / 360 * Math.pow(2, z)); }
function latToTileY(lat, z) {
  const r = lat * Math.PI / 180;
  return Math.floor((1 - Math.log(Math.tan(r) + 1 / Math.cos(r)) / Math.PI) / 2 * Math.pow(2, z));
}
function tileTopLeftGeo(tx, ty, z) {
  const n = Math.pow(2, z);
  const lng = tx / n * 360 - 180;
  const latRad = Math.atan(Math.sinh(Math.PI * (1 - 2 * ty / n)));
  return { lat: latRad * 180 / Math.PI, lng };
}

const TILE_CACHE = {};
function getTile(tx, ty, z) {
  const s = ["a","b","c"][Math.abs(tx + ty) % 3];
  const url = `https://${s}.tile.openstreetmap.org/${z}/${tx}/${ty}.png`;
  if (TILE_CACHE[url]) return TILE_CACHE[url];
  const img = new Image();
  img.crossOrigin = "anonymous";
  img.src = url;
  TILE_CACHE[url] = img;
  return img;
}

export default function MapPage() {
  const canvasRef    = useRef(null);
  const frameRef     = useRef(null);
  const roadRef      = useRef(null);
  const autoRef      = useRef(null);
  const transformRef = useRef({ scale: 1, tx: 0, ty: 0 });
  const dragRef      = useRef({ dragging: false, lastX: 0, lastY: 0, movedPx: 0 });

  const [frame, setFrame]       = useState(null);
  const [selected, setSelected] = useState(null);
  const [autoplay, setAutoplay] = useState(false);
  const [loading, setLoading]   = useState(false);
  const [zoom, setZoom]         = useState(1);
  const [tileStatus, setTileStatus] = useState("loading");

  const TILE_ZOOM = 12;

  useEffect(() => {
    loadRoads();
    fetchFrame();
    preloadTiles();
    return () => clearInterval(autoRef.current);
  }, []);

  useEffect(() => {
    if (frame) { frameRef.current = frame; draw(); }
  }, [frame, selected]);

  function preloadTiles() {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const W = canvas.width, H = canvas.height;
    const z = TILE_ZOOM;
    const x0 = lngToTileX(BBOX.minLng, z), x1 = lngToTileX(BBOX.maxLng, z);
    const y0 = latToTileY(BBOX.maxLat, z), y1 = latToTileY(BBOX.minLat, z);
    let loaded = 0;
    const total = (x1 - x0 + 1) * (y1 - y0 + 1);
    for (let tx = x0; tx <= x1; tx++) {
      for (let ty = y0; ty <= y1; ty++) {
        const img = getTile(tx, ty, z);
        const done = () => {
          loaded++;
          if (loaded >= Math.floor(total * 0.5)) setTileStatus("ready");
          if (frameRef.current) draw();
        };
        if (img.complete) done();
        else { img.onload = done; img.onerror = done; }
      }
    }
  }

  async function loadRoads() {
    try {
      const data = await fetch(`${API}/road-edges`).then(r => r.json());
      roadRef.current = data;
      if (frameRef.current) draw();
    } catch (e) { console.error(e); }
  }

  async function fetchFrame() {
    setLoading(true);
    try {
      const data = await fetch(`${API}/frame`).then(r => r.json());
      setFrame(data);
    } catch (e) { console.error(e); }
    setLoading(false);
  }

  function drawTiles(ctx, W, H) {
    const z = TILE_ZOOM;
    const x0 = lngToTileX(BBOX.minLng, z), x1 = lngToTileX(BBOX.maxLng, z);
    const y0 = latToTileY(BBOX.maxLat, z), y1 = latToTileY(BBOX.minLat, z);
    for (let tx = x0; tx <= x1; tx++) {
      for (let ty = y0; ty <= y1; ty++) {
        const tl = tileTopLeftGeo(tx,   ty,   z);
        const br = tileTopLeftGeo(tx+1, ty+1, z);
        const p1 = geo(tl.lat, tl.lng, W, H);
        const p2 = geo(br.lat, br.lng, W, H);
        const pw = p2.x - p1.x, ph = p2.y - p1.y;
        const img = getTile(tx, ty, z);
        if (img.complete && img.naturalWidth > 0) {
          ctx.drawImage(img, p1.x, p1.y, pw + 1, ph + 1);
        } else {
          ctx.fillStyle = "#e8e0d8";
          ctx.fillRect(p1.x, p1.y, pw + 1, ph + 1);
        }
      }
    }
  }

  function draw() {
    const canvas = canvasRef.current;
    if (!canvas || !frameRef.current?.sensors) return;
    const ctx = canvas.getContext("2d");
    const W = canvas.width, H = canvas.height;
    const { scale, tx, ty } = transformRef.current;
    const sensors = frameRef.current.sensors;

    // Clear to map-like beige so no dark gaps
    ctx.fillStyle = "#f2ece3";
    ctx.fillRect(0, 0, W, H);

    ctx.save();
    ctx.translate(tx, ty);
    ctx.scale(scale, scale);

    // OSM tiles
    drawTiles(ctx, W, H);

    // Subtle dark tint so dots pop
    ctx.fillStyle = "rgba(0, 0, 0, 0)";
    ctx.fillRect(0, 0, W, H);

    // Road edges — deduplicated, one line per pair
    if (roadRef.current) {
      const { edges, coords } = roadRef.current;
      ctx.lineCap = "round";
      // Deduplicate: only draw src < dst
      const seen = new Set();
      edges.forEach(([src, dst]) => {
        const key = src < dst ? src + "-" + dst : dst + "-" + src;
        if (seen.has(key)) return;
        seen.add(key);
        if (src >= sensors.length || dst >= sensors.length) return;
        const sa = sensors[src], sb = sensors[dst];
        const a = geo(coords[src][0], coords[src][1], W, H);
        const b = geo(coords[dst][0], coords[dst][1], W, H);
        const avg = (sa.current_speed + sb.current_speed) / 2;
        const c = avg > 60 ? "#10b981" : avg > 40 ? "#f59e0b" : avg > 20 ? "#ef4444" : "#a855f7";
        // Thin dark casing
        ctx.strokeStyle = "rgba(0,0,0,0.35)";
        ctx.lineWidth = 2.5 / scale;
        ctx.beginPath(); ctx.moveTo(a.x, a.y); ctx.lineTo(b.x, b.y); ctx.stroke();
        // Colored road line
        ctx.strokeStyle = c;
        ctx.lineWidth = 1.5 / scale;
        ctx.globalAlpha = 0.85;
        ctx.beginPath(); ctx.moveTo(a.x, a.y); ctx.lineTo(b.x, b.y); ctx.stroke();
        ctx.globalAlpha = 1;
      });
    }

    // Sensor dots
    const dotR = Math.max(2.5, 4.5 / scale);
    sensors.forEach(s => {
      const { x, y } = geo(s.lat, s.lng, W, H);
      const color = COLORS[s.congestion] || "#6b7280";
      const isSel = selected?.id === s.id;

      if (s.congestion === "heavy" || s.congestion === "severe") {
        ctx.beginPath(); ctx.arc(x, y, dotR * 2, 0, Math.PI * 2);
        ctx.fillStyle = color + "33"; ctx.fill();
      }
      if (isSel) {
        ctx.beginPath(); ctx.arc(x, y, dotR * 2.8, 0, Math.PI * 2);
        ctx.strokeStyle = "white"; ctx.lineWidth = 1.5 / scale;
        ctx.setLineDash([3/scale, 2/scale]); ctx.stroke(); ctx.setLineDash([]);
      }
      ctx.beginPath(); ctx.arc(x, y, isSel ? dotR * 1.5 : dotR, 0, Math.PI * 2);
      ctx.fillStyle = color; ctx.fill();
      ctx.strokeStyle = "rgba(255,255,255,0.9)"; ctx.lineWidth = 1 / scale; ctx.stroke();

      if (scale > 3) {
        ctx.fillStyle = "white";
        ctx.font = `bold ${Math.min(10, 9/scale*3)}px Inter,sans-serif`;
        ctx.textAlign = "left"; ctx.textBaseline = "middle";
        ctx.fillText(`${s.id}`, x + dotR + 2/scale, y);
      }
    });

    // Tooltip
    if (selected) {
      const s = sensors.find(s => s.id === selected.id);
      if (s) {
        const { x, y } = geo(s.lat, s.lng, W, H);
        const bw = 165/scale, bh = 48/scale, r = 5/scale;
        const bx = Math.min(x + 12/scale, W/scale - bw - 8/scale);
        const by = Math.max(y - 56/scale, 6/scale);
        ctx.fillStyle = "rgba(10,14,26,0.95)";
        ctx.beginPath(); ctx.roundRect(bx, by, bw, bh, r); ctx.fill();
        ctx.strokeStyle = "rgba(255,255,255,0.15)"; ctx.lineWidth = 0.7/scale; ctx.stroke();
        ctx.fillStyle = "#ffffff"; ctx.font = `bold ${12/scale}px Inter,sans-serif`;
        ctx.textAlign = "left"; ctx.textBaseline = "top";
        ctx.fillText(`Sensor ${s.id}`, bx + 8/scale, by + 8/scale);
        ctx.fillStyle = COLORS[s.congestion]; ctx.font = `${11/scale}px Inter,sans-serif`;
        ctx.fillText(`${s.current_speed} mph  ·  ${s.congestion}`, bx + 8/scale, by + 26/scale);
      }
    }

    ctx.restore();

    // Fixed bottom overlay
    ctx.fillStyle = "rgba(0,0,0,0.5)";
    ctx.fillRect(0, H - 24, W, 24);
    ctx.fillStyle = "rgba(255,255,255,0.5)";
    ctx.font = "10px Inter,sans-serif";
    ctx.textAlign = "left"; ctx.textBaseline = "middle";
    ctx.fillText("© OpenStreetMap contributors  ·  Scroll to zoom  ·  Drag to pan", 10, H - 12);
    ctx.textAlign = "right";
    ctx.fillText(`${zoom.toFixed(1)}×`, W - 10, H - 12);
  }

  function onWheel(e) {
    e.preventDefault();
    const canvas = canvasRef.current;
    const rect = canvas.getBoundingClientRect();
    const cx = (e.clientX - rect.left) * (canvas.width / rect.width);
    const cy = (e.clientY - rect.top)  * (canvas.height / rect.height);
    const { scale, tx, ty } = transformRef.current;
    const factor = e.deltaY < 0 ? 1.15 : 0.87;
    const newScale = Math.max(0.5, Math.min(12, scale * factor));
    transformRef.current = {
      scale: newScale,
      tx: cx - (cx - tx) * (newScale / scale),
      ty: cy - (cy - ty) * (newScale / scale),
    };
    setZoom(newScale);
    draw();
  }

  function onMouseDown(e) {
    dragRef.current = { dragging: true, lastX: e.clientX, lastY: e.clientY, movedPx: 0 };
  }

  function onMouseMove(e) {
    if (!dragRef.current.dragging) return;
    const dx = e.clientX - dragRef.current.lastX;
    const dy = e.clientY - dragRef.current.lastY;
    dragRef.current.movedPx += Math.abs(dx) + Math.abs(dy);
    dragRef.current.lastX = e.clientX;
    dragRef.current.lastY = e.clientY;
    const t = transformRef.current;
    transformRef.current = { scale: t.scale, tx: t.tx + dx, ty: t.ty + dy };
    draw();
  }

  function onMouseUp(e) {
    const moved = dragRef.current.movedPx;
    dragRef.current.dragging = false;
    dragRef.current.movedPx = 0;
    if (moved < 5) handleClick(e);
  }

  function handleClick(e) {
    if (!frameRef.current) return;
    const canvas = canvasRef.current;
    const rect = canvas.getBoundingClientRect();
    const { scale, tx, ty } = transformRef.current;
    const cx = (e.clientX - rect.left) * (canvas.width  / rect.width);
    const cy = (e.clientY - rect.top)  * (canvas.height / rect.height);
    const wx = (cx - tx) / scale, wy = (cy - ty) / scale;
    const W = canvas.width, H = canvas.height;
    const hitR = Math.max(12, 12 / scale);
    let closest = null, minDist = hitR;
    frameRef.current.sensors.forEach(s => {
      const { x, y } = geo(s.lat, s.lng, W, H);
      const d = Math.sqrt((wx-x)**2 + (wy-y)**2);
      if (d < minDist) { minDist = d; closest = s; }
    });
    if (closest) setSelected(closest);
  }

  function resetView() {
    transformRef.current = { scale: 1, tx: 0, ty: 0 };
    setZoom(1); draw();
  }

  function toggleAutoplay() {
    if (autoplay) { clearInterval(autoRef.current); setAutoplay(false); }
    else { autoRef.current = setInterval(fetchFrame, 2500); setAutoplay(true); }
  }

  const counts = frame?.sensors?.reduce((a, s) => {
    a[s.congestion] = (a[s.congestion]||0)+1; return a;
  }, {}) || {};

  return (
    <div className="map-page">
      <header className="page-header">
        <div style={{ display:"flex", justifyContent:"space-between", alignItems:"flex-start" }}>
          <div>
            <h1>Live Traffic Map</h1>
            <p className="subtitle">
              Los Angeles · 207 METR-LA sensors · Real road network
              {tileStatus === "ready"   && <span style={{ color:"var(--green)",  marginLeft:8 }}>● Map loaded</span>}
              {tileStatus === "loading" && <span style={{ color:"var(--amber)",  marginLeft:8 }}>● Loading tiles...</span>}
            </p>
          </div>
          <div style={{ display:"flex", gap:8 }}>
            <button className="btn" onClick={resetView}>⊙ Reset</button>
            <button className="btn" onClick={fetchFrame} disabled={loading}>{loading ? "..." : "↻ Next frame"}</button>
            <button className={`btn ${autoplay ? "btn-danger" : "btn-primary"}`} onClick={toggleAutoplay}>
              {autoplay ? "⏹ Stop" : "▶ Animate"}
            </button>
          </div>
        </div>
      </header>

      <div className="map-layout">
        <div className="map-controls">
          <div className="legend">
            <h4>Congestion level</h4>
            {Object.entries(COLORS).map(([level, color]) => (
              <div key={level} className="legend-item">
                <div className="legend-row">
                  <span className="legend-dot" style={{ background:color }} />
                  <span className="legend-name">{level}</span>
                  <span className="legend-count" style={{ color }}>{counts[level]||0}</span>
                </div>
                <span className="legend-desc">{LABELS[level]}</span>
              </div>
            ))}
          </div>

          <div className="card" style={{ fontSize:12 }}>
            <div className="card-label">Frame</div>
            <div style={{ fontFamily:"var(--mono)", color:"var(--text)" }}>
              {frame ? `${frame.frame_idx} / ${frame.total_frames}` : "—"}
            </div>
            <div style={{ marginTop:4, fontSize:11, color:"var(--muted)" }}>
              Zoom: {zoom.toFixed(1)}×
              {zoom > 3 && <span style={{ color:"var(--accent2)" }}> · IDs visible</span>}
            </div>
            {autoplay && <div style={{ color:"var(--green)", marginTop:4, fontSize:11 }}>● Animating</div>}
          </div>

          {selected ? (
            <div className="sensor-panel">
              <div style={{ display:"flex", justifyContent:"space-between" }}>
                <h4>Sensor {selected.id}</h4>
                <button onClick={()=>setSelected(null)} style={{ background:"none", border:"none", cursor:"pointer", color:"var(--muted)", fontSize:18 }}>×</button>
              </div>
              <div style={{ fontSize:11, color:"var(--muted)" }}>
                {selected.lat?.toFixed(5)}°N, {Math.abs(selected.lng)?.toFixed(5)}°W
              </div>
              <div style={{ fontSize:14, marginTop:6, fontWeight:600 }}>{selected.current_speed} mph</div>
              <span className="badge" style={{ marginTop:4, background:COLORS[selected.congestion]+"20", color:COLORS[selected.congestion], border:`1px solid ${COLORS[selected.congestion]}40` }}>
                {selected.congestion}
              </span>
              {selected.predicted_speeds && (
                <SensorChart sensor={selected} />
              )}
            </div>
          ) : (
            <div style={{ padding:"14px", background:"var(--surface)", border:"1px dashed var(--border2)", borderRadius:10, textAlign:"center", color:"var(--muted)", fontSize:12, lineHeight:1.7 }}>
              Click any dot to see<br/>its 60-min forecast
            </div>
          )}
        </div>

        <div style={{ flex:1, borderRadius:12, border:"1px solid var(--border)", overflow:"hidden", minHeight:520 }}>
          <canvas
            ref={canvasRef}
            width={900} height={600}
            onWheel={onWheel}
            onMouseDown={onMouseDown}
            onMouseMove={onMouseMove}
            onMouseUp={onMouseUp}
            onMouseLeave={()=>{ dragRef.current.dragging=false; }}
            style={{ width:"100%", height:"100%", display:"block", cursor:"crosshair" }}
          />
        </div>
      </div>
    </div>
  );
}