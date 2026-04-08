// frontend/src/pages/MapPage.jsx
// Live traffic map using Leaflet with congestion overlay

import { useState, useEffect, useRef } from "react";

const API = import.meta.env.VITE_API_URL || "http://localhost:8000";

const CONGESTION_COLORS = {
  free:     "#16a34a",
  moderate: "#ca8a04",
  heavy:    "#dc2626",
  severe:   "#7c3aed",
};

// LA bounding box defaults
const LA_CENTER = [34.0522, -118.2437];

export default function MapPage() {
  const mapRef = useRef(null);
  const leafletMap = useRef(null);
  const markersRef = useRef({});
  const [congestion, setCongestion] = useState(null);
  const [selectedSensor, setSelectedSensor] = useState(null);
  const [forecast, setForecast] = useState(null);
  const [showAttention, setShowAttention] = useState(false);
  const [attentionLines, setAttentionLines] = useState([]);

  // Init Leaflet map
  useEffect(() => {
    if (!mapRef.current || leafletMap.current) return;
    // Leaflet must be loaded via CDN in index.html
    const L = window.L;
    if (!L) return;

    leafletMap.current = L.map(mapRef.current).setView(LA_CENTER, 11);
    L.tileLayer("https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png", {
      attribution: "© OpenStreetMap",
      maxZoom: 18,
    }).addTo(leafletMap.current);

    fetchCongestion();
    const interval = setInterval(fetchCongestion, 30_000);
    return () => {
      clearInterval(interval);
      leafletMap.current?.remove();
    };
  }, []);

  async function fetchCongestion() {
    try {
      const res = await fetch(`${API}/congestion-map`);
      const data = await res.json();
      setCongestion(data);
      updateMarkers(data);
    } catch (e) {
      console.error("Congestion fetch failed:", e);
    }
  }

  function updateMarkers(data) {
    const L = window.L;
    if (!L || !leafletMap.current) return;

    Object.entries(data).forEach(([sid, info]) => {
      const color = CONGESTION_COLORS[info.congestion] || "#6b7280";
      const icon = L.divIcon({
        className: "",
        html: `<div style="width:10px;height:10px;border-radius:50%;background:${color};border:1.5px solid white;box-shadow:0 1px 3px rgba(0,0,0,0.3)"></div>`,
        iconSize: [10, 10],
        iconAnchor: [5, 5],
      });

      if (markersRef.current[sid]) {
        markersRef.current[sid].setIcon(icon);
      } else {
        const marker = L.marker([info.lat, info.lng], { icon })
          .addTo(leafletMap.current)
          .on("click", () => handleSensorClick(sid, info));
        markersRef.current[sid] = marker;
      }
    });
  }

  async function handleSensorClick(sid, info) {
    setSelectedSensor({ id: sid, ...info });
    // Fetch 1-hour forecast for this sensor
    try {
      const res = await fetch(`${API}/forecast`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ sensor_ids: [parseInt(sid)] }),
      });
      const data = await res.json();
      setForecast(data.sensors?.[0] ?? null);
    } catch (_) {}

    if (showAttention) fetchAttention(parseInt(sid));
  }

  async function fetchAttention(sensorId) {
    const L = window.L;
    if (!L || !leafletMap.current) return;

    // Clear old lines
    attentionLines.forEach((l) => l.remove());
    try {
      const res = await fetch(`${API}/attention/${sensorId}?top_k=8`);
      const data = await res.json();
      const srcInfo = congestion?.[String(sensorId)];
      if (!srcInfo) return;

      const lines = data.neighbors.map((n) => {
        const weight = n.weight;
        return L.polyline(
          [[srcInfo.lat, srcInfo.lng], [n.lat, n.lng]],
          {
            color: "#6366f1",
            weight: Math.max(1, weight * 15),
            opacity: Math.min(0.9, weight * 5 + 0.2),
            dashArray: "4 4",
          }
        ).addTo(leafletMap.current);
      });
      setAttentionLines(lines);
    } catch (_) {}
  }

  const stats = congestion
    ? Object.values(congestion).reduce((acc, s) => {
        acc[s.congestion] = (acc[s.congestion] || 0) + 1;
        return acc;
      }, {})
    : {};

  return (
    <div className="map-page">
      <header className="page-header">
        <h1>Live Traffic Map</h1>
        <p className="subtitle">Real-time congestion overlay · Click a sensor for forecast</p>
      </header>

      <div className="map-layout">
        {/* Controls panel */}
        <div className="map-controls">
          <div className="legend">
            <h4>Congestion level</h4>
            {Object.entries(CONGESTION_COLORS).map(([level, color]) => (
              <div key={level} className="legend-item">
                <span className="legend-dot" style={{ background: color }} />
                <span className="legend-label">{level}</span>
                <span className="legend-count">{stats[level] || 0}</span>
              </div>
            ))}
          </div>

          <label className="toggle-row">
            <input
              type="checkbox"
              checked={showAttention}
              onChange={(e) => {
                setShowAttention(e.target.checked);
                if (!e.target.checked) {
                  attentionLines.forEach((l) => l.remove());
                  setAttentionLines([]);
                }
              }}
            />
            Show attention edges
          </label>

          {selectedSensor && (
            <div className="sensor-panel">
              <h4>Sensor {selectedSensor.id}</h4>
              <p className="sensor-speed">
                Current: <strong>{selectedSensor.current_speed} mph</strong>
              </p>
              <div
                className="congestion-pill"
                style={{ background: CONGESTION_COLORS[selectedSensor.congestion] }}
              >
                {selectedSensor.congestion}
              </div>

              {forecast && (
                <div className="mini-forecast">
                  <h5>1-hour forecast</h5>
                  <div className="forecast-bars">
                    {forecast.predicted_speeds.map((s, i) => (
                      <div key={i} className="forecast-bar-wrap" title={`+${(i + 1) * 5}m: ${s} mph`}>
                        <div
                          className="forecast-bar"
                          style={{
                            height: `${(s / 80) * 60}px`,
                            background: speedToColor(s),
                          }}
                        />
                        {(i + 1) % 3 === 0 && (
                          <span className="bar-label">+{(i + 1) * 5}m</span>
                        )}
                      </div>
                    ))}
                  </div>
                </div>
              )}
            </div>
          )}
        </div>

        {/* Map container */}
        <div ref={mapRef} className="leaflet-container" />
      </div>
    </div>
  );
}

function speedToColor(speed) {
  if (speed > 60) return "#16a34a";
  if (speed > 40) return "#ca8a04";
  if (speed > 20) return "#dc2626";
  return "#7c3aed";
}
