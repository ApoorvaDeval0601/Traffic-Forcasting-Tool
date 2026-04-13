"""
ST-GNN Traffic Forecasting API — Demo version
Cycles through METR-LA test set to simulate live traffic
Uses real sensor GPS coordinates
"""

import os, sys, json, asyncio
import numpy as np
import torch
from pathlib import Path
from datetime import datetime, timedelta
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

load_dotenv()
sys.path.insert(0, str(Path(__file__).parent.parent))
from models.stgnn import build_model

def speed_to_congestion(s):
    if s > 60: return "free"
    if s > 40: return "moderate"
    if s > 20: return "heavy"
    return "severe"

def load_sensor_coords(data_dir):
    """Load real METR-LA GPS coordinates."""
    p = Path(data_dir) / "raw" / "sensor_coords.json"
    if p.exists():
        with open(p) as f:
            coords = json.load(f)
        return [(c[0], c[1]) for c in coords[:207]]

    # Try loading from CSV
    csv_p = Path(data_dir) / "raw" / "graph_sensor_locations.csv"
    if csv_p.exists():
        import pandas as pd
        df = pd.read_csv(csv_p, header=None, skiprows=1)
        df.columns = ["index", "sensor_id", "lat", "lng"]
        df = df.sort_values("index").reset_index(drop=True)
        coords = [(float(r.lat), float(r.lng)) for _, r in df.iterrows()][:207]
        # Save for next time
        with open(p, "w") as f:
            json.dump(coords, f)
        return coords

    # Fallback: approximate LA coords
    coords = []
    for i in range(207):
        lat = 34.05 + (i % 23) * 0.009
        lng = -118.45 + (i // 23) * 0.013
        coords.append((lat, lng))
    return coords


class TrafficSimulator:
    def __init__(self, model_path, data_dir, dataset="metr-la"):
        self.device  = "cuda" if torch.cuda.is_available() else "cpu"
        self.dataset = dataset
        self.mean    = float(np.load(Path(data_dir) / "processed" / dataset / "mean.npy"))
        self.std     = float(np.load(Path(data_dir) / "processed" / dataset / "std.npy"))
        self.test_X  = np.load(Path(data_dir) / "processed" / dataset / "test_X.npy")
        self.test_Y  = np.load(Path(data_dir) / "processed" / dataset / "test_Y.npy")
        edge_index   = np.load(Path(data_dir) / "graphs" / f"{dataset}_edge_index.npy")
        self.edge_index = torch.from_numpy(edge_index).long().to(self.device)
        self.coords  = load_sensor_coords(data_dir)

        ckpt = torch.load(model_path, map_location=self.device, weights_only=False)
        self.model = build_model(ckpt["config"]["model"])
        self.model.load_state_dict(ckpt["model_state"])
        self.model.eval().to(self.device)

        self.idx   = 0
        self.total = len(self.test_X)
        print(f"Simulator ready: {self.total} frames, device={self.device}")
        print(f"Loaded {len(self.coords)} real sensor coordinates")

    def dn(self, x): return x * self.std + self.mean

    @torch.no_grad()
    def get_frame(self):
        idx = self.idx % self.total
        self.idx += 1
        x = self.test_X[idx]
        y = self.test_Y[idx]
        xt = torch.from_numpy(x).float().unsqueeze(0).to(self.device)
        pred = self.model(xt, self.edge_index).squeeze(0).cpu().numpy()

        current  = self.dn(x[:, -1, 0])
        predicted = self.dn(pred[:, :, 0])
        actual   = self.dn(y[:, :, 0])
        now      = datetime.utcnow()
        ts       = [(now + timedelta(minutes=5*(h+1))).strftime("%H:%M") for h in range(12)]

        sensors = []
        for i in range(min(207, len(self.coords))):
            lat, lng = self.coords[i]
            cs = float(current[i])
            sensors.append({
                "id":               i,
                "lat":              lat,
                "lng":              lng,
                "current_speed":    round(max(0, cs), 1),
                "congestion":       speed_to_congestion(cs),
                "predicted_speeds": [round(float(v), 1) for v in predicted[i]],
                "actual_speeds":    [round(float(v), 1) for v in actual[i]],
                "timestamps":       ts,
            })

        return {
            "frame_idx":    idx,
            "total_frames": self.total,
            "timestamp":    now.isoformat(),
            "sensors":      sensors,
            "congestion_counts": {
                "free":     sum(1 for s in sensors if s["congestion"] == "free"),
                "moderate": sum(1 for s in sensors if s["congestion"] == "moderate"),
                "heavy":    sum(1 for s in sensors if s["congestion"] == "heavy"),
                "severe":   sum(1 for s in sensors if s["congestion"] == "severe"),
            }
        }


simulator: Optional[TrafficSimulator] = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global simulator
    model_path = os.getenv("MODEL_PATH", "checkpoints/stgnn_metr_la/best_model.pt")
    data_dir   = os.getenv("DATA_DIR",   "data")
    dataset    = os.getenv("DATASET",    "metr-la")
    print(f"Loading model from {model_path}...")
    simulator = TrafficSimulator(model_path, data_dir, dataset)
    print("Model loaded and ready")
    yield

app = FastAPI(title="ST-GNN Traffic API", version="2.0.0", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": simulator is not None}

@app.get("/frame")
def get_frame():
    if not simulator: return {"error": "Model not loaded"}
    return simulator.get_frame()

@app.get("/forecast/{sensor_id}")
def get_forecast(sensor_id: int):
    if not simulator: return {"error": "Model not loaded"}
    frame = simulator.get_frame()
    s = next((s for s in frame["sensors"] if s["id"] == sensor_id), None)
    return s or {"error": "Sensor not found"}

@app.get("/benchmark")
def get_benchmark():
    p = Path("evaluation/benchmark_results.json")
    if not p.exists(): return {"error": "Run benchmark.py first"}
    with open(p) as f: return json.load(f)

@app.get("/attention/{sensor_id}")
def get_attention(sensor_id: int, top_k: int = 10):
    if not simulator: return {"error": "Model not loaded"}
    try:
        x  = simulator.test_X[0]
        xt = torch.from_numpy(x).float().unsqueeze(0).to(simulator.device)
        with torch.no_grad():
            _, attn_list = simulator.model(xt, simulator.edge_index, return_attention=True)
        attn = attn_list[0].mean(dim=-1).cpu().numpy()
        ei   = simulator.edge_index.cpu().numpy()
        nbrs = {}
        for e in range(ei.shape[1]):
            src, dst = int(ei[0, e]), int(ei[1, e])
            if src == sensor_id:
                nbrs[dst] = float(attn[e])
        top = sorted(nbrs.items(), key=lambda x: -x[1])[:top_k]
        result = [{"sensor_id": nid, "weight": round(w,4),
                   "lat": simulator.coords[nid][0], "lng": simulator.coords[nid][1]}
                  for nid, w in top]
        return {"source_sensor": sensor_id, "neighbors": result, "top_k": top_k}
    except Exception as e:
        return {"error": str(e)}

@app.websocket("/ws/live")
async def ws_live(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            if simulator:
                f = simulator.get_frame()
                await websocket.send_json({
                    "type": "frame",
                    "frame_idx": f["frame_idx"],
                    "congestion_counts": f["congestion_counts"],
                    "sensors": [{"id":s["id"],"lat":s["lat"],"lng":s["lng"],
                                 "current_speed":s["current_speed"],"congestion":s["congestion"]}
                                for s in f["sensors"]]
                })
            await asyncio.sleep(5)
    except WebSocketDisconnect:
        pass


@app.get("/road-edges")
def get_road_edges():
    import json
    p = Path("data/raw/road_edges.json")
    if not p.exists():
        return {"error": "Run the road edges script first"}
    with open(p) as f:
        return json.load(f)


@app.get("/road-edges")
def get_road_edges():
    import json
    p = Path("data/raw/road_edges.json")
    if not p.exists():
        return {"error": "Run the road edges script first"}
    with open(p) as f:
        return json.load(f)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)