"""
Cached API for ST-GNN Traffic Forecasting.
Serves pre-computed predictions from JSON cache.
Zero PyTorch/ML dependencies — runs on 512MB RAM.
"""

import os
import json
import asyncio
from pathlib import Path
from datetime import datetime, timedelta
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

load_dotenv()


def speed_to_congestion(s: float) -> str:
    if s > 60:  return "free"
    if s > 40:  return "moderate"
    if s > 20:  return "heavy"
    return "severe"


class CachedTrafficSimulator:
    """Serves predictions from pre-computed JSON cache. No ML needed."""

    def __init__(self, cache_path: str):
        print(f"Loading cache from {cache_path}...")
        with open(cache_path) as f:
            cache = json.load(f)

        self.total    = cache["total_frames"]
        self.coords   = cache["coords"]
        self.edges    = cache["edges"]
        self.frames   = cache["frames"]
        self.mean     = cache["mean"]
        self.std      = cache["std"]
        self.idx      = 0

        print(f"Cache loaded: {self.total} frames, {len(self.coords)} sensors")

    def get_frame(self) -> dict:
        idx = self.idx % self.total
        self.idx += 1

        sensors_raw = self.frames[idx]
        now = datetime.utcnow()
        ts  = [(now + timedelta(minutes=5*(h+1))).strftime("%H:%M") for h in range(12)]

        sensors = []
        for s in sensors_raw:
            cs = s["cs"]
            sensors.append({
                "id":               s["id"],
                "lat":              s["lat"],
                "lng":              s["lng"],
                "current_speed":    cs,
                "congestion":       speed_to_congestion(cs),
                "predicted_speeds": s["pr"],
                "actual_speeds":    s["ac"],
                "timestamps":       ts,
            })

        counts = {}
        for s in sensors:
            c = s["congestion"]
            counts[c] = counts.get(c, 0) + 1

        return {
            "frame_idx":         idx,
            "total_frames":      self.total,
            "timestamp":         now.isoformat(),
            "sensors":           sensors,
            "congestion_counts": counts,
        }

    def get_road_edges(self) -> dict:
        return {"edges": self.edges, "coords": self.coords}

    def get_benchmark(self) -> dict:
        p = Path("evaluation/benchmark_results.json")
        if p.exists():
            with open(p) as f:
                return json.load(f)
        return {"error": "No benchmark results found"}


simulator: Optional[CachedTrafficSimulator] = None


def find_cache():
    """Look for cache file in multiple locations."""
    candidates = [
        os.getenv("CACHE_PATH", ""),
        "data/predictions_cache.json",
        "/tmp/predictions_cache.json",
    ]
    for p in candidates:
        if p and Path(p).exists():
            return p
    return None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global simulator

    cache_path = find_cache()

    if cache_path:
        print(f"Found cache at {cache_path}")
        simulator = CachedTrafficSimulator(cache_path)
        print("Ready to serve predictions")
    else:
        # Try downloading from HuggingFace
        hf_token = os.getenv("HF_TOKEN", "")
        repo_id  = os.getenv("HF_REPO", "Apoorva06/stgnn-traffic")
        print(f"Cache not found locally. Downloading from HuggingFace: {repo_id}")
        try:
            from huggingface_hub import hf_hub_download
            path = hf_hub_download(
                repo_id=repo_id,
                filename="data/predictions_cache.json",
                repo_type="model",
                token=hf_token or None,
                local_dir="/tmp",
            )
            simulator = CachedTrafficSimulator(path)
            print("Downloaded and loaded cache from HuggingFace")
        except Exception as e:
            print(f"Failed to load cache: {e}")
            print("API will start but return errors until cache is available")

    yield


app = FastAPI(
    title="ST-GNN Traffic Forecasting API (Cached)",
    version="3.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health():
    return {
        "status":       "ok",
        "model_loaded": simulator is not None,
        "mode":         "cached",
        "total_frames": simulator.total if simulator else 0,
    }


@app.get("/frame")
def get_frame():
    if not simulator:
        return {"error": "Cache not loaded yet"}
    return simulator.get_frame()


@app.get("/forecast/{sensor_id}")
def get_forecast(sensor_id: int):
    if not simulator:
        return {"error": "Cache not loaded"}
    frame = simulator.get_frame()
    s = next((s for s in frame["sensors"] if s["id"] == sensor_id), None)
    return s or {"error": "Sensor not found"}


@app.get("/road-edges")
def get_road_edges():
    if not simulator:
        return {"error": "Cache not loaded"}
    return simulator.get_road_edges()


@app.get("/benchmark")
def get_benchmark():
    if not simulator:
        return {"error": "Cache not loaded"}
    return simulator.get_benchmark()


@app.get("/attention/{sensor_id}")
def get_attention(sensor_id: int, top_k: int = 10):
    """
    Returns pre-defined attention weights based on graph adjacency.
    (Real GAT attention not available in cached mode — uses edge weights instead)
    """
    if not simulator:
        return {"error": "Cache not loaded"}

    edges   = simulator.edges
    coords  = simulator.coords
    weights = {}

    for src, dst in edges:
        if src == sensor_id:
            dlat = coords[src][0] - coords[dst][0]
            dlng = coords[src][1] - coords[dst][1]
            dist = (dlat**2 + dlng**2) ** 0.5
            if dist > 0:
                weights[dst] = round(1 / dist, 4)

    top = sorted(weights.items(), key=lambda x: -x[1])[:top_k]
    # Normalize weights
    if top:
        max_w = top[0][1]
        top   = [(nid, round(w / max_w, 4)) for nid, w in top]

    neighbors = [
        {
            "sensor_id": nid,
            "weight":    w,
            "lat":       coords[nid][0],
            "lng":       coords[nid][1],
        }
        for nid, w in top
    ]

    return {"source_sensor": sensor_id, "neighbors": neighbors, "top_k": top_k}


@app.websocket("/ws/live")
async def ws_live(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            if simulator:
                f = simulator.get_frame()
                await websocket.send_json({
                    "type":              "frame",
                    "frame_idx":         f["frame_idx"],
                    "congestion_counts": f["congestion_counts"],
                    "sensors": [
                        {
                            "id":            s["id"],
                            "lat":           s["lat"],
                            "lng":           s["lng"],
                            "current_speed": s["current_speed"],
                            "congestion":    s["congestion"],
                        }
                        for s in f["sensors"]
                    ],
                })
            await asyncio.sleep(5)
    except WebSocketDisconnect:
        pass


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)