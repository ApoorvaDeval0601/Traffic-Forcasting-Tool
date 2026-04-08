"""
FastAPI backend for ST-GNN traffic forecasting.

Endpoints:
  GET  /health                     — Health check
  POST /forecast                   — Single forecast request
  GET  /sensors                    — Sensor metadata
  GET  /attention/{sensor_id}      — GAT attention weights for a sensor
  WS   /ws/live                    — Live forecast WebSocket
"""

import os
import json
import asyncio
import numpy as np
from pathlib import Path
from typing import Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from api.schemas import (
    ForecastRequest,
    ForecastResponse,
    SensorInfo,
    AttentionResponse,
)
from api.inference import TrafficInference


# ─── App lifecycle ──────────────────────────────────────────────────────────

inference: Optional[TrafficInference] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global inference
    model_path = os.getenv("MODEL_PATH", "checkpoints/stgnn_metrla/best_model.pt")
    data_dir = os.getenv("DATA_DIR", "data")
    dataset = os.getenv("DATASET", "metr-la")

    print(f"Loading model from {model_path}...")
    inference = TrafficInference(
        model_path=model_path,
        data_dir=data_dir,
        dataset=dataset,
    )
    print("✓ Model loaded")
    yield
    print("Shutting down...")


app = FastAPI(
    title="ST-GNN Traffic Forecasting API",
    description="Spatio-Temporal GNN for traffic speed prediction",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],       # Tighten in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─── REST Endpoints ──────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    return {"status": "ok", "model_loaded": inference is not None}


@app.get("/sensors", response_model=list[SensorInfo])
async def get_sensors():
    """Return all sensor metadata (id, lat, lng, name)."""
    if inference is None:
        raise HTTPException(503, "Model not loaded")
    return inference.get_sensor_metadata()


@app.post("/forecast", response_model=ForecastResponse)
async def forecast(request: ForecastRequest):
    """
    Run a forecast from the provided input window.

    Input window: last 12 time steps per sensor (5-min intervals).
    Returns: 12-step (1 hour) forecast per sensor.
    """
    if inference is None:
        raise HTTPException(503, "Model not loaded")

    result = inference.forecast(
        input_window=request.input_window,
        sensor_ids=request.sensor_ids,
        return_attention=request.return_attention,
    )
    return result


@app.get("/attention/{sensor_id}", response_model=AttentionResponse)
async def get_attention(sensor_id: int, top_k: int = 10):
    """Return top-K most attended neighboring sensors for a given sensor."""
    if inference is None:
        raise HTTPException(503, "Model not loaded")

    attn = inference.get_attention_weights(sensor_id, top_k=top_k)
    if attn is None:
        raise HTTPException(404, f"Sensor {sensor_id} not found")
    return attn


@app.get("/congestion-map")
async def congestion_map():
    """Current predicted congestion level per sensor (for map overlay)."""
    if inference is None:
        raise HTTPException(503, "Model not loaded")
    return inference.get_congestion_snapshot()


@app.get("/model-comparison")
async def model_comparison():
    """Return LSTM vs Transformer benchmark metrics."""
    results_path = Path("evaluation/benchmark_results.json")
    if not results_path.exists():
        raise HTTPException(404, "Run benchmark.py first to generate comparison")
    with open(results_path) as f:
        return json.load(f)


# ─── WebSocket ───────────────────────────────────────────────────────────────

class ConnectionManager:
    def __init__(self):
        self.active: list[WebSocket] = []

    async def connect(self, ws: WebSocket):
        await ws.accept()
        self.active.append(ws)

    def disconnect(self, ws: WebSocket):
        self.active.remove(ws)

    async def broadcast(self, data: dict):
        for ws in self.active:
            try:
                await ws.send_json(data)
            except Exception:
                pass


manager = ConnectionManager()


@app.websocket("/ws/live")
async def live_forecast(websocket: WebSocket):
    """
    Streams live forecast updates every 30 seconds.
    Sends: { type: "forecast", data: { sensor_id: { speeds: [...], timestamps: [...] } } }
    """
    await manager.connect(websocket)
    try:
        while True:
            if inference is not None:
                snapshot = inference.get_live_forecast()
                await websocket.send_json({"type": "forecast", "data": snapshot})
            await asyncio.sleep(30)
    except WebSocketDisconnect:
        manager.disconnect(websocket)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
