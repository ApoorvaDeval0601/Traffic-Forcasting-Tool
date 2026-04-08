"""Pydantic models for API request/response validation."""

from pydantic import BaseModel, Field
from typing import Optional


class ForecastRequest(BaseModel):
    # Optional: if None, uses most recent data from cache
    input_window: Optional[list[list[list[float]]]] = Field(
        None,
        description="Input shape: (N_sensors, T_steps, features). If None, uses cached latest."
    )
    sensor_ids: Optional[list[int]] = Field(
        None,
        description="Filter to specific sensor IDs. If None, returns all."
    )
    return_attention: bool = Field(False, description="Include GAT attention weights in response")

    model_config = {"json_schema_extra": {"example": {"sensor_ids": [0, 1, 2], "return_attention": False}}}


class SensorForecast(BaseModel):
    sensor_id: int
    lat: float
    lng: float
    name: Optional[str]
    current_speed: float            # mph, most recent observation
    predicted_speeds: list[float]   # 12-step forecast (mph)
    timestamps: list[str]           # ISO timestamps for each forecast step
    congestion_level: str           # "free", "moderate", "heavy", "severe"


class ForecastResponse(BaseModel):
    sensors: list[SensorForecast]
    model: str = "ST-GNN (Transformer)"
    dataset: str
    horizon_minutes: int = 60
    attention_weights: Optional[dict] = None  # sensor_id → {neighbor_id: weight}


class SensorInfo(BaseModel):
    sensor_id: int
    lat: float
    lng: float
    name: Optional[str]
    road_type: Optional[str]    # highway, arterial, etc.


class AttentionResponse(BaseModel):
    source_sensor: int
    neighbors: list[dict]       # [{sensor_id, weight, lat, lng}]
    top_k: int
