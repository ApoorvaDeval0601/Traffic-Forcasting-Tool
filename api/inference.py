"""
Inference engine: loads trained model and runs predictions.
Handles normalization, edge index setup, and result formatting.
"""

import torch
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional

from models.stgnn import STGNN, build_model
from api.schemas import ForecastResponse, SensorForecast, SensorInfo, AttentionResponse


CONGESTION_THRESHOLDS = {
    "free": 60,      # > 60 mph
    "moderate": 40,  # 40-60
    "heavy": 20,     # 20-40
    "severe": 0,     # < 20
}


def speed_to_congestion(speed_mph: float) -> str:
    if speed_mph > 60:
        return "free"
    elif speed_mph > 40:
        return "moderate"
    elif speed_mph > 20:
        return "heavy"
    return "severe"


class TrafficInference:
    def __init__(self, model_path: str, data_dir: str, dataset: str = "metr-la"):
        self.dataset = dataset
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        processed_dir = Path(data_dir) / "processed" / dataset
        graph_dir = Path(data_dir) / "graphs"

        # Load normalization stats
        self.mean = np.load(processed_dir / "mean.npy")
        self.std = np.load(processed_dir / "std.npy")

        # Load graph
        edge_index = np.load(graph_dir / f"{dataset}_edge_index.npy")
        self.edge_index = torch.from_numpy(edge_index).long().to(self.device)

        # Load model
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
        cfg = checkpoint["config"]
        self.model = build_model(cfg["model"])
        self.model.load_state_dict(checkpoint["model_state"])
        self.model.eval().to(self.device)

        # Load sensor metadata (lat/lng)
        meta_path = Path(data_dir) / "raw" / f"sensor_locations_{dataset}.csv"
        self._load_sensor_metadata(meta_path)

        # Cache for latest window
        self._latest_window: Optional[np.ndarray] = None
        self._load_latest_test_window(processed_dir)

    def _load_sensor_metadata(self, meta_path: Path):
        import pandas as pd
        if meta_path.exists():
            df = pd.read_csv(meta_path)
            self._sensor_meta = df.to_dict("records")
        else:
            # Fallback: LA bounding box approximate
            n = self.edge_index.max().item() + 1
            self._sensor_meta = [
                {"sensor_id": i, "lat": 34.0 + i * 0.001, "lng": -118.2 + i * 0.001,
                 "name": f"Sensor {i}", "road_type": "unknown"}
                for i in range(n)
            ]

    def _load_latest_test_window(self, processed_dir: Path):
        test_X = np.load(processed_dir / "test_X.npy")
        self._latest_window = test_X[-1]  # (N, T, F) — last test sample

    def _normalize(self, x: np.ndarray) -> np.ndarray:
        return (x - self.mean) / (self.std + 1e-8)

    def _denormalize(self, x: np.ndarray) -> np.ndarray:
        return x * self.std + self.mean

    @torch.no_grad()
    def forecast(
        self,
        input_window: Optional[list] = None,
        sensor_ids: Optional[list[int]] = None,
        return_attention: bool = False,
    ) -> ForecastResponse:
        if input_window is not None:
            x = np.array(input_window, dtype=np.float32)  # (N, T, F)
            x = self._normalize(x)
        else:
            x = self._latest_window  # (N, T, F)

        x_tensor = torch.from_numpy(x).float().unsqueeze(0).to(self.device)  # (1, N, T, F)

        if return_attention:
            pred, spatial_attn = self.model(x_tensor, self.edge_index, return_attention=True)
        else:
            pred = self.model(x_tensor, self.edge_index)
            spatial_attn = None

        pred_np = self._denormalize(pred.squeeze(0).cpu().numpy())  # (N, H, 1)
        pred_np = pred_np.squeeze(-1)  # (N, H)

        # Build response
        now = datetime.utcnow()
        timestamps = [(now + timedelta(minutes=5 * (h + 1))).isoformat() for h in range(pred_np.shape[1])]

        sensors = []
        indices = range(len(self._sensor_meta))
        if sensor_ids is not None:
            indices = sensor_ids

        for i in indices:
            meta = self._sensor_meta[i]
            speeds = pred_np[i].tolist()
            current_speed = self._denormalize(x[i, -1, 0]).item()
            sensors.append(SensorForecast(
                sensor_id=meta["sensor_id"],
                lat=meta["lat"],
                lng=meta["lng"],
                name=meta.get("name"),
                current_speed=round(current_speed, 1),
                predicted_speeds=[round(s, 1) for s in speeds],
                timestamps=timestamps,
                congestion_level=speed_to_congestion(current_speed),
            ))

        attn_dict = None
        if spatial_attn and return_attention:
            # First layer attention averaged over heads
            attn = spatial_attn[0].mean(dim=-1).cpu().numpy()  # (E,)
            edge_idx = self.edge_index.cpu().numpy()
            attn_dict = {}
            for e_idx in range(edge_idx.shape[1]):
                src, dst = int(edge_idx[0, e_idx]), int(edge_idx[1, e_idx])
                attn_dict.setdefault(str(src), {})[str(dst)] = float(attn[e_idx])

        return ForecastResponse(
            sensors=sensors,
            dataset=self.dataset,
            attention_weights=attn_dict,
        )

    def get_sensor_metadata(self) -> list[SensorInfo]:
        return [SensorInfo(**{k: v for k, v in m.items() if k in SensorInfo.model_fields})
                for m in self._sensor_meta]

    def get_attention_weights(self, sensor_id: int, top_k: int = 10) -> Optional[AttentionResponse]:
        result = self.forecast(return_attention=True)
        if result.attention_weights is None:
            return None
        neighbors_raw = result.attention_weights.get(str(sensor_id), {})
        sorted_neighbors = sorted(neighbors_raw.items(), key=lambda x: -x[1])[:top_k]
        neighbors = [
            {"sensor_id": int(nid), "weight": round(w, 4),
             "lat": self._sensor_meta[int(nid)]["lat"],
             "lng": self._sensor_meta[int(nid)]["lng"]}
            for nid, w in sorted_neighbors
        ]
        return AttentionResponse(source_sensor=sensor_id, neighbors=neighbors, top_k=top_k)

    def get_congestion_snapshot(self) -> dict:
        result = self.forecast()
        return {
            str(s.sensor_id): {
                "lat": s.lat,
                "lng": s.lng,
                "current_speed": s.current_speed,
                "congestion": s.congestion_level,
                "next_15min": s.predicted_speeds[2] if len(s.predicted_speeds) > 2 else None,
            }
            for s in result.sensors
        }

    def get_live_forecast(self) -> dict:
        return self.get_congestion_snapshot()
