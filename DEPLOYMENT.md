# Deployment Guide

## Local Development

```bash
# 1. Train models first
python scripts/download_data.py --dataset metr-la
python scripts/build_graph.py --dataset metr-la
python scripts/train.py --model stgnn --dataset metr-la
python scripts/train.py --model lstm --dataset metr-la   # Baseline

# 2. Run benchmark
python evaluation/benchmark.py \
  --transformer-ckpt checkpoints/stgnn_metrla/best_model.pt \
  --lstm-ckpt checkpoints/lstm_metrla/best_model.pt

# 3. Start API
cd stgnn-traffic
uvicorn api.main:app --reload --port 8000

# 4. Start frontend (new terminal)
cd frontend
npm install
npm run dev  # → http://localhost:3000
```

---

## Docker (recommended)

```bash
docker-compose up --build
# API:      http://localhost:8000/docs
# Frontend: http://localhost:3000
```

---

## Deploy to Render (Free Tier)

### Option A: Blueprint (easiest)
1. Push repo to GitHub
2. Go to [render.com](https://render.com) → New → Blueprint
3. Connect your GitHub repo
4. Render reads `render.yaml` and creates both services automatically

### Option B: Manual

**Backend (Web Service)**
- Runtime: Python 3
- Build Command: `pip install -r requirements.txt`
- Start Command: `uvicorn api.main:app --host 0.0.0.0 --port $PORT`
- Environment variables:
  - `MODEL_PATH` = `checkpoints/stgnn_metrla/best_model.pt`
  - `DATASET` = `metr-la`

**Frontend (Static Site)**
- Build Command: `cd frontend && npm ci && npm run build`
- Publish Directory: `frontend/dist`
- Environment variable:
  - `VITE_API_URL` = `https://your-api-service.onrender.com`

### ⚠️ Free tier notes
- API service **spins down after 15 min** of inactivity (cold start ~30s)
- No persistent disk on free tier → serve test-set predictions
- For always-on: upgrade to Starter ($7/mo)

---

## Environment variables

| Variable | Default | Description |
|---|---|---|
| `MODEL_PATH` | `checkpoints/stgnn_metrla/best_model.pt` | Trained model checkpoint |
| `DATA_DIR` | `data` | Data directory |
| `DATASET` | `metr-la` | Dataset name |
| `VITE_API_URL` | `http://localhost:8000` | Frontend API endpoint |
