#!/bin/bash
# startup_cached.sh — Downloads cache from HuggingFace then starts lightweight API

set -e

echo "Installing huggingface_hub..."
pip install huggingface_hub --quiet

echo "Downloading predictions cache from HuggingFace..."
python -c "
from huggingface_hub import hf_hub_download
import os

path = hf_hub_download(
    repo_id='Apoorva06/stgnn-traffic',
    filename='data/predictions_cache_small.json',
    repo_type='model',
    token=os.getenv('HF_TOKEN') or None,
    local_dir='/tmp',
)
print(f'Cache downloaded to {path}')
"

echo "Starting cached API..."
export CACHE_PATH=/tmp/data/predictions_cache_small.json
exec uvicorn api.main:app --host 0.0.0.0 --port $PORT