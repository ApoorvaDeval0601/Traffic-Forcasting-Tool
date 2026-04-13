#!/bin/bash
# startup.sh — Downloads model and data from HuggingFace then starts API
# Runs on Render before the main process starts

set -e

REPO="Apoorva06/stgnn-traffic"
echo "Downloading model and data from HuggingFace: $REPO"

pip install huggingface_hub --quiet

python -c "
from huggingface_hub import snapshot_download
import os, shutil

local = snapshot_download(
    repo_id='Apoorva06/stgnn-traffic',
    repo_type='model',
    local_dir='/tmp/hf_download',
    ignore_patterns=['*.md', '.gitattributes'],
)
print(f'Downloaded to {local}')

# Copy checkpoints
if os.path.exists('/tmp/hf_download/checkpoints'):
    if os.path.exists('checkpoints'):
        shutil.rmtree('checkpoints')
    shutil.copytree('/tmp/hf_download/checkpoints', 'checkpoints')
    print('Checkpoints ready')

# Copy data
if os.path.exists('/tmp/hf_download/data'):
    if os.path.exists('data'):
        shutil.rmtree('data')
    shutil.copytree('/tmp/hf_download/data', 'data')
    print('Data ready')

print('All files ready')
"

echo "Starting API..."
exec uvicorn api.main:app --host 0.0.0.0 --port $PORT