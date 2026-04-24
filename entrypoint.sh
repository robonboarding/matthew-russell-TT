#!/bin/sh
set -e

# Build the FAISS index on first startup if it does not already exist.
# This keeps secrets out of the image: Azure credentials are injected as
# environment variables at container runtime, never baked into layers.
if [ ! -f /app/data/processed/faiss.index ]; then
    echo "[entrypoint] Index not found, building from data/raw ..."
    python -m src.ingest data/raw
    python -m src.index
    echo "[entrypoint] Index built."
else
    echo "[entrypoint] Index found, skipping build."
fi

exec uvicorn src.api:app --host 0.0.0.0 --port 8000
