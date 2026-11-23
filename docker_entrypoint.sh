#!/usr/bin/env bash
set -e

if [ "${SKIP_WEIGHT_DOWNLOAD:-0}" != "1" ] && [ "${SKIP_WEIGHT_DOWNLOAD:-false}" != "true" ]; then
    echo "[Entrypoint] Ensuring model weights from weights_manifest.yaml..."
    python setup_env.py
fi

MODE="${SERVER_MODE:-api}"
PORT_NUM="${PORT:-7860}"

if [ "$MODE" = "streamlit" ]; then
    exec streamlit run app.py --server.port="${PORT_NUM}" --server.address=0.0.0.0
else
    exec uvicorn api_server:app --host 0.0.0.0 --port "${PORT_NUM}"
fi
