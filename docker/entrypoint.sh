#!/bin/bash
set -e

echo "[*] Initializing FRADSCR Production Container..."
echo "[*] Binding Port: ${PORT:-3000}"
echo "[*] ML Service URL: ${ML_SERVICE_URL:-http://127.0.0.1:8000}"
echo "[*] Calibration Temperature: ${CALIBRATION_TEMPERATURE:-0.35}"

# Launch supervisord to manage both FastAPI ML Service and Express Web Gateway
exec /usr/bin/supervisord -c /etc/supervisor/conf.d/supervisord.conf
