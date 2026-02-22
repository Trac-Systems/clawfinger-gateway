#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
GW_DIR="$(dirname "$SCRIPT_DIR")"
VENV_DIR="$GW_DIR/.venv"
PID_DIR="$GW_DIR/tmp"
mkdir -p "$PID_DIR"

# Keep all model downloads inside gateway/.models/
export HF_HOME="$GW_DIR/.models"
mkdir -p "$HF_HOME"

MLX_AUDIO_HOST="${MLX_AUDIO_HOST:-127.0.0.1}"
MLX_AUDIO_PORT="${MLX_AUDIO_PORT:-8765}"
GW_HOST="${GW_HOST:-127.0.0.1}"
GW_PORT="${GW_PORT:-8996}"

cleanup() {
  echo "[start.sh] Stopping..."
  if [ -f "$PID_DIR/mlx_audio.pid" ]; then
    kill "$(cat "$PID_DIR/mlx_audio.pid")" 2>/dev/null || true
    rm -f "$PID_DIR/mlx_audio.pid"
  fi
  if [ -f "$PID_DIR/gateway.pid" ]; then
    kill "$(cat "$PID_DIR/gateway.pid")" 2>/dev/null || true
    rm -f "$PID_DIR/gateway.pid"
  fi
}
trap cleanup EXIT INT TERM

# Activate venv
if [ -d "$VENV_DIR" ]; then
  source "$VENV_DIR/bin/activate"
else
  echo "[start.sh] No .venv found at $VENV_DIR — using system Python"
fi

# Start mlx_audio server
echo "[start.sh] Starting mlx_audio on $MLX_AUDIO_HOST:$MLX_AUDIO_PORT..."
python -m mlx_audio.server --host "$MLX_AUDIO_HOST" --port "$MLX_AUDIO_PORT" &
MLX_PID=$!
echo "$MLX_PID" > "$PID_DIR/mlx_audio.pid"

# Wait for mlx_audio to be ready
echo "[start.sh] Waiting for mlx_audio..."
for i in $(seq 1 60); do
  if curl -sf "http://$MLX_AUDIO_HOST:$MLX_AUDIO_PORT/v1/models" > /dev/null 2>&1; then
    echo "[start.sh] mlx_audio ready"
    break
  fi
  if ! kill -0 "$MLX_PID" 2>/dev/null; then
    echo "[start.sh] mlx_audio process died"
    exit 1
  fi
  sleep 2
done

# Start gateway
echo "[start.sh] Starting gateway on $GW_HOST:$GW_PORT..."
cd "$GW_DIR"
python -m uvicorn app:app --host "$GW_HOST" --port "$GW_PORT" --log-level info &
GW_PID=$!
echo "$GW_PID" > "$PID_DIR/gateway.pid"

echo "[start.sh] Gateway PID=$GW_PID, mlx_audio PID=$MLX_PID"
echo "[start.sh] Control center: http://$GW_HOST:$GW_PORT/"

# Wait for either to exit
wait -n "$MLX_PID" "$GW_PID" 2>/dev/null || true
