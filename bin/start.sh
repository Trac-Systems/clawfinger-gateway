#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
GW_DIR="$(dirname "$SCRIPT_DIR")"
VENV_DIR="$GW_DIR/.venv"
PID_DIR="$GW_DIR/tmp"
mkdir -p "$PID_DIR"

# Ensure Homebrew tools (ffmpeg etc.) are in PATH for non-interactive shells
if [ -d "/opt/homebrew/bin" ]; then
  export PATH="/opt/homebrew/bin:$PATH"
fi

# Keep all model downloads inside gateway/.models/
export HF_HOME="$GW_DIR/.models"
mkdir -p "$HF_HOME"

MLX_AUDIO_HOST="${MLX_AUDIO_HOST:-127.0.0.1}"
MLX_AUDIO_PORT="${MLX_AUDIO_PORT:-8765}"
PIPER_PORT="${PIPER_PORT:-5123}"
PIPER_MODEL="${PIPER_MODEL:-$GW_DIR/voices/de_DE-thorsten-high.onnx}"
GW_HOST="${GW_HOST:-127.0.0.1}"
GW_PORT="${GW_PORT:-8996}"

cleanup() {
  echo "[start.sh] Stopping..."
  for pidfile in "$PID_DIR/mlx_audio.pid" "$PID_DIR/piper.pid" "$PID_DIR/gateway.pid"; do
    if [ -f "$pidfile" ]; then
      kill "$(cat "$pidfile")" 2>/dev/null || true
      rm -f "$pidfile"
    fi
  done
}
# Only cleanup on explicit stop signals, NOT on EXIT (which fires on
# SIGHUP when SSH disconnects and kills background children).
trap cleanup INT TERM
# Ignore HUP so SSH disconnect doesn't kill us
trap '' HUP

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

# Start Piper TTS (German) if model file exists
PIPER_PID=""
if [ -f "$PIPER_MODEL" ]; then
  echo "[start.sh] Starting Piper TTS on $MLX_AUDIO_HOST:$PIPER_PORT..."
  python -m piper.http_server --model "$PIPER_MODEL" --host "$MLX_AUDIO_HOST" --port "$PIPER_PORT" &
  PIPER_PID=$!
  echo "$PIPER_PID" > "$PID_DIR/piper.pid"
  for i in $(seq 1 30); do
    if curl -sf -X POST -H "Content-Type: application/json" -d '{"text":"test"}' "http://$MLX_AUDIO_HOST:$PIPER_PORT/" -o /dev/null 2>&1; then
      echo "[start.sh] Piper ready"
      break
    fi
    sleep 1
  done
else
  echo "[start.sh] Piper model not found at $PIPER_MODEL — skipping German TTS"
fi

# Start gateway
echo "[start.sh] Starting gateway on $GW_HOST:$GW_PORT..."
cd "$GW_DIR"
python -m uvicorn app:app --host "$GW_HOST" --port "$GW_PORT" --log-level info &
GW_PID=$!
echo "$GW_PID" > "$PID_DIR/gateway.pid"

echo "[start.sh] Gateway PID=$GW_PID, mlx_audio PID=$MLX_PID${PIPER_PID:+, Piper PID=$PIPER_PID}"
echo "[start.sh] Control center: http://$GW_HOST:$GW_PORT/"

# Wait for any to exit
if [ -n "$PIPER_PID" ]; then
  wait -n "$MLX_PID" "$PIPER_PID" "$GW_PID" 2>/dev/null || true
else
  wait -n "$MLX_PID" "$GW_PID" 2>/dev/null || true
fi
