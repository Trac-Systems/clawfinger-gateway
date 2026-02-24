#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
GW_DIR="$(dirname "$SCRIPT_DIR")"
PID_DIR="$GW_DIR/tmp"

for pidfile in "$PID_DIR/gateway.pid" "$PID_DIR/piper.pid" "$PID_DIR/mlx_audio.pid"; do
  if [ -f "$pidfile" ]; then
    pid=$(cat "$pidfile")
    name=$(basename "$pidfile" .pid)
    if kill -0 "$pid" 2>/dev/null; then
      echo "[stop.sh] Stopping $name (PID $pid)..."
      kill "$pid" 2>/dev/null || true
    else
      echo "[stop.sh] $name (PID $pid) not running"
    fi
    rm -f "$pidfile"
  fi
done

echo "[stop.sh] Done"
