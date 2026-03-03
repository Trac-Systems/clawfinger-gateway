#!/usr/bin/env bash
#
# intercom-setup.sh — One-time Intercom installation for the gateway.
#
# Checks prerequisites, clones Intercom into gateway/intercom/,
# runs npm install, generates ED25519 keypair, prints gateway pubkey.
#
# Usage:
#   cd gateway
#   bin/intercom-setup.sh
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
GW_ROOT="$(dirname "$SCRIPT_DIR")"
INTERCOM_DIR="$GW_ROOT/intercom"
STORE_NAME="${1:-gateway}"
INTERCOM_REPO="https://github.com/Trac-Systems/intercom.git"

echo "=== Intercom Setup ==="
echo "Gateway root: $GW_ROOT"
echo "Intercom dir: $INTERCOM_DIR"
echo "Store name:   $STORE_NAME"
echo

# --- Check prerequisites ---

# Node 22.x
if ! command -v node &>/dev/null; then
    echo "ERROR: Node.js not found. Install Node 22.x first."
    exit 1
fi
NODE_VER=$(node -v | sed 's/v//' | cut -d. -f1)
if [ "$NODE_VER" -lt 22 ]; then
    echo "ERROR: Node $NODE_VER found, but 22.x or later is required."
    exit 1
fi
echo "[OK] Node $(node -v)"

# Pear runtime
PEAR_BIN=""
if command -v pear &>/dev/null; then
    PEAR_BIN="$(command -v pear)"
elif [ -x "$HOME/local/node/bin/pear" ]; then
    PEAR_BIN="$HOME/local/node/bin/pear"
elif [ -x "/usr/local/bin/pear" ]; then
    PEAR_BIN="/usr/local/bin/pear"
fi
if [ -z "$PEAR_BIN" ]; then
    echo "WARNING: Pear runtime not found in PATH."
    echo "  Install: npm i -g pear"
    echo "  Continuing — you can install Pear later."
else
    echo "[OK] Pear: $PEAR_BIN"
fi

# --- Clone Intercom ---

if [ -d "$INTERCOM_DIR/.git" ]; then
    echo "[OK] Intercom already cloned, pulling latest..."
    cd "$INTERCOM_DIR"
    git pull --ff-only || echo "WARNING: git pull failed (offline?), using existing"
else
    echo "Cloning Intercom..."
    git clone "$INTERCOM_REPO" "$INTERCOM_DIR"
fi

# --- Install dependencies ---

echo "Installing npm dependencies..."
cd "$INTERCOM_DIR"
npm install --omit=dev

# --- Generate keypair ---

STORE_DIR="$INTERCOM_DIR/stores/$STORE_NAME"
KEYPAIR_FILE="$STORE_DIR/db/keypair.json"

if [ -f "$KEYPAIR_FILE" ]; then
    echo "[OK] Keypair already exists at $KEYPAIR_FILE"
else
    echo "Generating keypair (brief Intercom run)..."
    mkdir -p "$STORE_DIR/db"
    if [ -n "$PEAR_BIN" ]; then
        # Run Intercom briefly to trigger keypair generation, then kill
        timeout 10 "$PEAR_BIN" run "$INTERCOM_DIR" \
            --peer-store-name "$STORE_NAME" \
            --sc-bridge 0 \
            --sidechannel-quiet 1 \
            2>/dev/null || true
    else
        echo "WARNING: Pear not found — cannot auto-generate keypair."
        echo "  Install Pear, then run:"
        echo "  pear run $INTERCOM_DIR --peer-store-name $STORE_NAME"
        exit 1
    fi
fi

# --- Print pubkey ---

if [ -f "$KEYPAIR_FILE" ]; then
    PUBKEY=$(python3 -c "import json; print(json.load(open('$KEYPAIR_FILE'))['publicKey'])" 2>/dev/null || echo "")
    if [ -n "$PUBKEY" ]; then
        echo
        echo "=== Setup Complete ==="
        echo
        echo "Gateway public key (hex):"
        echo "  $PUBKEY"
        echo
        echo "Copy this key to the robot's config as 'inviterKeys'."
        echo "Then run: bin/intercom-pair.sh <robot-pubkey-hex>"
    else
        echo "WARNING: Could not read pubkey from $KEYPAIR_FILE"
    fi
else
    echo "WARNING: Keypair file not found at $KEYPAIR_FILE"
    echo "  Run Intercom manually to generate it."
fi
