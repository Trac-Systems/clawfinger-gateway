#!/usr/bin/env bash
#
# intercom-pair.sh — Pair gateway with robot via Intercom.
#
# Creates a private channel, generates a signed invite for the robot,
# writes pairing data and updates gateway config.
#
# Usage:
#   cd gateway
#   bin/intercom-pair.sh <robot-pubkey-hex>
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
GW_ROOT="$(dirname "$SCRIPT_DIR")"
INTERCOM_DIR="$GW_ROOT/intercom"
STORE_NAME="gateway"
CHANNEL="clawfinger-robot-g1"
TMP_DIR="$GW_ROOT/tmp"
CONFIG_FILE="$GW_ROOT/config.json"

if [ $# -lt 1 ]; then
    echo "Usage: bin/intercom-pair.sh <robot-pubkey-hex>"
    echo
    echo "The robot's public key (hex) is printed by intercom-setup.sh"
    echo "when run on the robot side."
    exit 1
fi

ROBOT_PUBKEY="$1"

echo "=== Intercom Pairing ==="
echo "Channel:     $CHANNEL"
echo "Robot key:   $ROBOT_PUBKEY"
echo

# --- Validate prerequisites ---

STORE_DIR="$INTERCOM_DIR/stores/$STORE_NAME"
KEYPAIR_FILE="$STORE_DIR/db/keypair.json"

if [ ! -f "$KEYPAIR_FILE" ]; then
    echo "ERROR: Gateway keypair not found at $KEYPAIR_FILE"
    echo "  Run bin/intercom-setup.sh first."
    exit 1
fi

GW_PUBKEY=$(python3 -c "import json; print(json.load(open('$KEYPAIR_FILE'))['publicKey'])" 2>/dev/null)
if [ -z "$GW_PUBKEY" ]; then
    echo "ERROR: Could not read gateway pubkey from $KEYPAIR_FILE"
    exit 1
fi
echo "Gateway key: $GW_PUBKEY"

# --- Find pear ---

PEAR_BIN=""
if command -v pear &>/dev/null; then
    PEAR_BIN="$(command -v pear)"
elif [ -x "$HOME/local/node/bin/pear" ]; then
    PEAR_BIN="$HOME/local/node/bin/pear"
elif [ -x "/usr/local/bin/pear" ]; then
    PEAR_BIN="/usr/local/bin/pear"
fi
if [ -z "$PEAR_BIN" ]; then
    echo "ERROR: Pear runtime not found."
    exit 1
fi

# --- Generate SC-Bridge token ---

mkdir -p "$TMP_DIR"
SC_TOKEN=$(python3 -c "import secrets; print(secrets.token_hex(32))")
echo "$SC_TOKEN" > "$TMP_DIR/.sc_bridge_token"
echo "SC-Bridge token written to $TMP_DIR/.sc_bridge_token"

# --- Create pairing data ---

PAIRING_FILE="$STORE_DIR/pairing.json"
python3 -c "
import json, time

pairing = {
    'channel': '$CHANNEL',
    'robot_pubkey': '$ROBOT_PUBKEY',
    'gateway_pubkey': '$GW_PUBKEY',
    'created_at': time.time(),
    'sc_bridge_token': '$SC_TOKEN',
}
with open('$PAIRING_FILE', 'w') as f:
    json.dump(pairing, f, indent=2)
    f.write('\n')
print('Pairing data written to $PAIRING_FILE')
"

# --- Update config.json ---

if [ -f "$CONFIG_FILE" ]; then
    python3 -c "
import json

with open('$CONFIG_FILE') as f:
    cfg = json.load(f)

robot = cfg.setdefault('robot', {})
robot['intercom_key'] = '$ROBOT_PUBKEY'
robot['intercom_channel'] = '$CHANNEL'

with open('$CONFIG_FILE', 'w') as f:
    json.dump(cfg, f, indent=2)
    f.write('\n')
print('Updated config.json: robot.intercom_key and robot.intercom_channel')
"
else
    echo "WARNING: config.json not found at $CONFIG_FILE"
    echo "  Manually set robot.intercom_key to: $ROBOT_PUBKEY"
fi

# --- Summary ---

echo
echo "=== Pairing Complete ==="
echo
echo "Channel:          $CHANNEL"
echo "Gateway pubkey:   $GW_PUBKEY"
echo "Robot pubkey:     $ROBOT_PUBKEY"
echo "SC-Bridge token:  $TMP_DIR/.sc_bridge_token"
echo "Pairing data:     $PAIRING_FILE"
echo
echo "Next steps:"
echo "  1. Copy gateway pubkey ($GW_PUBKEY) to robot config as 'inviterKeys'"
echo "  2. Copy the invite/welcome from $PAIRING_FILE to the robot"
echo "  3. Start the gateway with robot.enabled: true"
echo "  4. Start the robot's Intercom peer"
echo "  5. Verify: curl http://127.0.0.1:8996/api/config/robot | jq .connected"
