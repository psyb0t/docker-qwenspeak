#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

IMAGE="psyb0t/qwenspeak:latest"
CONTAINER="qwenspeak"
PORT="${PORT:-2222}"
MODELS_DIR="${MODELS_DIR:-/mnt/hdd-2tb-1/bw/models/qwen3-tts}"

AUTHKEYS="$SCRIPT_DIR/authorized_keys"
HOST_KEYS="$SCRIPT_DIR/host_keys"
WORK="$SCRIPT_DIR/work"

if [ ! -f "$AUTHKEYS" ]; then
    echo "Error: $AUTHKEYS not found"
    echo "Create it: cat ~/.ssh/id_rsa.pub > authorized_keys"
    exit 1
fi

if [ ! -d "$MODELS_DIR" ]; then
    echo "Error: models dir not found: $MODELS_DIR"
    echo "Set MODELS_DIR to your qwen3-tts models directory"
    exit 1
fi

mkdir -p "$HOST_KEYS" "$WORK"

docker rm -f "$CONTAINER" >/dev/null 2>&1 || true

docker run -d \
    --name "$CONTAINER" \
    --restart unless-stopped \
    --memory 4g \
    --memory-swap 4g \
    -p "$PORT":22 \
    -e "LOCKBOX_UID=$(id -u)" \
    -e "LOCKBOX_GID=$(id -g)" \
    -v "$AUTHKEYS":/etc/lockbox/authorized_keys:ro \
    -v "$HOST_KEYS":/etc/lockbox/host_keys \
    -v "$WORK":/work \
    -v "$MODELS_DIR":/models:ro \
    "$IMAGE"

echo "Running: $CONTAINER"
echo "  SSH:    ssh -p $PORT tts@localhost"
echo "  Models: $MODELS_DIR -> /models (ro)"
echo "  Work:   $WORK -> /work"
echo "  Memory: 4GB (hard limit)"
