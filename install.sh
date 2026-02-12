#!/bin/bash
IMAGE="psyb0t/qwenspeak"
INSTALL_PATH="/usr/local/bin/qwenspeak"

# Resolve the real user when running under sudo
if [ -n "$SUDO_USER" ]; then
    REAL_HOME=$(getent passwd "$SUDO_USER" | cut -d: -f6)
    REAL_UID="$SUDO_UID"
    REAL_GID="$SUDO_GID"
else
    REAL_HOME="$HOME"
    REAL_UID=$(id -u)
    REAL_GID=$(id -g)
fi

QWENSPEAK_HOME="$REAL_HOME/.qwenspeak"

mkdir -p "$QWENSPEAK_HOME/work" "$QWENSPEAK_HOME/host_keys" "$QWENSPEAK_HOME/models"
touch "$QWENSPEAK_HOME/authorized_keys"

if [ ! -f "$QWENSPEAK_HOME/.env" ]; then
    cat > "$QWENSPEAK_HOME/.env" << ENVEOF
QWENSPEAK_PORT=2222
QWENSPEAK_MODELS_DIR=$QWENSPEAK_HOME/models
QWENSPEAK_CPUS=0
QWENSPEAK_MEMORY=0
QWENSPEAK_SWAP=0
ENVEOF
fi

cat > "$QWENSPEAK_HOME/docker-compose.yml" << EOF
services:
  qwenspeak:
    image: ${IMAGE}
    ports:
      - "\${QWENSPEAK_PORT:-2222}:22"
    environment:
      - LOCKBOX_UID=${REAL_UID}
      - LOCKBOX_GID=${REAL_GID}
    volumes:
      - ./authorized_keys:/etc/lockbox/authorized_keys:ro
      - ./host_keys:/etc/lockbox/host_keys
      - ./work:/work
      - \${QWENSPEAK_MODELS_DIR:-./models}:/models:ro
    cpus: \${QWENSPEAK_CPUS:-0}
    mem_limit: \${QWENSPEAK_MEMORY:-0}
    memswap_limit: \${QWENSPEAK_MEMSWAP:-0}
    restart: unless-stopped
EOF

cat > "$INSTALL_PATH" << 'SCRIPT'
#!/bin/bash

QWENSPEAK_HOME="__QWENSPEAK_HOME__"
ENV_FILE="$QWENSPEAK_HOME/.env"

compose() {
    docker compose --env-file "$ENV_FILE" -f "$QWENSPEAK_HOME/docker-compose.yml" "$@"
}

# Convert size string (e.g. 4g, 512m) to bytes
to_bytes() {
    local val="$1"
    if [ "$val" = "0" ]; then echo 0; return; fi
    local num="${val%[bBkKmMgG]*}"
    local unit="${val##*[0-9.]}"
    case "${unit,,}" in
        g) echo $(( ${num%.*} * 1073741824 )) ;;
        m) echo $(( ${num%.*} * 1048576 )) ;;
        k) echo $(( ${num%.*} * 1024 )) ;;
        *) echo "$num" ;;
    esac
}

# Compute memswap (Docker's memswap_limit = ram + swap)
compute_memswap() {
    . "$ENV_FILE"
    local mem="$QWENSPEAK_MEMORY"
    local swap="$QWENSPEAK_SWAP"

    if [ "$mem" = "0" ] || [ -z "$mem" ]; then
        sed -i '/^QWENSPEAK_MEMSWAP=/d' "$ENV_FILE"
        echo "QWENSPEAK_MEMSWAP=0" >> "$ENV_FILE"
        return
    fi

    if [ "$swap" = "0" ] || [ -z "$swap" ]; then
        sed -i '/^QWENSPEAK_MEMSWAP=/d' "$ENV_FILE"
        echo "QWENSPEAK_MEMSWAP=$mem" >> "$ENV_FILE"
        return
    fi

    local mem_bytes swap_bytes total
    mem_bytes=$(to_bytes "$mem")
    swap_bytes=$(to_bytes "$swap")
    total=$(( mem_bytes + swap_bytes ))

    sed -i '/^QWENSPEAK_MEMSWAP=/d' "$ENV_FILE"
    echo "QWENSPEAK_MEMSWAP=$total" >> "$ENV_FILE"
}

usage() {
    echo "Usage: qwenspeak <command>"
    echo ""
    echo "Commands:"
    echo "  start [-d] [-p PORT] [-m MODELS_DIR] [-c CPUS] [-r MEMORY] [-s SWAP]"
    echo "                        Start qwenspeak (-d for detached)"
    echo "                        -m  Models directory"
    echo "                        -c  CPU limit (e.g. 4, 0.5) - 0 = unlimited"
    echo "                        -r  RAM limit (e.g. 4g, 512m) - 0 = unlimited"
    echo "                        -s  Swap limit (e.g. 2g, 512m) - 0 = no swap"
    echo "  stop                  Stop qwenspeak"
    echo "  upgrade               Pull latest image and restart if needed"
    echo "  uninstall             Stop qwenspeak and remove everything"
    echo "  status                Show container status"
    echo "  logs                  Show container logs (pass extra args to docker compose logs)"
}

case "${1:-}" in
    start)
        shift
        DETACHED=false
        while [ $# -gt 0 ]; do
            case "$1" in
                -d) DETACHED=true ;;
                -p) shift; sed -i "s/^QWENSPEAK_PORT=.*/QWENSPEAK_PORT=$1/" "$ENV_FILE" ;;
                -m) shift; sed -i "s|^QWENSPEAK_MODELS_DIR=.*|QWENSPEAK_MODELS_DIR=$1|" "$ENV_FILE" ;;
                -c) shift; sed -i "s/^QWENSPEAK_CPUS=.*/QWENSPEAK_CPUS=$1/" "$ENV_FILE" ;;
                -r) shift; sed -i "s/^QWENSPEAK_MEMORY=.*/QWENSPEAK_MEMORY=$1/" "$ENV_FILE" ;;
                -s) shift; sed -i "s/^QWENSPEAK_SWAP=.*/QWENSPEAK_SWAP=$1/" "$ENV_FILE" ;;
            esac
            shift
        done

        if compose ps --status running 2>/dev/null | grep -q qwenspeak; then
            read -rp "qwenspeak is already running. Recreate? [y/N] " answer
            if [ "$answer" != "y" ] && [ "$answer" != "Y" ]; then
                exit 0
            fi
        fi

        compute_memswap

        COMPOSE_ARGS="up --force-recreate"
        if [ "$DETACHED" = true ]; then
            COMPOSE_ARGS="up --force-recreate -d"
        fi

        compose $COMPOSE_ARGS
        ;;
    stop)
        compose down
        ;;
    upgrade)
        WAS_RUNNING=false
        if compose ps --status running 2>/dev/null | grep -q qwenspeak; then
            WAS_RUNNING=true
            read -rp "qwenspeak is running. Stop it to upgrade? [y/N] " answer
            if [ "$answer" != "y" ] && [ "$answer" != "Y" ]; then
                echo "Upgrade cancelled"
                exit 0
            fi
            compose down
        fi

        docker pull psyb0t/qwenspeak
        sudo -v
        echo "Updating qwenspeak..."
        curl -fsSL https://raw.githubusercontent.com/psyb0t/docker-qwenspeak/main/install.sh | sudo bash
        echo "Upgrade complete"

        if [ "$WAS_RUNNING" = true ]; then
            read -rp "Start qwenspeak again? [y/N] " answer
            if [ "$answer" != "y" ] && [ "$answer" != "Y" ]; then
                exit 0
            fi
            compose up -d
        fi
        ;;
    uninstall)
        read -rp "Uninstall qwenspeak? [y/N] " answer
        if [ "$answer" != "y" ] && [ "$answer" != "Y" ]; then
            exit 0
        fi

        compose down 2>/dev/null
        rm -f "$0"

        read -rp "Remove $QWENSPEAK_HOME? This deletes all data including work files. [y/N] " answer
        if [ "$answer" = "y" ] || [ "$answer" = "Y" ]; then
            rm -rf "$QWENSPEAK_HOME"
        fi

        echo "qwenspeak uninstalled"
        ;;

    status)
        compose ps
        ;;
    logs)
        shift
        compose logs "$@"
        ;;
    *)
        usage
        ;;
esac
SCRIPT

sed -i "s|__QWENSPEAK_HOME__|$QWENSPEAK_HOME|g" "$INSTALL_PATH"
chmod +x "$INSTALL_PATH"

chown -R "$REAL_UID:$REAL_GID" "$QWENSPEAK_HOME"

docker pull "$IMAGE"

echo ""
echo "qwenspeak installed!"
echo ""
echo "  Command:         $INSTALL_PATH"
echo "  Authorized keys: $QWENSPEAK_HOME/authorized_keys"
echo "  Work directory:  $QWENSPEAK_HOME/work"
echo "  Models directory: $QWENSPEAK_HOME/models/"
echo ""
echo "Add your SSH public key(s) to the authorized_keys file and run:"
echo ""
echo "  qwenspeak start -d"
echo ""
