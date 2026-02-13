#!/bin/bash
set -e

IMAGE="psyb0t/qwenspeak:latest-test"
CONTAINER="qwenspeak-test-$$"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
TMPDIR="$SCRIPT_DIR/.test-tmp-$$"
mkdir -p "$TMPDIR"
KEY="$TMPDIR/id_test"
AUTHKEYS="$TMPDIR/authorized_keys"
PASSED=0
FAILED=0
TOTAL=0

cleanup() {
    echo ""
    echo "Cleaning up..."
    docker rm -f "$CONTAINER" >/dev/null 2>&1 || true
    rm -rf "$TMPDIR"
}

trap cleanup EXIT

fail() {
    echo "  FAIL: $1"
    if [ -n "$2" ]; then
        echo "        got: $(echo "$2" | head -1)"
    fi
    FAILED=$((FAILED + 1))
    TOTAL=$((TOTAL + 1))
}

pass() {
    echo "  PASS: $1"
    PASSED=$((PASSED + 1))
    TOTAL=$((TOTAL + 1))
}

ssh_cmd() {
    ssh -p 22 \
        -i "$KEY" \
        -o StrictHostKeyChecking=no \
        -o UserKnownHostsFile=/dev/null \
        -o LogLevel=ERROR \
        -o ConnectTimeout=5 \
        "tts@$CONTAINER_IP" "$1" 2>&1 || true
}

# run_test <test_name> <ssh_command> <grep_pattern> [case_insensitive]
run_test() {
    local name="$1"
    local cmd="$2"
    local pattern="$3"
    local case_insensitive="${4:-}"

    local output
    output=$(ssh_cmd "$cmd")

    local grep_flags="-q"
    if [ "$case_insensitive" = "i" ]; then
        grep_flags="-qi"
    fi

    if echo "$output" | grep $grep_flags "$pattern"; then
        pass "$name"
        return
    fi

    fail "$name" "$output"
}

echo "=== Building test image ==="
make build-test

echo ""
echo "=== Generating test SSH key ==="
ssh-keygen -t ed25519 -f "$KEY" -N "" -q
cp "$KEY.pub" "$AUTHKEYS"

echo ""
echo "=== Starting container ==="
docker run -d \
    --name "$CONTAINER" \
    -e "LOCKBOX_UID=$(id -u)" \
    -e "LOCKBOX_GID=$(id -g)" \
    "$IMAGE" >/dev/null

docker cp "$AUTHKEYS" "$CONTAINER:/etc/lockbox/authorized_keys"
docker exec "$CONTAINER" chmod 644 /etc/lockbox/authorized_keys

CONTAINER_IP=$(docker inspect -f '{{range .NetworkSettings.Networks}}{{.IPAddress}}{{end}}' "$CONTAINER")
echo "Container IP: $CONTAINER_IP"

echo "Waiting for sshd..."
for i in $(seq 1 30); do
    if docker exec "$CONTAINER" pgrep sshd >/dev/null 2>&1; then
        break
    fi
    sleep 0.5
done
sleep 1

# --- table-driven tests ---
# format: "command|pattern|flags|test name"
# flags: i = case insensitive, empty = default

TESTS=(
    # main help
    "tts --help|usage:|i|tts help shows usage"
    "tts --help|print-yaml||tts help shows print-yaml"
    "tts --help|list-speakers||tts help shows list-speakers"
    "tts --help|tokenize||tts help shows tokenize"
    "tts --help|models-dir||tts help shows --models-dir"

    # print-yaml
    "tts print-yaml|steps:||print-yaml shows steps"
    "tts print-yaml|custom-voice||print-yaml shows custom-voice"
    "tts print-yaml|voice-design||print-yaml shows voice-design"
    "tts print-yaml|voice-clone||print-yaml shows voice-clone"
    "tts print-yaml|temperature||print-yaml shows temperature"
    "tts print-yaml|speaker||print-yaml shows speaker"
    "tts print-yaml|ref_audio||print-yaml shows ref_audio"
    "tts print-yaml|instruct||print-yaml shows instruct"

    # list-speakers
    "tts list-speakers|Available speakers||list-speakers runs"
    "tts list-speakers|Vivian||list-speakers shows Vivian"
    "tts list-speakers|Ryan||list-speakers shows Ryan"
    "tts list-speakers|Aiden||list-speakers shows Aiden"
    "tts list-speakers|Sohee||list-speakers shows Sohee"
    "tts list-speakers|Ono_Anna||list-speakers shows Ono_Anna"

    # tokenize help
    "tts tokenize --help|audio||tokenize shows audio arg"
    "tts tokenize --help|output||tokenize shows --output"
)

echo ""
echo "=== Running tests ==="

for entry in "${TESTS[@]}"; do
    IFS='|' read -r cmd pattern flags name <<< "$entry"
    run_test "$name" "$cmd" "$pattern" "$flags"
done

echo ""
echo "================================"
echo "Results: $PASSED passed, $FAILED failed, $TOTAL total"
echo "================================"

if [ "$FAILED" -gt 0 ]; then
    exit 1
fi
