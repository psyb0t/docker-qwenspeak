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

# run_test_negative <test_name> <ssh_command> <grep_pattern_that_should_NOT_match>
run_test_negative() {
    local name="$1"
    local cmd="$2"
    local pattern="$3"

    local output
    output=$(ssh_cmd "$cmd")

    if echo "$output" | grep -q "$pattern"; then
        fail "$name" "$output"
        return
    fi

    pass "$name"
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

echo ""
echo "=== Testing tts command help ==="

#                  name                              command                                         pattern
run_test           "tts help shows usage"            "tts --help"                                    "usage:" "i"
run_test           "tts help shows custom-voice"     "tts --help"                                    "custom-voice"
run_test           "tts help shows voice-design"     "tts --help"                                    "voice-design"
run_test           "tts help shows voice-clone"      "tts --help"                                    "voice-clone"
run_test           "tts help shows list-speakers"    "tts --help"                                    "list-speakers"
run_test           "tts help shows tokenize"         "tts --help"                                    "tokenize"
run_test           "tts help shows --models-dir"     "tts --help"                                    "models-dir"

echo ""
echo "=== Testing subcommand help ==="

run_test           "custom-voice help"               "tts custom-voice --help"                       "speaker"
run_test           "custom-voice shows --instruct"   "tts custom-voice --help"                       "instruct"
run_test           "custom-voice shows --language"   "tts custom-voice --help"                       "language"
run_test           "custom-voice shows --output"     "tts custom-voice --help"                       "output"
run_test           "voice-design help"               "tts voice-design --help"                       "instruct"
run_test           "voice-clone help"                "tts voice-clone --help"                        "ref-audio"
run_test           "voice-clone shows --ref-text"    "tts voice-clone --help"                        "ref-text"
run_test           "voice-clone shows --x-vector"    "tts voice-clone --help"                        "x-vector-only"
run_test           "tokenize help"                   "tts tokenize --help"                           "audio"

echo ""
echo "=== Testing list-speakers ==="

run_test           "list-speakers runs"              "tts list-speakers"                             "Available speakers"
run_test           "list-speakers shows Vivian"      "tts list-speakers"                             "Vivian"
run_test           "list-speakers shows Ryan"        "tts list-speakers"                             "Ryan"
run_test           "list-speakers shows Aiden"       "tts list-speakers"                             "Aiden"
run_test           "list-speakers shows Sohee"       "tts list-speakers"                             "Sohee"

echo ""
echo "=== Testing error handling ==="

run_test           "no subcommand shows help"        "tts"                                           "usage:" "i"
run_test           "bad subcommand errors"           "tts yolo"                                      "invalid choice\|error" "i"
run_test           "custom-voice no text errors"     "tts custom-voice"                              "required\|error" "i"
run_test           "voice-clone no ref-audio errors" "tts voice-clone test"                          "required\|error" "i"

echo ""
echo "================================"
echo "Results: $PASSED passed, $FAILED failed, $TOTAL total"
echo "================================"

if [ "$FAILED" -gt 0 ]; then
    exit 1
fi
