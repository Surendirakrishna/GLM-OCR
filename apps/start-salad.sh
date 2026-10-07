#!/bin/bash
# ocr.z.ai UI + GLM-OCR harness with salad as the region recognizer instead of
# GLM-OCR, and docling-layout-heron-101 for layout instead of PP-DocLayoutV3.
# See README.md.
#
#   salad (SGLang, docker sglang_salad_fp8)        :30000
#     FP8 weights: 1.6x faster decoding than bf16 sglang_salad, same
#     benchmark accuracy. SALAD_CONTAINER=sglang_salad uses bf16.
#   heron layout (docker docling-layout-heron101)  :8600
#   glmocr harness (configs/salad.yaml)            :5012
#   backend (FastAPI)                              :8010
#   frontend (vite)                                :3006
#
# Other GPU model containers are stopped first: salad needs ~26 GB of the
# 32 GB card. Ctrl+C stops the three app processes; the model containers keep
# running (docker stop sglang_salad_fp8 to free the GPU).

set -e

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG="$ROOT/configs/salad.yaml"
LOG_DIR="$ROOT/apps/logs"
SDK_PY="$(dirname "$ROOT")/.venv/bin/python"

HARNESS_PORT=5012
BACKEND_PORT=8010
FRONTEND_PORT=3006
SALAD_CONTAINER=${SALAD_CONTAINER:-sglang_salad_fp8}
HERON_CONTAINER=docling-layout-heron101
# Stopped first: other GPU models, and the other salad variant (same port).
OTHER_MODEL_CONTAINERS="sglang_mineru25_pro sglang_salad sglang_salad_fp8"
OTHER_MODEL_CONTAINERS=$(echo $OTHER_MODEL_CONTAINERS | tr ' ' '\n' | grep -vx "$SALAD_CONTAINER")

mkdir -p "$LOG_DIR"
HARNESS_PID= BACKEND_PID= FRONTEND_PID=

kill_tree() {  # pnpm/vite spawn children that a plain kill would orphan
    for child in $(pgrep -P "$1"); do kill_tree "$child"; done
    kill "$1" 2>/dev/null || true
}

stop_apps() {
    for p in $HARNESS_PID $BACKEND_PID $FRONTEND_PID; do kill_tree "$p"; done
}

wait_for() {  # url, name, seconds
    for _ in $(seq 1 "$3"); do
        curl -sf -m 2 "$1" >/dev/null && { echo "  $2 is up"; return 0; }
        sleep 2
    done
    echo "  $2 did not come up; see $LOG_DIR" >&2
    stop_apps  # don't leave the services started so far running
    exit 1
}

echo "[1/4] salad + heron model servers"
for c in $OTHER_MODEL_CONTAINERS; do
    if [ "$(docker inspect -f '{{.State.Running}}' "$c" 2>/dev/null)" = "true" ]; then
        echo "  stopping $c"
        docker stop "$c" >/dev/null
    fi
done
if [ "$(docker inspect -f '{{.State.Running}}' "$SALAD_CONTAINER")" != "true" ]; then
    docker start "$SALAD_CONTAINER" >/dev/null
fi
if [ "$(docker inspect -f '{{.State.Running}}' "$HERON_CONTAINER")" != "true" ]; then
    docker start "$HERON_CONTAINER" >/dev/null
fi
wait_for http://127.0.0.1:30000/v1/models salad 300
wait_for http://127.0.0.1:8600/health heron 120

echo "[2/4] glmocr harness :$HARNESS_PORT"
cd "$ROOT"
"$SDK_PY" -m glmocr.server --config "$CONFIG" >"$LOG_DIR/harness.log" 2>&1 &
HARNESS_PID=$!
wait_for "http://127.0.0.1:$HARNESS_PORT/health" harness 90

echo "[3/4] backend :$BACKEND_PORT"
cd "$ROOT/apps/backend"
layout_ocr_url="http://127.0.0.1:$HARNESS_PORT/glmocr/parse" \
    .venv/bin/uvicorn app.main:app --host 0.0.0.0 --port "$BACKEND_PORT" \
    >"$LOG_DIR/backend.log" 2>&1 &
BACKEND_PID=$!
wait_for "http://127.0.0.1:$BACKEND_PORT/docs" backend 60

echo "[4/4] frontend :$FRONTEND_PORT"
cd "$ROOT/apps/frontend"
BACKEND_URL="http://127.0.0.1:$BACKEND_PORT" \
    npx -y pnpm@9 dev --host 0.0.0.0 --port "$FRONTEND_PORT" --strictPort \
    >"$LOG_DIR/frontend.log" 2>&1 &
FRONTEND_PID=$!
wait_for "http://127.0.0.1:$FRONTEND_PORT/" frontend 60

echo
echo "UI:      http://localhost:$FRONTEND_PORT"
echo "Harness: http://localhost:$HARNESS_PORT/glmocr/parse"
echo "Logs:    $LOG_DIR"

trap 'stop_apps; exit 0' INT TERM
wait
