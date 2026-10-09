#!/usr/bin/env bash
# Run llama-bench and an HTTP API benchmark using a host-mounted model volume.
# Usage: scripts/run_docker_bench.sh [model_key]
set -euo pipefail

REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
IMAGE=${IMAGE:-a6000-gguf-bench:latest}
MODEL_ROOT=${MODEL_ROOT:-$REPO_ROOT/weights}
RESULT_ROOT=${RESULT_ROOT:-$REPO_ROOT/results}
GPU_REQUEST=${GPU_REQUEST:-0}
GPU_MODE=${GPU_MODE:-1gpu}
PORT=${PORT:-18080}
REPEATS=${REPEATS:-5}
MODELS_ENV=${MODELS_ENV:-$REPO_ROOT/configs/models.env}

# shellcheck disable=SC1090
source "$MODELS_ENV"

model_info() {
  case "${1:?model key required}" in
    qwen_q4)     echo "QWEN_Q4_MODEL qwen36-q4 qwen-q4-1gpu" ;;
    qwen_q6)     echo "QWEN_Q6_MODEL qwen36-q6 qwen-q6-1gpu" ;;
    qwen_q8)     echo "QWEN_Q8_MODEL qwen36-q8 qwen-q8-1gpu" ;;
    qwen38_q4)   echo "QWEN38_Q4_MODEL qwen38-q4 qwen38-q4-1gpu" ;;
    qwen36_35b_q4km) echo "QWEN36_35B_Q4KM_MODEL qwen36-35b-q4km qwen36-35b-q4km-1gpu" ;;
    nemotron_q4ks) echo "NEMOTRON_Q4KS_MODEL nemotron-3.5-q4ks nemotron-q4ks-1gpu" ;;
    laguna_q4xl) echo "LAGUNA_Q4XL_MODEL laguna-q4xl laguna-q4xl" ;;
    laguna_iq4nl) echo "LAGUNA_IQ4NL_MODEL laguna-iq4nl laguna-iq4nl" ;;
    laguna_q4m)  echo "LAGUNA_Q4M_MODEL laguna-q4m laguna-q4m" ;;
    *) echo "Unknown model key: $1" >&2; exit 2 ;;
  esac
}

configure_gpu_mode() {
  SERVER_GPU_ARGS=()
  BENCH_GPU_ARGS=()
  case "$GPU_MODE" in
    1gpu)
      SERVER_GPU_ARGS+=(--split-mode none)
      BENCH_GPU_ARGS+=(-sm none)
      ;;
    2gpu-layer)
      [[ "$GPU_REQUEST" == *,* ]] || { echo '2gpu modes require GPU=0,1 (or another comma-separated pair).' >&2; exit 2; }
      SERVER_GPU_ARGS+=(--split-mode layer --tensor-split 1,1)
      BENCH_GPU_ARGS+=(-sm layer -ts 1/1)
      ;;
    2gpu-row)
      [[ "$GPU_REQUEST" == *,* ]] || { echo '2gpu modes require GPU=0,1 (or another comma-separated pair).' >&2; exit 2; }
      SERVER_GPU_ARGS+=(--split-mode row --tensor-split 1,1)
      BENCH_GPU_ARGS+=(-sm row -ts 1/1)
      ;;
    *) echo "Unknown GPU_MODE: $GPU_MODE (use 1gpu, 2gpu-layer, or 2gpu-row)." >&2; exit 2 ;;
  esac
}

MODEL_KEY=${1:-qwen_q4}
read -r MODEL_ENV_VAR MODEL_ALIAS LABEL_SUFFIX <<< "$(model_info "$MODEL_KEY")"
MODEL_PATH=${!MODEL_ENV_VAR:-}
[[ -n "$MODEL_PATH" ]] || { echo "Model path for $MODEL_KEY is empty; check $MODELS_ENV" >&2; exit 2; }
HOST_MODEL_PATH="$MODEL_ROOT${MODEL_PATH#/models}"
[[ -s "$HOST_MODEL_PATH" ]] || { echo "Model file not found: $HOST_MODEL_PATH" >&2; exit 2; }

configure_gpu_mode

STAMP=${RUN_STAMP:-$(date -u +%Y%m%dT%H%M%SZ)}
OUT="$RESULT_ROOT/$STAMP/$LABEL_SUFFIX"
CONTAINER_OUT="/results/$STAMP/$LABEL_SUFFIX"
SERVER_NAME="a6000-${MODEL_KEY}-${PORT}"
mkdir -p "$OUT/bench"

echo "=== Docker benchmark for $MODEL_KEY ==="
echo "Image:       $IMAGE"
echo "Model host:  $HOST_MODEL_PATH"
echo "GPU:         device=$GPU_REQUEST mode=$GPU_MODE"
echo "Output dir:  $OUT"

echo "[1/3] Running llama-bench..."
docker run --rm \
  --gpus "device=$GPU_REQUEST" \
  --ipc=host \
  -v "$MODEL_ROOT:/models:ro" \
  "$IMAGE" \
  llama-bench \
    -m "$MODEL_PATH" \
    -ngl 99 -fa on \
    "${BENCH_GPU_ARGS[@]}" \
    -r "$REPEATS" -o json \
    -p 512,2048,8192 \
    -n 128,512 \
    > "$OUT/bench/result.json" \
    2> "$OUT/bench/stderr.log"

echo "[2/3] Starting llama-server..."
docker rm -f "$SERVER_NAME" >/dev/null 2>&1 || true
cleanup() {
  local status=$?
  docker logs "$SERVER_NAME" > "$OUT/server.log" 2>&1 || true
  docker rm -f "$SERVER_NAME" >/dev/null 2>&1 || true
  exit "$status"
}
trap cleanup EXIT INT TERM

docker run -d \
  --gpus "device=$GPU_REQUEST" \
  --ipc=host \
  --name "$SERVER_NAME" \
  -v "$MODEL_ROOT:/models:ro" \
  -p "$PORT:8080" \
  "$IMAGE" \
  llama-server \
    -m "$MODEL_PATH" \
    --alias "$MODEL_ALIAS" \
    --host 0.0.0.0 --port 8080 \
    --n-gpu-layers 99 \
    --ctx-size 8192 \
    --flash-attn on \
    --parallel 1 \
    --jinja \
    --reasoning off \
    --temp 0 --seed 42 \
    "${SERVER_GPU_ARGS[@]}" \
    >/dev/null

echo "Waiting for llama-server..."
ready=0
for _ in $(seq 1 300); do
  if curl -fsS "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; then
    ready=1
    break
  fi
  sleep 2
done
if [[ "$ready" != 1 ]]; then
  echo "llama-server did not become ready." >&2
  docker logs "$SERVER_NAME" 2>&1 | tail -50 >&2 || true
  exit 1
fi

echo "[3/3] Running API benchmark..."
docker run --rm \
  --network=host \
  -v "$REPO_ROOT/scripts:/scripts:ro" \
  -v "$RESULT_ROOT:/results" \
  "$IMAGE" \
  python3 /scripts/api_bench.py \
    --base-url "http://127.0.0.1:$PORT" \
    --prompt-tokens 512,2048,4096 \
    --generate-tokens 128,512 \
    --repetitions "$REPEATS" \
    --warmups 1 \
    --timeout 1800 \
    --output "$CONTAINER_OUT/api-bench.jsonl"

echo "Results saved to: $OUT"
