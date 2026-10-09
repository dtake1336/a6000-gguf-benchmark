#!/usr/bin/env bash
set -euo pipefail

SERVER_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
REPO_ROOT=$(cd "$SERVER_DIR/.." && pwd)
ENV_FILE=${ENV_FILE:-$SERVER_DIR/.env}

if [[ -f "$ENV_FILE" ]]; then
  while IFS='=' read -r key value; do
    [[ "$key" =~ ^[A-Za-z_][A-Za-z0-9_]*$ ]] || continue
    if [[ ! -v "$key" ]]; then
      export "$key=$value"
    fi
  done < <(sed -E 's/[[:space:]]*#.*$//' "$ENV_FILE")
fi

BENCH_IMAGE=${BENCH_IMAGE:-a6000-gguf-bench:latest}
BASE_URL=${BASE_URL:-http://127.0.0.1:${PORT:-8080}}
PROMPT_TOKENS=${PROMPT_TOKENS:-8192,32768,65536,131072,196608,245760,261888}
GENERATE_TOKENS=${GENERATE_TOKENS:-2048}
REPETITIONS=${REPETITIONS:-1}
WARMUPS=${WARMUPS:-1}
TIMEOUT=${TIMEOUT:-7200}
RUN_STAMP=${RUN_STAMP:-$(date -u +%Y%m%dT%H%M%SZ)}
RESULT_DIR=${RESULT_DIR:-$SERVER_DIR/benchmarks/$RUN_STAMP}
RESULTS_CSV=${RESULTS_CSV:-$SERVER_DIR/benchmarks/results.csv}

curl --fail --silent --show-error "$BASE_URL/health" >/dev/null
command -v jq >/dev/null 2>&1 || { echo "jq is required" >&2; exit 127; }
mkdir -p "$RESULT_DIR"

GPU_LOG="$RESULT_DIR/nvidia-smi.csv"
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi \
    --query-gpu=index,name,memory.used,memory.total,utilization.gpu,power.draw \
    --format=csv,noheader,nounits -l 1 >"$GPU_LOG" 2>&1 &
  GPU_MONITOR_PID=$!
else
  GPU_MONITOR_PID=
fi

cleanup() {
  if [[ -n "${GPU_MONITOR_PID:-}" ]]; then
    kill "$GPU_MONITOR_PID" >/dev/null 2>&1 || true
    wait "$GPU_MONITOR_PID" >/dev/null 2>&1 || true
  fi
}
trap cleanup EXIT INT TERM

docker run --rm \
  --network=host \
  -v "$REPO_ROOT/scripts:/scripts:ro" \
  -v "$RESULT_DIR:/results" \
  "$BENCH_IMAGE" \
  python3 /scripts/api_bench.py \
    --base-url "$BASE_URL" \
    --prompt-tokens "$PROMPT_TOKENS" \
    --generate-tokens "$GENERATE_TOKENS" \
    --repetitions "$REPETITIONS" \
    --warmups "$WARMUPS" \
    --timeout "$TIMEOUT" \
    --measure-ttft \
    --output /results/api-bench.jsonl

gpu_memory() {
  local gpu_index=$1
  [[ -s "$GPU_LOG" ]] || return 0
  awk -F, -v wanted="$gpu_index" '
    {
      index_value=$1
      memory_value=$3
      gsub(/^[[:space:]]+|[[:space:]]+$/, "", index_value)
      gsub(/^[[:space:]]+|[[:space:]]+$/, "", memory_value)
      if (index_value == wanted && memory_value + 0 > maximum + 0) maximum=memory_value
    }
    END { if (maximum != "") print maximum }
  ' "$GPU_LOG"
}

gpu0_vram=$(gpu_memory 0)
gpu1_vram=$(gpu_memory 1)
llama_commit=${LLAMA_CPP_REF:-unknown}
context_size=${CTX_SIZE:-261888}
kv_k=${CACHE_TYPE_K:-q8_0}
kv_v=${CACHE_TYPE_V:-q8_0}
batch_size=${BATCH_SIZE:-2048}
ubatch_size=${UBATCH_SIZE:-256}

if [[ ! -e "$RESULTS_CSV" ]]; then
  mkdir -p "$(dirname "$RESULTS_CSV")"
  printf 'timestamp,llama_commit,context_size,kv_k,kv_v,batch,ubatch,prompt_tokens,generated_tokens,prompt_tok_s,generation_tok_s,ttft_seconds,gpu0_vram_mb,gpu1_vram_mb\n' >"$RESULTS_CSV"
fi

jq -r \
  --arg timestamp "$RUN_STAMP" \
  --arg llama_commit "$llama_commit" \
  --arg context_size "$context_size" \
  --arg kv_k "$kv_k" \
  --arg kv_v "$kv_v" \
  --arg batch "$batch_size" \
  --arg ubatch "$ubatch_size" \
  --arg gpu0 "$gpu0_vram" \
  --arg gpu1 "$gpu1_vram" \
  '[
    $timestamp,
    $llama_commit,
    $context_size,
    $kv_k,
    $kv_v,
    $batch,
    $ubatch,
    (.tokenized_prompt_tokens // ""),
    (.generated_tokens // ""),
    (.prompt_tps // ""),
    (.generation_tps // ""),
    (.ttft_seconds // ""),
    $gpu0,
    $gpu1
  ] | @csv' "$RESULT_DIR/api-bench.jsonl" >>"$RESULTS_CSV"

cat <<EOF
Benchmark complete.
Raw results: $RESULT_DIR/api-bench.jsonl
CSV results: $RESULTS_CSV
ttft_seconds is measured from the first non-empty native streaming completion chunk.
EOF