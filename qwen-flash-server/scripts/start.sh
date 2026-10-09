#!/usr/bin/env bash
set -euo pipefail

: "${MODEL_PATH:?MODEL_PATH is required}"

HOST=${HOST:-0.0.0.0}
PORT=${PORT:-8080}
MODEL_ALIAS=${MODEL_ALIAS:-qwen3.8-flash-next}
CTX_SIZE=${CTX_SIZE:-261888}
PARALLEL=${PARALLEL:-1}
BATCH_SIZE=${BATCH_SIZE:-2048}
UBATCH_SIZE=${UBATCH_SIZE:-256}
CACHE_TYPE_K=${CACHE_TYPE_K:-q8_0}
CACHE_TYPE_V=${CACHE_TYPE_V:-q8_0}
TENSOR_SPLIT=${TENSOR_SPLIT:-1,1}
N_GPU_LAYERS=${N_GPU_LAYERS:-999}
CONT_BATCHING=${CONT_BATCHING:-true}
TEMPERATURE=${TEMPERATURE:-0.2}
TOP_P=${TOP_P:-0.95}

if [[ ! -s "$MODEL_PATH" ]]; then
  echo "Model shard not found or empty: $MODEL_PATH" >&2
  exit 1
fi

echo "Starting Qwen3.8-Flash-Next llama-server" >&2
echo "model=$MODEL_PATH" >&2
echo "host=$HOST port=$PORT alias=$MODEL_ALIAS" >&2
echo "ctx_size=$CTX_SIZE parallel=$PARALLEL batch=$BATCH_SIZE ubatch=$UBATCH_SIZE" >&2
echo "kv_k=$CACHE_TYPE_K kv_v=$CACHE_TYPE_V tensor_split=$TENSOR_SPLIT" >&2
echo "gpu_layers=$N_GPU_LAYERS flash_attn=on ple=CPU" >&2
echo "continuous_batching=$CONT_BATCHING temperature=$TEMPERATURE top_p=$TOP_P" >&2
echo "llama-server version:" >&2
llama-server --version >&2 || true

server_args=(
  -m "$MODEL_PATH"
  --alias "$MODEL_ALIAS"
  --host "$HOST"
  --port "$PORT"
  --n-gpu-layers "$N_GPU_LAYERS"
  --split-mode layer
  --tensor-split "$TENSOR_SPLIT"
  -ot 'per_layer_token_embd.weight=CPU'
  --flash-attn on
  --cache-type-k "$CACHE_TYPE_K"
  --cache-type-v "$CACHE_TYPE_V"
  --ctx-size "$CTX_SIZE"
  --parallel "$PARALLEL"
  --batch-size "$BATCH_SIZE"
  --ubatch-size "$UBATCH_SIZE"
  --jinja
  --temp "$TEMPERATURE"
  --top-p "$TOP_P"
)

case "${CONT_BATCHING,,}" in
  1|true|yes|on) server_args+=(--cont-batching) ;;
  0|false|no|off) ;;
  *) echo "CONT_BATCHING must be a boolean value: $CONT_BATCHING" >&2; exit 2 ;;
esac

exec llama-server "${server_args[@]}"