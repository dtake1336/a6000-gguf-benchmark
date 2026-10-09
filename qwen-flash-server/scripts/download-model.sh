#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "$SCRIPT_DIR/../.." && pwd)
MODEL_ROOT=${MODEL_ROOT:-$REPO_ROOT/weights}
MODEL_DIR=${MODEL_DIR:-qwen38-flash-next}
HF_REPO=${HF_REPO:-unsloth/Qwen3.8-Flash-Next-GGUF}
HF_QUANT=${HF_QUANT:-UD-IQ4_XS}
TARGET_DIR="$MODEL_ROOT/$MODEL_DIR"

command -v hf >/dev/null 2>&1 || {
  echo "hf is required. Run this script in the existing a6000-gguf-bench image or install huggingface_hub." >&2
  exit 127
}

mkdir -p "$TARGET_DIR"
echo "Downloading $HF_REPO ($HF_QUANT) to $TARGET_DIR" >&2
hf download "$HF_REPO" \
  --include "*${HF_QUANT}*.gguf" \
  --exclude ".gitattributes" \
  --local-dir "$TARGET_DIR"

shopt -s nullglob
shards=("$TARGET_DIR"/*"$HF_QUANT"*.gguf)
if [[ ${#shards[@]} -eq 0 ]]; then
  echo "No ${HF_QUANT} GGUF files were downloaded to $TARGET_DIR" >&2
  exit 1
fi

printf 'Downloaded %d GGUF file(s):\n' "${#shards[@]}"
printf '  %s\n' "${shards[@]##*/}"
echo "Set MODEL_PATH to the first shard when starting the server." >&2