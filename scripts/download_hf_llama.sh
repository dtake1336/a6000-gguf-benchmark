#!/usr/bin/env bash
# Download a GGUF model from Hugging Face using the llama-cli -hf style spec:
#   scripts/download_hf_llama.sh <repo>:<quant>
# Examples:
#   scripts/download_hf_llama.sh unsloth/Qwen3.6-35B-A3B-MTP-GGUF:UD-Q4_K_M
#   scripts/download_hf_llama.sh unsloth/Llama-3.3-70B-Instruct-GGUF:Q4_K_M
set -euo pipefail

# Load HF_TOKEN from .env if present (for non-Docker usage)
REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
if [[ -f "$REPO_ROOT/.env" ]]; then
  # shellcheck disable=SC1090
  export $(grep -v '^#' "$REPO_ROOT/.env" | xargs) 2>/dev/null || true
fi

MODEL_ROOT=${MODEL_ROOT:-/models}
SPEC=${1:?"Usage: $0 <repo>:<quant_pattern>"}
mkdir -p "$MODEL_ROOT"

# Parse repo:pattern
repo="${SPEC%%:*}"
pattern="${SPEC#*:}"

# Derive output directory from the repo name (last path component)
subdir=$(basename "$repo")
out_dir="$MODEL_ROOT/${subdir,,}"  # lowercase directory name for consistency

# Download matching files using hf download (from huggingface_hub[cli])
echo "Downloading from $repo matching '*${pattern}*' to $out_dir ..."
hf download "$repo" \
  --include "*${pattern}*" \
  --exclude ".gitattributes" \
  --local-dir "$out_dir"

echo "Done. Files saved to $out_dir"
