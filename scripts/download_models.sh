#!/usr/bin/env bash
set -euo pipefail

# Load HF_TOKEN from .env if present (for non-Docker usage)
REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
if [[ -f "$REPO_ROOT/.env" ]]; then
  # shellcheck disable=SC1090
  export $(grep -v '^#' "$REPO_ROOT/.env" | xargs) 2>/dev/null || true
fi

MODEL_ROOT=${MODEL_ROOT:-/models}
TARGET=${1:-recommended}
mkdir -p "$MODEL_ROOT"

qwen36_35b_q4km() {
  hf download unsloth/Qwen3.6-35B-A3B-MTP-GGUF \
    Qwen3.6-35B-A3B-UD-Q4_K_M.gguf \
    --local-dir "$MODEL_ROOT/qwen36-35b-a3b-mtp"
}

qwen_q4() {
  hf download unsloth/Qwen3.6-27B-MTP-GGUF \
    Qwen3.6-27B-UD-Q4_K_XL.gguf \
    --local-dir "$MODEL_ROOT/qwen36-27b-mtp"
}

qwen_q6() {
  hf download unsloth/Qwen3.6-27B-MTP-GGUF \
    Qwen3.6-27B-UD-Q6_K_XL.gguf \
    --local-dir "$MODEL_ROOT/qwen36-27b-mtp"
}

qwen_q8() {
  hf download unsloth/Qwen3.6-27B-MTP-GGUF \
    Qwen3.6-27B-UD-Q8_K_XL.gguf \
    --local-dir "$MODEL_ROOT/qwen36-27b-mtp"
}

qwen38_q4() {
  hf download unsloth/Qwen3.8-27B-GGUF \
    Qwen3.8-27B-UD-Q4_K_XL.gguf \
    --local-dir "$MODEL_ROOT/qwen38-27b-mtp"
}

nemotron_q4ks() {
  hf download \
    hf://unsloth/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-GGUF/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-UD-Q4_K_S.gguf \
    --local-dir "$MODEL_ROOT/nemotron-3.5-lightning-30b-a3b"
}

laguna_q4xl() {
  hf download unsloth/Laguna-S-2.1-GGUF \
    --include 'UD-Q4_K_XL/*' \
    --local-dir "$MODEL_ROOT/laguna-s-2.1"
}

laguna_iq4nl() {
  hf download unsloth/Laguna-S-2.1-GGUF \
    --include 'UD-IQ4_NL/*' \
    --local-dir "$MODEL_ROOT/laguna-s-2.1"
}

laguna_q4m() {
  hf download unsloth/Laguna-S-2.1-GGUF \
    --include 'Q4_K_M/*' \
    --local-dir "$MODEL_ROOT/laguna-s-2.1"
}

case "$TARGET" in
  recommended) qwen_q4; qwen36_35b_q4km; laguna_q4xl; nemotron_q4ks ;;
  quality)     qwen_q4; qwen_q6; qwen36_35b_q4km; laguna_q4xl; laguna_q4m; nemotron_q4ks ;;
  all)         qwen_q4; qwen_q6; qwen_q8; qwen38_q4; qwen36_35b_q4km; nemotron_q4ks; laguna_q4xl; laguna_iq4nl; laguna_q4m ;;
  qwen-q4)     qwen_q4 ;;
  qwen-q6)     qwen_q6 ;;
  qwen-q8)     qwen_q8 ;;
  qwen38-q4)   qwen38_q4 ;;
  qwen36-35b-q4km) qwen36_35b_q4km ;;
  nemotron-q4ks) nemotron_q4ks ;;
  laguna-q4xl) laguna_q4xl ;;
  laguna-iq4nl) laguna_iq4nl ;;
  laguna-q4m)  laguna_q4m ;;
  *)
    echo "Usage: $0 {recommended|quality|all|qwen-q4|qwen-q6|qwen-q8|qwen38-q4|qwen36-35b-q4km|nemotron-q4ks|laguna-q4xl|laguna-iq4nl|laguna-q4m}" >&2
    exit 2
    ;;
esac
