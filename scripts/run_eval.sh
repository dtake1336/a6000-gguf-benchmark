#!/usr/bin/env bash
set -euo pipefail
ALIAS=${1:?model alias required}
PORT=${2:-8000}
OUT=${3:?output directory required}
TASKS=${4:-mmlu_pro,gpqa_diamond_zeroshot,hellaswag,arc_challenge,winogrande,gsm8k,ifeval}
LIMIT=${LIMIT:-}
MAX_LENGTH=${MAX_LENGTH:-8192}
mkdir -p "$OUT"

# MCQ/loglikelihood tasks must use the completions endpoint, not chat-completions.
MODEL_ARGS="model=${ALIAS},base_url=http://127.0.0.1:${PORT}/v1/completions,tokenizer_backend=remote,tokenized_requests=true,num_concurrent=1,max_length=${MAX_LENGTH},timeout=1800"

ARGS=(
  run
  --model local-completions
  --model_args "$MODEL_ARGS"
  --tasks "$TASKS"
  --apply_chat_template
  --batch_size 1
  --output_path "$OUT"
  --log_samples
  --confirm_run_unsafe_code
)
if [[ -n "$LIMIT" ]]; then
  ARGS+=(--limit "$LIMIT")
fi

printf '%q ' lm-eval "${ARGS[@]}" > "$OUT/lm-eval.command.txt"
printf '\n' >> "$OUT/lm-eval.command.txt"
lm-eval "${ARGS[@]}" 2>&1 | tee "$OUT/lm-eval.log"
