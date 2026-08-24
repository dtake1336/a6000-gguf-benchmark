# Qwen3.8-27B Q4 / llama.cpp / RTX A6000

## MTP（Multi-Token Prediction）実環境 有効性検証計画

この計画は、リポジトリにある Docker 実行環境、モデル配置、ベンチマークスクリプト、結果保存形式に合わせたものです。

> 今回の実測ではユーザー指定により `--ctx-size 262144`（256K）を使用する。以下に記載するリポジトリ既定の `8192` は比較用の基準値であり、今回の Baseline/MTP 実行には適用しない。

## 0. 元の計画からの重要な変更点

| 項目 | このリポジトリで使用する条件 | 理由 |
|---|---|---|
| 実行環境 | `a6000-gguf-bench:latest` | `Dockerfile` が CUDA 12.4.1 と llama.cpp をまとめてビルドする |
| llama.cpp | `LLAMA_CPP_REF=b10509` | `Makefile` / `Dockerfile` の固定値 |
| GPU | `--gpus device=0`、`--split-mode none` | 既存の 1 GPU 実行方式 |
| Main GGUF | 既定は `Qwen3.8-27B-UD-Q4_K_XL.gguf` | `configs/models.env` と既存結果の実体 |
| Context | 今回は `262144`（256K） | リポジトリ既定は `8192`。今回の実測ではユーザー指定を優先 |
| Batch / UBatch | `2048 / 512` | 現行 llama.cpp の既定値と既存 `llama-bench` 結果 |
| 速度測定 | `scripts/api_bench.py` の native `/completion` | `api-bench.jsonl` に `generation_tps` が保存される |
| サーバーログ | `results/<UTC stamp>/<label>/server.log` | 既存スクリプトと同じ保存形式 |
| MTP 起動 | `--spec-type draft-mtp` | 現行 llama.cpp の方式。外部 `-md` は付けない |

現行イメージの実測 version は次の形式です。

```text
version: 0.1.2-dev (build 10509, commit fe8156f78)
```

現行の `draft-mtp` は、外部 MTP GGUF を読む方式ではなく、メイン GGUF の MTP head/context を使います。したがって元の計画にある次の指定は主検証では使用しません。

```text
-md MTP/mtp-Qwen3.8-27B-Q4_0.gguf
```

`-md` / `--spec-draft-model` は通常の draft model（例えば `draft-simple`）用です。外部ファイルを `-md` でロードできたことを、現行 `draft-mtp` の成功とは扱いません。

## 1. 目的と合格条件

RTX A6000 48 GB 1 枚、CUDA、同じ llama.cpp image、同じ Main GGUF で Baseline と MTP を比較する。

確認する項目は次のとおり。

1. MTP が実際に有効化されているか
2. draft token が生成され、accept されているか
3. MTP なしより decode throughput が向上するか
4. 出力品質、greedy 出力、CUDA 実行に異常がないか

サーバー起動、MTP 引数の受理、モデルロードだけでは成功と判定しない。

### 1.1 MTP 動作の判定

`server.log` に次の形式の行があり、`generated > 0` であることを必須とする。

```text
draft acceptance = 0.12345 (   12 accepted /    40 generated), mean len =  1.30
```

現行ログの authoritative な値は `accepted / generated` である。行がない、または `generated = 0` の場合は `NO` または `UNCONFIRMED` とする。

### 1.2 高速化の判定

現行 `scripts/api_bench.py` で warmup 1 回、測定 5 回を実施する。主比較値は `prompt=4096`、`generate=512` の 5 回の `generation_tps` とし、median を使う。

```text
speedup ratio = MTP median generation_tps / Baseline median generation_tps
speedup % = (speedup ratio - 1) * 100
+5%以上        : YES
0%以上 +5%未満 : MARGINAL
0%未満         : NO
```

5% は本検証の分類値であり、llama.cpp の公式基準ではない。

### 1.3 出力健全性

両条件で次を確認する。

- 文字化け、garbage、無意味な token 列がない
- 異常な反復、明らかな文章崩壊がない
- CUDA error、OOM、NaN、segmentation fault がない
- `temperature=0`、同じ seed、同じ入力の出力を比較する

完全一致しない場合、速度とは別に `lossless性: NO または UNCONFIRMED` とする。

## 2. 結果ディレクトリ

リポジトリルートで実行する。`logs/` は既存のダウンロードログ用なので、この検証の server log には使わない。

```bash
set -euo pipefail

export REPO_ROOT="$(pwd)"
export IMAGE="${IMAGE:-a6000-gguf-bench:latest}"
export MODEL_ROOT="${MODEL_ROOT:-$REPO_ROOT/weights}"
export RESULT_ROOT="${RESULT_ROOT:-$REPO_ROOT/results}"
export GPU="${GPU:-0}"
export PORT="${PORT:-18080}"
export RUN_STAMP="${RUN_STAMP:-$(date -u +%Y%m%dT%H%M%SZ)}"

export BASE_LABEL="qwen38-q4-1gpu"
export MTP_LABEL="qwen38-mtp-n3-p02-1gpu"
export BASE_OUT="$RESULT_ROOT/$RUN_STAMP/$BASE_LABEL"
export MTP_OUT="$RESULT_ROOT/$RUN_STAMP/$MTP_LABEL"
export RUN_OUT="$RESULT_ROOT/$RUN_STAMP"

mkdir -p "$BASE_OUT/bench" "$MTP_OUT"
echo "RUN_OUT=$RUN_OUT"
```

既存形式に合わせた保存先は次のようにする。

```text
results/<UTC timestamp>/
├── environment.txt
├── llama-server-help.txt
├── llama-server-version.txt
├── llama-devices.txt
├── model-manifest.txt
├── correctness-request.json
├── correctness.diff
├── analysis.json
├── qwen38-q4-1gpu/
│   ├── server-command.txt
│   ├── container-inspect.json
│   ├── server.log
│   ├── correctness.json
│   ├── correctness-content.txt
│   ├── api-bench.jsonl
│   ├── api-bench.summary.json
│   ├── gpu-before.csv
│   └── gpu-after.csv
└── qwen38-mtp-n3-p02-1gpu/
    ├── server-command.txt
    ├── container-inspect.json
    ├── server.log
    ├── correctness.json
    ├── correctness-content.txt
    ├── api-bench.jsonl
    ├── api-bench.summary.json
    ├── mtp-draft-stats.txt
    ├── gpu-before.csv
    └── gpu-after.csv
```

同じ `RUN_STAMP` の下に Baseline と MTP を置き、別の測定日・条件と混ぜない。

## 3. Docker と環境の確認

### 3.1 image

既存 image を使う。image がない場合だけ現行 Makefile でビルドする。

```bash
docker image inspect "$IMAGE" >/dev/null 2>&1 || \
  make build LLAMA_CPP_REF=b10509 IMAGE="$IMAGE"
```

この計画では最新 master を別途 clone したり、ローカル `llama.cpp/build` を再利用したりしない。

### 3.2 環境情報

```bash
{
  echo "===== DATE ====="
  date -Is

  echo
  echo "===== HOST ====="
  uname -a

  echo
  echo "===== GPU ====="
  nvidia-smi

  echo
  echo "===== GPU QUERY ====="
  nvidia-smi \
    --query-gpu=name,driver_version,memory.total,memory.used,pstate,power.limit \
    --format=csv,noheader

  echo
  echo "===== GPU TOPOLOGY ====="
  nvidia-smi topo -m

  echo
  echo "===== DOCKER IMAGE ====="
  docker image inspect "$IMAGE" --format '{{.Id}} {{.Created}}'

  echo
  echo "===== LLAMA SERVER VERSION ====="
  docker run --rm --gpus "device=$GPU" "$IMAGE" llama-server --version

  echo
  echo "===== LLAMA DEVICES ====="
  docker run --rm --gpus "device=$GPU" "$IMAGE" llama-server --list-devices

  echo
  echo "===== CUDA IMAGE ====="
  docker run --rm --gpus "device=$GPU" "$IMAGE" bash -lc 'nvcc --version || true'
} 2>&1 | tee "$RUN_OUT/environment.txt"

docker run --rm --gpus "device=$GPU" "$IMAGE" llama-server --version \
  2>&1 | tee "$RUN_OUT/llama-server-version.txt"

docker run --rm --gpus "device=$GPU" "$IMAGE" llama-server --list-devices \
  2>&1 | tee "$RUN_OUT/llama-devices.txt"

docker run --rm --gpus "device=$GPU" "$IMAGE" llama-server --help \
  > "$RUN_OUT/llama-server-help.txt" 2>&1

grep -E -- \
  '--spec-type|--spec-draft-model|--spec-draft-n-max|--spec-draft-p-min|--spec-draft-ngl' \
  "$RUN_OUT/llama-server-help.txt"
```

`NVIDIA RTX A6000`、CUDA backend、GPU 0 が確認できない場合は測定を開始しない。他の GPU workload があれば記録し、可能なら停止する。

## 4. モデル確認

### 4.1 現行の既定 Main GGUF

```bash
# shellcheck disable=SC1091
source "$REPO_ROOT/configs/models.env"

export MAIN_CONTAINER="$QWEN38_Q4_MODEL"
export MAIN_HOST="$MODEL_ROOT/${MAIN_CONTAINER#/models/}"

test -s "$MAIN_HOST"

{
  echo "main_container=$MAIN_CONTAINER"
  echo "main_host=$MAIN_HOST"
  stat -c '%n %s bytes' "$MAIN_HOST"
  sha256sum "$MAIN_HOST"
} | tee "$RUN_OUT/model-manifest.txt"
```

現行設定の既定値は次のファイルである。

```text
weights/qwen38-27b-mtp/Qwen3.8-27B-UD-Q4_K_XL.gguf
```

元の計画にある Q4_K_M は現行設定の既定ではない。Q4_K_M を対象にする場合は、実在する Q4_K_M を別途配置し、`MAIN_CONTAINER` と `MAIN_HOST` をそのファイルに変更する。Q4_K_XL の既存結果を Q4_K_M の Baseline として流用しない。

### 4.2 外部 MTP GGUF

元の計画にある次のファイルは、現行 `draft-mtp` 主検証ではロードしない。

```text
MTP/mtp-Qwen3.8-27B-Q4_0.gguf
```

参考 artifact として存在する場合だけ hash を記録する。

```bash
export MTP_REFERENCE_HOST="$MODEL_ROOT/qwen38-27b-mtp/MTP/mtp-Qwen3.8-27B-Q4_0.gguf"

if test -s "$MTP_REFERENCE_HOST"; then
  {
    echo "external_mtp_reference=$MTP_REFERENCE_HOST"
    stat -c '%n %s bytes' "$MTP_REFERENCE_HOST"
    sha256sum "$MTP_REFERENCE_HOST"
  } | tee -a "$RUN_OUT/model-manifest.txt"
else
  echo "external_mtp_reference=not present; not required by current draft-mtp path" \
    | tee -a "$RUN_OUT/model-manifest.txt"
fi
```

最終判定は外部ファイルの有無ではなく、Main GGUF を `--spec-type draft-mtp` で実行した draft 統計で行う。

## 5. 共通実行条件

Baseline と MTP の差は speculative decoding の有無と MTP パラメータだけにする。

```text
GPU                 : device 0, NVIDIA RTX A6000 48 GB
GPU split            : none
Main GPU layers      : 99
Context              : 262144（今回の実測。リポジトリ既定は8192）
Batch / UBatch       : 2048 / 512
Parallel             : 1
Flash Attention      : on
Fit                  : off（両条件で明示）
Jinja                : on
Reasoning            : off
Temperature / seed   : 0 / 42
Host / port          : 0.0.0.0:8080 / host port $PORT
```

`run_docker_bench.sh` は server の `--fit off` を省略しているが、現行 binary の default は `fit=on` である。この検証では MTP context 追加による自動調整差を避けるため、両条件に `--fit off` を明示する。

既存 `scripts/api_bench.py` の設定は次のとおりである。

```text
prompt tokens : 512, 2048, 4096
generation    : 128, 512
warmups       : 1 per combination
repetitions   : 5 per combination
API           : native /completion
cache_prompt  : false
ignore_eos    : true
```

`api-bench.summary.json` は mean/stdev であり median を持たない。median は raw `api-bench.jsonl` の `generation_tps` から計算する。

## 6. correctness request

速度測定は native `/completion`、出力健全性は既存 README と同じ OpenAI 互換 `/v1/chat/completions` を使う。

```bash
export MODEL_ALIAS="qwen38-mtp-validation"

cat > "$RUN_OUT/correctness-request.json" <<JSON
{
  "model": "$MODEL_ALIAS",
  "messages": [
    {
      "role": "user",
      "content": "Explain what happens when a Linux process reads a file that is not currently in the page cache. Give exactly ten numbered steps."
    }
  ],
  "temperature": 0,
  "top_p": 1.0,
  "seed": 42,
  "max_tokens": 512,
  "stream": false
}
JSON
```

## 7. サーバー共通処理

次の処理は、既存スクリプトと同じく Main model を `/models:ro` に、結果をホストの `results/` に置く。`server.log` は container 停止前に `docker logs` から保存する。

```bash
COMMON_SERVER_ARGS=(
  llama-server
  -m "$MAIN_CONTAINER"
  --alias "$MODEL_ALIAS"
  --host 0.0.0.0
  --port 8080
  --n-gpu-layers 99
  --ctx-size 262144
  --batch-size 2048
  --ubatch-size 512
  --flash-attn on
  --parallel 1
  --fit off
  --jinja
  --reasoning off
  --temp 0
  --seed 42
  --split-mode none
)

start_server() {
  local name="$1"
  local out="$2"
  local mode="$3"
  local extra=()

  if [[ "$mode" == "mtp" ]]; then
    extra=(
      --spec-type draft-mtp
      --spec-draft-n-max 3
      --spec-draft-p-min 0.2
      --spec-draft-ngl all
    )
  fi

  docker rm -f "$name" >/dev/null 2>&1 || true
  printf '%q ' "${COMMON_SERVER_ARGS[@]}" "${extra[@]}" > "$out/server-command.txt"
  printf '\n' >> "$out/server-command.txt"

  docker run --detach \
    --gpus "device=$GPU" \
    --ipc=host \
    --name "$name" \
    -v "$MODEL_ROOT:/models:ro" \
    -p "$PORT:8080" \
    "$IMAGE" \
    "${COMMON_SERVER_ARGS[@]}" "${extra[@]}" \
    > "$out/container-id.txt"

  local ready=0
  for _ in $(seq 1 300); do
    if curl -fsS "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; then
      ready=1
      break
    fi
    if [[ "$(docker inspect -f '{{.State.Running}}' "$name" 2>/dev/null || true)" != "true" ]]; then
      docker logs "$name" 2>&1 | tail -n 200
      return 1
    fi
    sleep 2
  done

  [[ "$ready" == 1 ]] || {
    docker logs "$name" 2>&1 | tail -n 300
    return 1
  }

  docker inspect "$name" > "$out/container-inspect.json"
  curl -fsS "http://127.0.0.1:$PORT/health"
}

stop_server() {
  local name="$1"
  local out="$2"

  docker logs "$name" > "$out/server.log" 2>&1 || true
  docker inspect "$name" > "$out/container-inspect.json" 2>/dev/null || true
  docker rm -f "$name" >/dev/null 2>&1 || true
}

run_api_bench() {
  local label="$1"

  docker run --rm \
    --network=host \
    -v "$REPO_ROOT/scripts:/scripts:ro" \
    -v "$RESULT_ROOT:/results" \
    "$IMAGE" \
    python3 /scripts/api_bench.py \
      --base-url "http://127.0.0.1:$PORT" \
      --prompt-tokens 512,2048,4096 \
      --generate-tokens 128,512 \
      --repetitions 5 \
      --warmups 1 \
      --timeout 1800 \
      --output "/results/$RUN_STAMP/$label/api-bench.jsonl"
}
```

MTP 起動時に `-md`、`--model-draft`、`--spec-draft-model` を追加しない。

## 8. Baseline

```bash
export BASE_CONTAINER="a6000-qwen38-mtp-baseline-$PORT"

nvidia-smi \
  --query-gpu=name,memory.used,memory.total,utilization.gpu,pstate,power.draw \
  --format=csv,noheader > "$BASE_OUT/gpu-before.csv"

start_server "$BASE_CONTAINER" "$BASE_OUT" baseline

curl --fail --silent --show-error \
  "http://127.0.0.1:$PORT/v1/chat/completions" \
  -H 'Content-Type: application/json' \
  --data-binary @"$RUN_OUT/correctness-request.json" \
  > "$BASE_OUT/correctness.json"

jq -e '.choices[0].message.content != null' "$BASE_OUT/correctness.json" >/dev/null
jq -r '.choices[0].message.content' \
  "$BASE_OUT/correctness.json" > "$BASE_OUT/correctness-content.txt"

run_api_bench "$BASE_LABEL"

stop_server "$BASE_CONTAINER" "$BASE_OUT"

nvidia-smi \
  --query-gpu=name,memory.used,memory.total,utilization.gpu,pstate,power.draw \
  --format=csv,noheader > "$BASE_OUT/gpu-after.csv"
```

Baseline correctness が正常でない場合は MTP に進まない。`BASE_OUT/server.log` と `diagnostics.txt` に原因を残す。

### 8.1 任意の llama-bench

`llama-bench` は kernel-oriented な参考値であり、MTP on/off の主比較には使わない。必要なら既存形式で Baseline だけ保存する。

```bash
docker run --rm \
  --gpus "device=$GPU" \
  --ipc=host \
  -v "$MODEL_ROOT:/models:ro" \
  "$IMAGE" \
  llama-bench \
    -m "$MAIN_CONTAINER" \
    -ngl 99 -fa on -sm none \
    -r 5 -o json \
    -p 512,2048,8192 \
    -n 128,512 \
    > "$BASE_OUT/bench/result.json" \
    2> "$BASE_OUT/bench/stderr.log"
```

## 9. MTP

### 9.1 起動条件

```text
spec type       : draft-mtp
n_max           : 3
p_min           : 0.2
draft ngl       : all
external -md    : なし
```

```bash
export MTP_CONTAINER="a6000-qwen38-mtp-$PORT"

nvidia-smi \
  --query-gpu=name,memory.used,memory.total,utilization.gpu,pstate,power.draw \
  --format=csv,noheader > "$MTP_OUT/gpu-before.csv"

start_server "$MTP_CONTAINER" "$MTP_OUT" mtp

docker logs "$MTP_CONTAINER" 2>&1 \
  | grep -iE 'mtp|spec|model|CUDA|failed|error|warning' \
  | tail -n 200 || true
```

MTP context の生成に失敗したら `-md` を足して続行しない。`MTP unsupported for this main GGUF / llama.cpp build` として記録する。

### 9.2 correctness と速度

```bash
curl --fail --silent --show-error \
  "http://127.0.0.1:$PORT/v1/chat/completions" \
  -H 'Content-Type: application/json' \
  --data-binary @"$RUN_OUT/correctness-request.json" \
  > "$MTP_OUT/correctness.json"

jq -e '.choices[0].message.content != null' "$MTP_OUT/correctness.json" >/dev/null
jq -r '.choices[0].message.content' \
  "$MTP_OUT/correctness.json" > "$MTP_OUT/correctness-content.txt"

run_api_bench "$MTP_LABEL"

stop_server "$MTP_CONTAINER" "$MTP_OUT"

grep -iE \
  'draft acceptance|statistics .*draft|eval time|tokens per second|error|warning|CUDA|NaN|out of memory' \
  "$MTP_OUT/server.log" \
  > "$MTP_OUT/mtp-draft-stats.txt" || true

nvidia-smi \
  --query-gpu=name,memory.used,memory.total,utilization.gpu,pstate,power.draw \
  --format=csv,noheader > "$MTP_OUT/gpu-after.csv"
```

### 9.3 現行 log で見るべき行

```text
draft acceptance = <rate> (<accepted> accepted / <generated> generated), mean len = <len>
```

古い計画にある `draft acceptance rate = ...` だけを検索しない。現行ログは `draft acceptance = ...` である。

## 10. 出力比較

```bash
jq -r '.choices[0].message.content' \
  "$BASE_OUT/correctness.json" > "$BASE_OUT/correctness-content.txt"

jq -r '.choices[0].message.content' \
  "$MTP_OUT/correctness.json" > "$MTP_OUT/correctness-content.txt"

diff -u \
  "$BASE_OUT/correctness-content.txt" \
  "$MTP_OUT/correctness-content.txt" \
  > "$RUN_OUT/correctness.diff" || true

if cmp -s \
  "$BASE_OUT/correctness-content.txt" \
  "$MTP_OUT/correctness-content.txt"; then
  echo "correctness: exact match"
else
  echo "correctness: different output; inspect $RUN_OUT/correctness.diff"
fi
```

両方が正常な文章でも完全一致しなければ `lossless性: UNCONFIRMED` とする。MTP 側だけが壊れていれば `NO` とする。

## 11. 速度・draft 統計の解析

以下は、現行 `api_bench.py` の JSONL と現行 server log に合わせた最小解析である。`4096 / 512` はスクリプトの最後の組み合わせなので、MTP log の最後の 5 件を測定値とする（その直前の 1 件は warmup）。

```bash
cat > "$RUN_OUT/analyze_mtp.py" <<'PY'
from __future__ import annotations

import json
import re
import statistics
import sys
from pathlib import Path

if len(sys.argv) != 5:
    raise SystemExit(f"usage: {sys.argv[0]} BASE_JSONL MTP_JSONL MTP_LOG OUTPUT_JSON")

base_jsonl, mtp_jsonl, mtp_log, output_json = map(Path, sys.argv[1:])

def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]

def primary_tps(path: Path) -> list[float]:
    rows = read_jsonl(path)
    values = [
        float(row["generation_tps"])
        for row in rows
        if row.get("target_prompt_tokens") == 4096
        and row.get("requested_generation_tokens") == 512
        and row.get("generation_tps") is not None
    ]
    if len(values) != 5:
        raise RuntimeError(f"expected 5 primary rows in {path}, got {len(values)}")
    return values

draft_re = re.compile(
    r"draft acceptance =\s*([0-9.]+)\s*"
    r"\(\s*(\d+)\s*accepted\s*/\s*(\d+)\s*generated\),\s*"
    r"mean len =\s*([0-9.]+)"
)

def draft_stats(path: Path) -> list[dict]:
    text = path.read_text(errors="replace")
    rows = [
        {
            "reported_acceptance_rate": float(rate),
            "accepted": int(accepted),
            "generated": int(generated),
            "mean_len": float(mean_len),
        }
        for rate, accepted, generated, mean_len in draft_re.findall(text)
    ]
    measured = rows[-5:]
    if len(measured) != 5:
        raise RuntimeError(
            f"expected 5 draft acceptance rows for primary block in {path}, got {len(measured)}"
        )
    return measured

base = primary_tps(base_jsonl)
mtp = primary_tps(mtp_jsonl)
stats = draft_stats(mtp_log)

base_median = statistics.median(base)
mtp_median = statistics.median(mtp)
ratio = mtp_median / base_median
speedup_pct = (ratio - 1.0) * 100.0

accepted = sum(row["accepted"] for row in stats)
generated = sum(row["generated"] for row in stats)
acceptance_rate = accepted / generated if generated else 0.0

result = {
    "primary": {
        "prompt_tokens": 4096,
        "requested_generation_tokens": 512,
        "baseline_generation_tps": base,
        "mtp_generation_tps": mtp,
        "baseline_median_generation_tps": base_median,
        "mtp_median_generation_tps": mtp_median,
        "speedup_ratio": ratio,
        "speedup_percent": speedup_pct,
    },
    "mtp": {
        "draft_stats_used": stats,
        "draft_accepted": accepted,
        "draft_generated": generated,
        "draft_acceptance_rate": acceptance_rate,
        "mean_len_arithmetic_mean": (
            statistics.fmean(row["mean_len"] for row in stats) if stats else None
        ),
        "active": generated > 0,
    },
}

output_json.write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps(result, indent=2))
PY

docker run --rm \
  -v "$RUN_OUT:/work" \
  "$IMAGE" \
  python3 /work/analyze_mtp.py \
    "/work/$BASE_LABEL/api-bench.jsonl" \
    "/work/$MTP_LABEL/api-bench.jsonl" \
    "/work/$MTP_LABEL/server.log" \
    "/work/analysis.json"
```

解析後に次を確認する。

```bash
jq '.primary, .mtp' "$RUN_OUT/analysis.json"
```

`mean len` は現行 llama.cpp の `1 + accepted / verification steps` であり、単純な「draft token の平均生成数」とは意味が異なる。報告では `llama.cpp log mean len` と明記し、draft の実数は `draft_generated`、accept 数は `draft_accepted` と分ける。

## 12. エラー・GPU 診断

```bash
grep -iE \
  'error|warning|cuda|nan|oom|out of memory|segmentation fault|failed' \
  "$BASE_OUT/server.log" > "$BASE_OUT/diagnostics.txt" || true

grep -iE \
  'error|warning|cuda|mtp|draft|accept|nan|oom|out of memory|segmentation fault|failed' \
  "$MTP_OUT/server.log" > "$MTP_OUT/diagnostics.txt" || true

nvidia-smi dmon -s pucm -d 1
```

他プロセスが GPU を使用した run、OOM で短縮された run、`generated_tokens` が要求数と異常に違う run は通常の集計から外し、理由を記載する。

## 13. MTP が有効でも高速化しない場合

次の場合だけ追加チューニングする。

```text
draft_generated > 0 だが speedup < +5%
または acceptance rate が極端に低い
```

同じ Main GGUF、Context、Batch、UBatch、Parallel、FA、temperature、seed を保ったまま、次を候補にする。

```text
n_max: 2, 3, 4
p_min: 0.0, 0.2, 0.5
```

各組み合わせは別 label の `results/<stamp>/<label>/` に保存し、warmup 1 回、測定 3 回以上とする。`-md` を追加した別方式を混ぜない。

優先順位は次のとおり。

```text
1. n_max=3, p_min=0.0
2. n_max=2, p_min=0.2
3. n_max=4, p_min=0.2
4. n_max=2, p_min=0.0
5. n_max=4, p_min=0.0
6. p_min=0.5 系
```

acceptance がほぼ 0 の場合は `p_min=0.0` を必ず確認する。それでも `generated > 0` かつ高速化しないなら、「この GPU・モデル・llama.cpp build の組み合わせでは実用的な高速化なし」と結論する。

## 14. 最終報告フォーマット

```text
# Qwen3.8-27B MTP Validation Result

## Environment

GPU:
VRAM:
Driver:
CUDA image:
llama.cpp build/version:
llama.cpp commit:
Docker image:
Run stamp:

## Models

Main model:
Main SHA256:
Main size:
External MTP artifact: N/A (current draft-mtp uses MTP heads/context from main GGUF)
External MTP reference SHA256: if present

## Common conditions

Context:
Batch / UBatch:
Parallel:
GPU layers:
Flash Attention:
Fit:
Reasoning:
Temperature / seed:
Primary API slice: prompt=4096, generation=512

## Baseline Performance

Run 1 generation_tps:
Run 2 generation_tps:
Run 3 generation_tps:
Run 4 generation_tps:
Run 5 generation_tps:
Median:
VRAM:

## MTP Performance

spec-type: draft-mtp
n_max: 3
p_min: 0.2
draft ngl: all

Run 1 generation_tps:
Run 2 generation_tps:
Run 3 generation_tps:
Run 4 generation_tps:
Run 5 generation_tps:
Median:
VRAM:

draft generated:
draft accepted:
draft acceptance rate: accepted / generated
llama.cpp mean len: arithmetic mean of reported mean len

## Comparison

speedup ratio:
speedup percent:
MTP active: YES / NO / UNCONFIRMED
MTP practically effective: YES / NO / MARGINAL
Baseline/MTP output exact match: YES / NO
Output health: PASS / FAIL
CUDA/OOM/NaN/segfault: none / details
Lossless behavior: YES / NO / UNCONFIRMED

## Final judgment

MTP is working in this environment: YES / NO
Recommended production setting:
Problems:

## Raw artifacts

results/<stamp>/environment.txt
results/<stamp>/llama-server-version.txt
results/<stamp>/llama-server-help.txt
results/<stamp>/model-manifest.txt
results/<stamp>/qwen38-q4-1gpu/server.log
results/<stamp>/qwen38-q4-1gpu/api-bench.jsonl
results/<stamp>/qwen38-q4-1gpu/api-bench.summary.json
results/<stamp>/qwen38-mtp-n3-p02-1gpu/server.log
results/<stamp>/qwen38-mtp-n3-p02-1gpu/api-bench.jsonl
results/<stamp>/qwen38-mtp-n3-p02-1gpu/api-bench.summary.json
results/<stamp>/qwen38-mtp-n3-p02-1gpu/mtp-draft-stats.txt
results/<stamp>/analysis.json
results/<stamp>/correctness.diff
```

## 15. 禁止事項

- Baseline と MTP で Context、Batch、UBatch、GPU offload、Parallel、FA、sampling、seed を変えない
- Q4_K_XL の既存結果を Q4_K_M の結果として扱わない
- `draft-mtp` に `-md` を足して外部 MTP model のロード成功を MTP 成功と扱わない
- `api-bench.summary.json` の mean だけで median として報告しない
- サーバー起動成功だけで MTP 成功と判定しない
- `draft acceptance` と `generated > 0` を確認せず MTP 成功と判定しない
- 1 回の tok/s だけで高速化を結論しない
- 異常出力、CUDA error、OOM、NaN、segmentation fault を速度結果から隠さない

最終的に必要なのは、次の数値と事実である。

```text
1. draft token を何 token generated したか
2. 何 token accepted したか
3. accepted / generated はいくつか
4. Baseline の median generation_tps はいくつか
5. MTP の median generation_tps はいくつか
6. 何 % 高速化または低速化したか
7. 同一 greedy 入力で出力差が発生したか
8. A6000 48 GB で推奨できる設定か
```
