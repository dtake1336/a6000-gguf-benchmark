# A6000 GGUF benchmark

`llama.cpp` の CUDA 対応Dockerイメージを作成し、GGUFのweightをホスト側で
別途取得して、実行時にvolumeとして読み込ませるベンチマーク環境です。
以下の手順はリポジトリのルートディレクトリで、上から順にコピーして実行
できます。

## 1. Docker imageをbuildする

```bash
export IMAGE=a6000-gguf-bench:latest
export MODEL_ROOT="$PWD/weights"
export GPU=0
export PORT=18080
export SERVER_NAME=a6000-llama-server
export MODEL_PATH=/models/qwen36-27b-mtp/Qwen3.6-27B-UD-Q4_K_XL.gguf

docker build \
  --build-arg CUDA_IMAGE=nvidia/cuda:12.4.1-devel-ubuntu22.04 \
  --build-arg LLAMA_CPP_REF=b10509 \
  --build-arg LM_EVAL_REF=v0.4.12 \
  -t "$IMAGE" .
```

## 2. モデルweightをダウンロードする

### Hugging Faceトークンの設定（推奨）

レート制限の回避や非公開モデルのダウンロードには、Hugging Faceの
アクセストークンが必要です。リポジトリルートの `.env` にトークンを
記入してください。

```bash
# .env
HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
```

トークンは [https://huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)
から取得できます。`.env` は `.gitignore` に含まれているため、
誤ってコミットされることはありません。

設定後、`make download` や手動の `docker run` コマンドで `--env-file .env`
または `-e HF_TOKEN` を付けることで、コンテナ内の `hf download` コマンドが
認証済みリクエストを使うようになります。

Dockerイメージにはweightを含めません。次のコマンドでQwen Q4を
`weights/`へ取得します。`recommended`、`quality`、`all`、個別モデル名を
指定できます。

```bash
docker run --rm \
  -v "$PWD:/workspace" \
  -w /workspace \
  --env-file "$PWD/.env" \
  "$IMAGE" \
  bash -lc 'MODEL_ROOT=/workspace/weights scripts/download_models.sh qwen-q4'
```

Nemotronだけを取得する場合:

```bash
docker run --rm \
  -v "$PWD:/workspace" \
  -w /workspace \
  --env-file "$PWD/.env" \
  "$IMAGE" \
  bash -lc 'MODEL_ROOT=/workspace/weights scripts/download_models.sh nemotron-q4ks'
```

`recommended`にはQwen Q4、Qwen3.6-35B Q4_K_M、Laguna Q4 XL、Nemotron Q4_K_Sが含まれ、
`quality`にはそれらに加えてQwen Q6とLaguna Q4 K-Mが含まれます。

別の保存先を使う場合は、ホスト側のディレクトリを `/models` にマウントして
実行します。

```bash
docker run --rm \
  -v "$PWD:/workspace:ro" \
  -v "/shared/models/gguf:/models" \
  -w /workspace \
  --env-file "$PWD/.env" \
  "$IMAGE" \
  bash -lc 'MODEL_ROOT=/models scripts/download_models.sh recommended'
```

### 任意のモデルを `llama-cli -hf` 形式でダウンロードする

`scripts/download_hf_llama.sh` を使うと、`repo:quant` 形式で未登録のモデルも
取得できます。`llama-cli -hf unsloth/Qwen3.6-35B-A3B-MTP-GGUF:UD-Q4_K_M` と
同じ指定方法でダウンロードできます。

```bash
docker run --rm \
  -v "$PWD:/workspace" \
  -v "$PWD/weights:/models" \
  -w /workspace \
  --env-file "$PWD/.env" \
  "$IMAGE" \
  bash -lc 'MODEL_ROOT=/models scripts/download_hf_llama.sh unsloth/Qwen3.6-35B-A3B-MTP-GGUF:UD-Q4_K_M'
```

またはMakefileから:

```bash
make download-hf HF_SPEC=unsloth/Qwen3.6-35B-A3B-MTP-GGUF:UD-Q4_K_M
```

## 3. Dockerでllama-serverを起動する

weightのあるディレクトリを `/models` へ読み取り専用でマウントします。

```bash
docker rm -f "$SERVER_NAME" 2>/dev/null || true

docker run --detach \
  --gpus "device=$GPU" \
  --ipc=host \
  --name "$SERVER_NAME" \
  -v "$MODEL_ROOT:/models:ro" \
  -p "$PORT:8080" \
  "$IMAGE" \
  llama-server \
    -m "$MODEL_PATH" \
    --alias qwen36-q4 \
    --host 0.0.0.0 \
    --port 8080 \
    --n-gpu-layers 99 \
    --ctx-size 8192 \
    --flash-attn on \
    --parallel 1 \
    --jinja \
    --reasoning off \
    --temp 0 \
    --seed 42 \
    --split-mode none
```

起動確認:

```bash
curl -fsS "http://127.0.0.1:$PORT/health"
```

## 4. APIをテストする

まずOpenAI互換APIの疎通を確認し、その後にprompt長・生成長ごとの計測を
実行します。結果は `results/manual/` に保存されます。

```bash
curl --fail --silent --show-error \
  -H 'Content-Type: application/json' \
  -d '{"model":"qwen36-q4","messages":[{"role":"user","content":"Say hello in one sentence."}],"temperature":0,"max_tokens":32}' \
  "http://127.0.0.1:$PORT/v1/chat/completions"

mkdir -p results/manual
docker run --rm \
  --network=host \
  -v "$PWD/scripts:/scripts:ro" \
  -v "$PWD/results/manual:/results" \
  "$IMAGE" \
  python3 /scripts/api_bench.py \
    --base-url "http://127.0.0.1:$PORT" \
    --prompt-tokens 512,2048,4096 \
    --generate-tokens 128,512 \
    --repetitions 5 \
    --warmups 1 \
    --timeout 1800 \
    --output /results/api-bench.jsonl
```

テスト終了後はサーバーを停止します。

```bash
docker rm -f "$SERVER_NAME"
```

## スクリプトの役割

スクリプトは、処理の入口ごとに4本へ整理しています。

| ファイル | 役割 |
|---|---|
| `scripts/download_models.sh` | Hugging Face HubからGGUFと必要な全シャードを取得する |
| `scripts/download_hf_llama.sh` | `llama-cli -hf` と同じ `repo:quant` 形式で任意のGGUFを取得する |
| `scripts/run_docker_bench.sh` | weight volumeをマウントし、`llama-bench`、`llama-server`、API計測を一括実行する |
| `scripts/api_bench.py` | llama-serverのcompletion APIをprompt長・生成長別に計測し、JSONLとsummaryを保存する |
| `scripts/run_eval.sh` | 起動済みllama-serverに対してlm-evaluation-harnessで品質評価を行う任意のスクリプト |

個別のサーバー起動・待機・HTTP計測・行列実行用の補助スクリプトは削除し、
サーバーのライフサイクルは `run_docker_bench.sh`へ集約しています。

## Makefileのショートカット

上記の処理はMakefileからも実行できます。

```bash
make build
make download DOWNLOAD_SET=recommended
GPU=0 make bench
```

既定のベンチマークはQwen Q4を1 GPUで実行します。別モデルは次のキーを
使用できます。

- `qwen_q4`、`qwen_q6`、`qwen_q8`
- `qwen38_q4`
- `qwen36_35b_q4km`
- `nemotron_q4ks`
- `laguna_q4xl`、`laguna_q4m`
- `laguna_iq4nl`

```bash
GPU=0,1 GPU_MODE=2gpu-layer make bench
GPU=0,1 GPU_MODE=2gpu-row make bench
GPU=0 make bench-qwen38
GPU=0 make bench-qwen36-35b
GPU=0 make bench-nemotron
make bench-laguna-iq4nl  # デフォルトでGPU 0,1をlayer-splitで使用
```

Qwen3.6-35Bを常駐serverとして起動する場合:

```bash
GPU=0 PORT=18080 make serve-qwen36-35b
curl -fsS http://127.0.0.1:18080/health
```

serverの停止:

```bash
docker rm -f a6000-qwen36-35b-server
```

モデルパスは `configs/models.env`、lm-evaluationのタスクセットは
`configs/suites.env`で管理します。GGUFの実体はGit管理せず、
`MODEL_ROOT`で指定したホストディレクトリに置きます。

## Docker imageの方針

`Dockerfile`はllama.cpp、llama-server、llama-bench、Python、
lm-evaluation-harnessなどの実行環境だけを構築します。`.dockerignore`で
`weights/`とGGUFをbuild contextから除外しているため、モデルを変更しても
Docker imageを作り直す必要はありません。

測定条件や結果の記録方法は [docs/benchmark_protocol.md](docs/benchmark_protocol.md)、
報告書の雛形は [docs/report_template.md](docs/report_template.md) を参照して
ください。
