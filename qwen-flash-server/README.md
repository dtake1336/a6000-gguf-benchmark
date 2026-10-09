# Qwen3.8-Flash-Next server

このディレクトリは、RTX A6000 48 GB x2向けのQwen3.8-Flash-Next
`UD-IQ4_XS` llama.cppサーバーです。GGUF本体はDocker imageに含めず、
ホスト側のディレクトリをコンテナ起動時に`/models:ro`でマウントします。

## 前提条件

- Linux
- Docker EngineとDocker Compose v2
- NVIDIA driver、NVIDIA Container Toolkit
- RTX A6000 48 GB x2
- GGUF用に約100 GB以上の空き容量と、PLEのCPU配置用RAM

GPUがコンテナから見えることを先に確認します。

```bash
nvidia-smi
docker run --rm --gpus '"device=0,1"' nvidia/cuda:12.8.1-runtime-ubuntu24.04 nvidia-smi
```

## モデルのダウンロード

ダウンロードは重いため、このリポジトリでは実行していません。既存の
ベンチマークimageに入っている`hf` CLIを使い、以下をユーザー側で実行
してください。出力先は既定で既存の`weights/`です。

```bash
docker run --rm \
  -v "$PWD/weights:/models" \
  -v "$PWD/qwen-flash-server/scripts:/scripts:ro" \
  -w /workspace \
  --env HF_REPO=unsloth/Qwen3.8-Flash-Next-GGUF \
  --env HF_QUANT=UD-IQ4_XS \
  --env MODEL_ROOT=/models \
  --env MODEL_DIR=qwen38-flash-next \
  --env HF_TOKEN \
  a6000-gguf-bench:latest \
  bash /scripts/download-model.sh
```

`a6000-gguf-bench:latest`がまだない場合は、リポジトリルートで先に
`make build`を実行してください。`hf`をホストにインストール済みなら、
次でも同じダウンロードができます。

```bash
MODEL_ROOT="$PWD/weights" \
  qwen-flash-server/scripts/download-model.sh
```

完了後、次のような全shardが同じディレクトリにあることを確認します。

```text
weights/qwen38-flash-next/
  UD-IQ4_XS/
    Qwen3.8-Flash-Next-UD-IQ4_XS-00001-of-00003.gguf
    Qwen3.8-Flash-Next-UD-IQ4_XS-00002-of-00003.gguf
    Qwen3.8-Flash-Next-UD-IQ4_XS-00003-of-00003.gguf
```

## Buildと起動

```bash
cp qwen-flash-server/.env.example qwen-flash-server/.env
docker compose \
  --env-file qwen-flash-server/.env \
  -f qwen-flash-server/docker-compose.yml \
  build
docker compose \
  --env-file qwen-flash-server/.env \
  -f qwen-flash-server/docker-compose.yml \
  up -d
```

`docker compose up -d --build`でも構いません。ComposeはGPU 0とGPU 1だけを
コンテナへ割り当てます。`MODEL_ROOT`はホスト側、`MODEL_PATH`はコンテナ側の
パスです。別のモデル保存先を使う場合は`.env`で`MODEL_ROOT`を変更します。

起動ログは次で確認できます。

```bash
docker compose --env-file qwen-flash-server/.env \
  -f qwen-flash-server/docker-compose.yml logs -f qwen
```

保存する場合:

```bash
mkdir -p logs
docker compose --env-file qwen-flash-server/.env \
  -f qwen-flash-server/docker-compose.yml logs --no-color qwen \
  > logs/llama-server.log
```

ログにはcontext、batch、KV cache、Flash Attention、PLE CPU配置の設定が
出力されます。llama.cppのロードログで、両GPUが使われ、
`per_layer_token_embd.weight`がCPUに配置されたことも確認してください。

## API

ヘルスチェック:

```bash
qwen-flash-server/scripts/healthcheck.sh
```

OpenAI-compatible base URLは次です。

```text
http://<server-ip>:8080/v1
```

疎通確認:

```bash
curl --fail --silent --show-error \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "qwen3.8-flash-next",
    "messages": [{"role": "user", "content": "Write a Python function that implements binary search."}],
    "temperature": 0.2,
    "top_p": 0.95,
    "max_tokens": 1024
  }' \
  http://127.0.0.1:8080/v1/chat/completions
```

Chat templateはGGUF内のものを使うため、起動時に`--jinja`を有効にしています。

停止:

```bash
docker compose --env-file qwen-flash-server/.env \
  -f qwen-flash-server/docker-compose.yml down
```

## 設定とベンチマーク

初期プロファイルは次の通りです。

```text
llama.cpp commit: 911f6cdc8ab8a530b2bee09ee61471a6f3178eeb
build date:      2026-09-18T20:10:11+09:00
CUDA:            12.8.93 / CUDA image 12.8.1 / Ampere architecture 86
NVIDIA driver:   570.124.06
GPU:             NVIDIA RTX A6000 48 GB x2 (host GPU 0/1)
context:         261888
quantization:    UD-IQ4_XS
KV cache:        q8_0 / q8_0
parallel:        1
batch/ubatch:    2048 / 256
tensor split:    1,1
PLE:             CPU
CPU:              2 sockets, Intel Xeon Gold 5218R, 80 logical CPUs
RAM:              754 GiB visible, swap 15 GiB
NUMA:             2 nodes; GPU 0/1 are on NUMA node 0
GPU topology:     GPU 0 <-> GPU 1 NV4; GPU 2 is a separate RTX 3090 and is hidden
                  from this service
```

On this pinned llama.cpp build, `UBATCH_SIZE=1024` reproduced a CUDA illegal
memory access during a 512-token completion on the A6000 pair. The default
`UBATCH_SIZE=256` completed repeated measurements while preserving
continuous batching.

基準測定はサーバーを起動した状態で、リポジトリルートの既存
`a6000-gguf-bench:latest` imageにある`api_bench.py`を再利用します。結果は
`qwen-flash-server/benchmarks/results.csv`と実行別JSONLへ保存されます。

```bash
qwen-flash-server/scripts/benchmark.sh
```

長いpromptは時間とVRAMを大きく消費するため、最初の疎通確認では次のように
短くしてから本番測定へ進められます。

```bash
PROMPT_TOKENS=8192,32768 GENERATE_TOKENS=512 REPETITIONS=1 \
  qwen-flash-server/scripts/benchmark.sh
```

`nvidia-smi`のサンプルは実行別ディレクトリの`nvidia-smi.csv`に残ります。
既存の`api_bench.py`へ追加した`--measure-ttft`を専用ベンチから有効にし、
native streamingの最初の非空chunkまでをCSVの`ttft_seconds`へ保存します。

## ハードウェア記録

実測値をREADMEや測定結果と一緒に残す前に、次を保存します。

```bash
mkdir -p qwen-flash-server/config
{
  date -Is
  lscpu
  free -h
  numactl --hardware || true
  nvidia-smi topo -m
} | tee "qwen-flash-server/config/hardware-$(date -u +%Y%m%dT%H%M%SZ).txt"
```

RAM channel・speedはこの環境では未取得です。ここでは実機モデルをまだ
ロードしていないため、推論速度・実使用VRAM・長文安定性の実測値は未記録です。

## 更新とrollback

`LLAMA_CPP_REF`はmasterではなくcommit hashで固定しています。更新時は`.env`の
値を変更してimageをbuildし、速度と長文安定性を比較します。

```bash
LLAMA_CPP_REF=<new-commit-hash> \
  docker compose --env-file qwen-flash-server/.env \
  -f qwen-flash-server/docker-compose.yml build
docker image tag qwen-flash-next:latest qwen-flash-next:<new-commit-hash>
```

問題があれば、以前のcommitを`.env`へ戻し、`docker compose build`して再起動
します。`262144`は既知のCUDA grid上限問題を避けるため標準値にしません。
KVを`q4_0`へ変更する場合や、`TENSOR_SPLIT`を微調整する場合は、一度に一つの
値だけを変え、同じprompt条件の結果を残してください。