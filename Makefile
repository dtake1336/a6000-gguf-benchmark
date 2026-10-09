IMAGE ?= a6000-gguf-bench:latest
CUDA_IMAGE ?= nvidia/cuda:12.4.1-devel-ubuntu22.04
LLAMA_CPP_REF ?= b10509
LM_EVAL_REF ?= v0.4.12
QWEN_FLASH_SERVER_DIR ?= $(CURDIR)/qwen-flash-server
QWEN_FLASH_COMPOSE_FILE ?= $(QWEN_FLASH_SERVER_DIR)/docker-compose.yml
QWEN_FLASH_ENV_FILE ?= $(QWEN_FLASH_SERVER_DIR)/.env
QWEN_FLASH_COMPOSE = docker compose $(if $(wildcard $(QWEN_FLASH_ENV_FILE)),--env-file $(QWEN_FLASH_ENV_FILE),) -f $(QWEN_FLASH_COMPOSE_FILE)
MODEL_ROOT ?= $(CURDIR)/weights
RESULT_ROOT ?= $(CURDIR)/results
DOWNLOAD_SET ?= recommended
GPU ?= 0
GPU_MODE ?= 1gpu
PORT ?= 18080
QWEN36_35B_SERVER_NAME ?= a6000-qwen36-35b-server
QWEN36_35B_MODEL_PATH ?= /models/qwen36-35b-a3b-mtp/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf
QWEN36_35B_CTX_SIZE ?= 262144
NEMOTRON_SERVER_NAME ?= a6000-nemotron-server
NEMOTRON_MODEL_PATH ?= /models/nemotron-3.5-lightning-30b-a3b/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-UD-Q4_K_S.gguf
NEMOTRON_CTX_SIZE ?= 1048576

.PHONY: all build download download-hf clean bench smoke serve-qwen36-35b serve-nemotron bench-qwen38 bench-qwen36-35b bench-nemotron bench-laguna-iq4nl qwen-flash-build qwen-flash-up qwen-flash-down qwen-flash-health qwen-flash-benchmark

all: build

build:
	docker build \
	  --build-arg CUDA_IMAGE=$(CUDA_IMAGE) \
	  --build-arg LLAMA_CPP_REF=$(LLAMA_CPP_REF) \
	  --build-arg LM_EVAL_REF=$(LM_EVAL_REF) \
	  -t $(IMAGE) .

download:
	docker run --rm \
	  -v "$(CURDIR):/workspace" \
	  -v "$(MODEL_ROOT):/models" \
	  --env-file "$(CURDIR)/.env" \
	  -w /workspace \
	  $(IMAGE) \
	  bash -lc 'MODEL_ROOT=/models scripts/download_models.sh $(DOWNLOAD_SET)'

download-hf:
	$(if $(HF_SPEC),,$(error HF_SPEC is required. Usage: make download-hf HF_SPEC=unsloth/Repo-GGUF:Q4_K_M))
	docker run --rm \
	  -v "$(CURDIR):/workspace" \
	  -v "$(MODEL_ROOT):/models" \
	  --env-file "$(CURDIR)/.env" \
	  -w /workspace \
	  $(IMAGE) \
	  bash -lc 'MODEL_ROOT=/models scripts/download_hf_llama.sh $(HF_SPEC)'

bench:
	IMAGE="$(IMAGE)" MODEL_ROOT="$(MODEL_ROOT)" RESULT_ROOT="$(RESULT_ROOT)" \
	GPU_REQUEST="$(GPU)" GPU_MODE="$(GPU_MODE)" PORT="$(PORT)" \
	bash scripts/run_docker_bench.sh qwen_q4

serve-qwen36-35b:
	docker rm -f "$(QWEN36_35B_SERVER_NAME)" >/dev/null 2>&1 || true
	docker run --detach \
	  --gpus "device=$(GPU)" \
	  --ipc=host \
	  --name "$(QWEN36_35B_SERVER_NAME)" \
	  -v "$(MODEL_ROOT):/models:ro" \
	  -p "$(PORT):8080" \
	  "$(IMAGE)" \
	  llama-server \
	    -m "$(QWEN36_35B_MODEL_PATH)" \
	    --alias qwen36-35b-q4km \
	    --host 0.0.0.0 \
	    --port 8080 \
	    --n-gpu-layers 99 \
	    --ctx-size $(QWEN36_35B_CTX_SIZE) \
	    --flash-attn on \
	    --parallel 1 \
	    --jinja \
	    --reasoning off \
	    --temp 0 \
	    --seed 42 \
	    --split-mode none

serve-nemotron:
	docker rm -f "$(NEMOTRON_SERVER_NAME)" >/dev/null 2>&1 || true
	docker run --detach \
	  --gpus "device=$(GPU)" \
	  --ipc=host \
	  --name "$(NEMOTRON_SERVER_NAME)" \
	  -v "$(MODEL_ROOT):/models:ro" \
	  -p "$(PORT):8080" \
	  "$(IMAGE)" \
	  llama-server \
	    -m "$(NEMOTRON_MODEL_PATH)" \
	    --alias nemotron-3.5-q4ks \
	    --host 0.0.0.0 \
	    --port 8080 \
	    --n-gpu-layers 99 \
	    --ctx-size $(NEMOTRON_CTX_SIZE) \
	    --flash-attn on \
	    --parallel 1 \
	    --jinja \
	    --reasoning off \
	    --temp 0 \
	    --seed 42 \
	    --split-mode none

bench-qwen38:
	IMAGE="$(IMAGE)" MODEL_ROOT="$(MODEL_ROOT)" RESULT_ROOT="$(RESULT_ROOT)" \
	GPU_REQUEST="$(GPU)" GPU_MODE="$(GPU_MODE)" PORT="$(PORT)" \
	bash scripts/run_docker_bench.sh qwen38_q4

bench-qwen36-35b:
	IMAGE="$(IMAGE)" MODEL_ROOT="$(MODEL_ROOT)" RESULT_ROOT="$(RESULT_ROOT)" \
	GPU_REQUEST="$(GPU)" GPU_MODE="$(GPU_MODE)" PORT="$(PORT)" \
	bash scripts/run_docker_bench.sh qwen36_35b_q4km

bench-nemotron:
	IMAGE="$(IMAGE)" MODEL_ROOT="$(MODEL_ROOT)" RESULT_ROOT="$(RESULT_ROOT)" \
	GPU_REQUEST="$(GPU)" GPU_MODE="$(GPU_MODE)" PORT="$(PORT)" \
	bash scripts/run_docker_bench.sh nemotron_q4ks

bench-laguna-iq4nl:
	IMAGE="$(IMAGE)" MODEL_ROOT="$(MODEL_ROOT)" RESULT_ROOT="$(RESULT_ROOT)" \
	GPU_REQUEST="${GPU:-0,1}" GPU_MODE="${GPU_MODE:-2gpu-layer}" PORT="$(PORT)" \
	bash scripts/run_docker_bench.sh laguna_iq4nl

qwen-flash-build:
	$(QWEN_FLASH_COMPOSE) build

qwen-flash-up:
	$(QWEN_FLASH_COMPOSE) up -d

qwen-flash-down:
	$(QWEN_FLASH_COMPOSE) down

qwen-flash-health:
	bash $(QWEN_FLASH_SERVER_DIR)/scripts/healthcheck.sh

qwen-flash-benchmark:
	bash $(QWEN_FLASH_SERVER_DIR)/scripts/benchmark.sh

smoke:
	IMAGE="$(IMAGE)" MODEL_ROOT="$(MODEL_ROOT)" RESULT_ROOT="$(RESULT_ROOT)" \
	GPU_REQUEST="$(GPU)" GPU_MODE="$(GPU_MODE)" PORT="$(PORT)" REPEATS=1 \
	bash scripts/run_docker_bench.sh qwen_q4

clean:
	rm -rf $(RESULT_ROOT)/* logs/*
