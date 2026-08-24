# syntax=docker/dockerfile:1.7
ARG CUDA_IMAGE=nvidia/cuda:12.4.1-devel-ubuntu22.04
FROM ${CUDA_IMAGE}

ARG DEBIAN_FRONTEND=noninteractive
ARG LLAMA_CPP_REF=b10509
ARG LM_EVAL_REF=v0.4.12

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential ca-certificates cmake curl git jq ninja-build pkg-config \
    python3 python3-dev python3-pip python3-venv \
    libcurl4-openssl-dev libgomp1 pciutils procps numactl \
    && rm -rf /var/lib/apt/lists/*

# Pin both frameworks so results remain reproducible. Keep llama.cpp history so
# its bXXXXX build number is reported accurately by the compiled binaries.
RUN git clone --filter=blob:none \
      https://github.com/ggml-org/llama.cpp.git /opt/llama.cpp \
    && cd /opt/llama.cpp \
    && git fetch origin "${LLAMA_CPP_REF}" \
    && git checkout --detach "${LLAMA_CPP_REF}" \
    && cmake -S /opt/llama.cpp -B /opt/llama.cpp/build -G Ninja \
      -DCMAKE_BUILD_TYPE=Release \
      -DGGML_CUDA=ON \
      -DLLAMA_CURL=ON \
      -DLLAMA_BUILD_TESTS=OFF \
      -DLLAMA_BUILD_EXAMPLES=ON \
      -DBUILD_SHARED_LIBS=OFF \
    && cmake --build /opt/llama.cpp/build --parallel "$(nproc)" \
      --target llama-cli llama-server llama-bench llama-gguf-split

RUN git clone --branch "${LM_EVAL_REF}" --depth 1 \
      https://github.com/EleutherAI/lm-evaluation-harness.git /opt/lm-eval \
    && python3 -m pip install --no-cache-dir --upgrade pip setuptools wheel \
    && python3 -m pip install --no-cache-dir "/opt/lm-eval[api,math,ifeval,sentencepiece]" \
       "huggingface_hub[cli]" requests pandas

ENV PATH="/opt/llama.cpp/build/bin:${PATH}" \
    HF_HOME=/cache/huggingface \
    HF_HUB_ENABLE_HF_TRANSFER=0 \
    PYTHONUNBUFFERED=1

WORKDIR /workspace
CMD ["bash"]
