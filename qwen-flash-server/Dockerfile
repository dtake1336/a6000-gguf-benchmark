# syntax=docker/dockerfile:1.7
ARG CUDA_IMAGE=nvidia/cuda:12.8.1-devel-ubuntu24.04
FROM ${CUDA_IMAGE} AS builder

ARG DEBIAN_FRONTEND=noninteractive
ARG LLAMA_CPP_REF=911f6cdc8ab8a530b2bee09ee61471a6f3178eeb
ARG CUDA_ARCHITECTURES=86

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential ca-certificates cmake curl git libcurl4-openssl-dev \
    libgomp1 ninja-build pkg-config \
    && rm -rf /var/lib/apt/lists/*

RUN git clone --filter=blob:none \
      https://github.com/ggml-org/llama.cpp.git /opt/llama.cpp \
    && cd /opt/llama.cpp \
    && git fetch --no-tags origin "${LLAMA_CPP_REF}" \
    && git checkout --detach "${LLAMA_CPP_REF}"

RUN if [ -n "${CUDA_ARCHITECTURES}" ]; then \
      cmake -S /opt/llama.cpp -B /opt/llama.cpp/build -G Ninja \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_CUDA_ARCHITECTURES="${CUDA_ARCHITECTURES}" \
        -DGGML_CUDA=ON \
        -DLLAMA_CURL=ON \
        -DLLAMA_BUILD_TESTS=OFF \
        -DLLAMA_BUILD_EXAMPLES=ON \
        -DBUILD_SHARED_LIBS=OFF; \
    else \
      cmake -S /opt/llama.cpp -B /opt/llama.cpp/build -G Ninja \
        -DCMAKE_BUILD_TYPE=Release \
        -DGGML_CUDA=ON \
        -DLLAMA_CURL=ON \
        -DLLAMA_BUILD_TESTS=OFF \
        -DLLAMA_BUILD_EXAMPLES=ON \
        -DBUILD_SHARED_LIBS=OFF; \
    fi \
    && cmake --build /opt/llama.cpp/build --parallel "$(nproc)" \
      --target llama-server llama-bench

FROM nvidia/cuda:12.8.1-runtime-ubuntu24.04

ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates curl libcurl4 libgomp1 libstdc++6 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /opt/llama.cpp/build/bin/llama-server /usr/local/bin/llama-server
COPY --from=builder /opt/llama.cpp/build/bin/llama-bench /usr/local/bin/llama-bench
COPY scripts/start.sh /app/scripts/start.sh

RUN chmod 0755 /app/scripts/start.sh

WORKDIR /app
EXPOSE 8080
ENTRYPOINT ["/app/scripts/start.sh"]