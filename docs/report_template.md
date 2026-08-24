# A6000 GGUF benchmark report

## Environment

| Field | Value |
|---|---|
| Date / run ID | |
| Host | |
| GPU | 2× RTX A6000 48 GB |
| Interconnect (`nvidia-smi topo -m`) | |
| Driver / CUDA image | |
| llama.cpp | b10509 / commit from output |
| lm-eval | v0.4.12 / commit from output |
| CPU / RAM / NUMA | |

## Model configurations

| Label | GGUF | Total shard bytes | GPU mode | Context | FA | Thinking | Speculation |
|---|---:|---:|---|---:|---|---|---|
| | | | | 8192 | on | off | off / MTP n=6 |

## Throughput

Keep llama-bench and API figures in separate tables.

### llama-bench

| Label | Split | Test | Mean tok/s | Stddev | Notes |
|---|---|---:|---:|---:|---|
| | | pp512 | | | |
| | | pp2048 | | | |
| | | pp8192 | | | |
| | | tg128 | | | |
| | | tg512 | | | |

### End-to-end API

| Label | Prompt tokens | Generated tokens | Prompt tok/s | Decode tok/s | Wall time | MTP |
|---|---:|---:|---:|---:|---:|---|
| | 512 | 128 | | | | |
| | 2048 | 512 | | | | |
| | 4096 | 512 | | | | |

## Quality

| Label | Quant | Task + version | Metric | Score | Samples | Few-shot | Chat template / thinking |
|---|---|---|---|---:|---:|---:|---|
| | | | | | | | |

## Conclusions

- Best one-GPU configuration:
- Best two-GPU configuration:
- Two-GPU scaling factor:
- Qwen MTP speedup:
- Quantization quality delta:
- OOM or context limitations:
- Recommended deployment configuration:
