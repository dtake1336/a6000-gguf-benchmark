# Qwen3.8-Flash-Next configuration

Runtime overrides belong in `qwen-flash-server/.env`, copied from
`.env.example`. The model path is a path inside the container; the host
directory containing the GGUF files is selected with `MODEL_ROOT`.

The first shard is passed to `llama-server`. Hugging Face preserves the
quantization subdirectory, so the default path includes `UD-IQ4_XS/`. All
remaining shards must stay in the same host directory. The service does not
download, copy, or modify model files.

The production defaults are intentionally conservative for this machine:

| Setting | Default |
|---|---|
| Context | `261888` |
| Parallel slots | `1` |
| KV K / V | `q8_0` / `q8_0` |
| Batch / micro-batch | `2048` / `1024` |
| GPU split | layer, `1,1` |
| PLE | CPU via `per_layer_token_embd.weight=CPU` |
| CUDA architecture | `86` |

Do not set `MODEL_PATH` to a non-first shard. Do not remove the PLE override
without measuring VRAM and long-context behavior again.