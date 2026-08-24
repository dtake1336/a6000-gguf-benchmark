# Benchmark protocol

## Primary questions

1. What is single-stream prompt-processing throughput (pp tokens/s)?
2. What is single-stream decode throughput (tg tokens/s)?
3. What is end-to-end API throughput including tokenization and sampling?
4. Does splitting across two A6000s improve or reduce throughput?
5. How much does Qwen MTP speculative decoding change decode throughput?
6. What accuracy is retained by each selected quantization under one fixed prompt protocol?

## Controlled variables

- Same llama.cpp build, lm-eval version, driver, CUDA image, CPU allocation, host, and model files.
- `--parallel 1`, greedy decoding, seed 42, 8,192-token context for the baseline.
- Thinking disabled globally via `chat_template_kwargs={"enable_thinking":false}`.
- One warm-up and five measured repetitions for API throughput.
- Record `nvidia-smi topo -m`; two-GPU results depend strongly on PCIe/NVLink topology.
- Run quality only after throughput configuration selection. Never compare scores produced by different task versions or chat-template settings as if they were identical.

## Throughput tiers

- `llama-bench`: low-overhead kernel-oriented pp/tg measurements. Tokenization and sampling are excluded.
- Native `/completion`: end-to-end single-user measurement, including server overhead; used for MTP on/off comparison.
- Optional serving phase: rerun with `PARALLEL=4` and a concurrent load generator. Keep this separate from the single-stream report. Qwen MTP should remain at parallel=1 unless current llama.cpp explicitly supports otherwise.

## Recommended quality suites

Baseline suite in the scripts:

- `mmlu_pro`
- `gpqa_diamond_zeroshot`
- `hellaswag`
- `arc_challenge`
- `winogrande`
- `gsm8k`
- `ifeval`

For a frozen Open LLM Leaderboard-style run, use the groups in `configs/suites.env` (`leaderboard_bbh`, `leaderboard_gpqa`, `leaderboard_mmlu_pro`, `leaderboard_musr`, `leaderboard_ifeval`, and `leaderboard_math_hard`). Treat that as a separate report because its few-shot settings and metrics differ from the baseline suite.

Before the expensive run, validate identifiers and dependencies:

```bash
lm-eval validate --tasks mmlu_pro,gpqa_diamond_zeroshot,hellaswag,arc_challenge,winogrande,gsm8k,ifeval
```

For Japanese use cases, add `jfinqa` and inspect available Japanese task groups with:

```bash
lm-eval ls tasks | grep -Ei 'jfinqa|japanese|ja_'
```

## Reporting

Report at least:

- model repository and exact GGUF filename/shards
- first-64-MiB hash and file size (or full SHA-256 if practical)
- llama.cpp and lm-eval revisions
- GPU model, clocks, power limit, driver, CUDA image
- GPU topology
- prompt length, generated length, repetitions, mean, standard deviation
- split mode and tensor split
- context size, flash attention, MTP settings
- chat-template kwargs and whether thinking is enabled
- task versions, few-shot settings, sample count, and random seed
