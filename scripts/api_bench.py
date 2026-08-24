#!/usr/bin/env python3
"""End-to-end llama-server benchmark using the native /completion endpoint."""
from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path
from typing import Any

import requests

BASE_TEXT = (
    "This is a neutral benchmarking passage about systems engineering, numerical "
    "methods, software reliability, and scientific measurement. Read it carefully "
    "and continue with a technically coherent explanation. "
)


def post_json(url: str, body: dict[str, Any], timeout: float) -> dict[str, Any]:
    r = requests.post(url, json=body, timeout=timeout)
    r.raise_for_status()
    return r.json()


def token_count(base_url: str, text: str, timeout: float) -> int:
    result = post_json(f"{base_url}/tokenize", {"content": text}, timeout)
    tokens = result.get("tokens", [])
    return len(tokens)


def make_prompt(base_url: str, target: int, timeout: float) -> tuple[str, int]:
    text = BASE_TEXT
    # Grow geometrically, then trim by words. Exact count is recorded in output.
    while token_count(base_url, text, timeout) < target:
        text += BASE_TEXT
    words = text.split()
    lo, hi = 1, len(words)
    best = text
    best_n = token_count(base_url, text, timeout)
    while lo <= hi:
        mid = (lo + hi) // 2
        candidate = " ".join(words[:mid])
        n = token_count(base_url, candidate, timeout)
        if abs(n - target) < abs(best_n - target):
            best, best_n = candidate, n
        if n < target:
            lo = mid + 1
        elif n > target:
            hi = mid - 1
        else:
            return candidate, n
    return best, best_n


def extract_metrics(response: dict[str, Any], wall_seconds: float) -> dict[str, Any]:
    timings = response.get("timings") or {}
    prompt_n = timings.get("prompt_n")
    predicted_n = timings.get("predicted_n")
    prompt_tps = timings.get("prompt_per_second")
    predicted_tps = timings.get("predicted_per_second")
    if predicted_tps is None and predicted_n:
        predicted_tps = predicted_n / wall_seconds
    return {
        "wall_seconds": wall_seconds,
        "prompt_tokens": prompt_n,
        "generated_tokens": predicted_n,
        "prompt_tps": prompt_tps,
        "generation_tps": predicted_tps,
        "stop_type": response.get("stop_type"),
        "stopped_eos": response.get("stopped_eos"),
        "truncated": response.get("truncated"),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://127.0.0.1:8000")
    ap.add_argument("--prompt-tokens", default="512,2048,4096")
    ap.add_argument("--generate-tokens", default="128,512")
    ap.add_argument("--repetitions", type=int, default=5)
    ap.add_argument("--warmups", type=int, default=1)
    ap.add_argument("--timeout", type=float, default=1800)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()

    prompt_sizes = [int(x) for x in args.prompt_tokens.split(",")]
    gen_sizes = [int(x) for x in args.generate_tokens.split(",")]
    prompts = {n: make_prompt(args.base_url, n, args.timeout) for n in prompt_sizes}
    records: list[dict[str, Any]] = []

    for target_prompt in prompt_sizes:
        prompt, measured_prompt = prompts[target_prompt]
        for n_predict in gen_sizes:
            body = {
                "prompt": prompt,
                "n_predict": n_predict,
                "temperature": 0,
                "seed": 42,
                "cache_prompt": False,
                "ignore_eos": True,
            }
            for _ in range(args.warmups):
                post_json(f"{args.base_url}/completion", body, args.timeout)
            for rep in range(args.repetitions):
                started = time.perf_counter()
                response = post_json(f"{args.base_url}/completion", body, args.timeout)
                elapsed = time.perf_counter() - started
                row = {
                    "target_prompt_tokens": target_prompt,
                    "tokenized_prompt_tokens": measured_prompt,
                    "requested_generation_tokens": n_predict,
                    "repetition": rep + 1,
                    **extract_metrics(response, elapsed),
                }
                records.append(row)
                print(json.dumps(row, ensure_ascii=False), flush=True)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        for row in records:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    summary: list[dict[str, Any]] = []
    for target_prompt in prompt_sizes:
        for n_predict in gen_sizes:
            rows = [
                r for r in records
                if r["target_prompt_tokens"] == target_prompt
                and r["requested_generation_tokens"] == n_predict
            ]
            for metric in ("prompt_tps", "generation_tps", "wall_seconds"):
                values = [float(r[metric]) for r in rows if r.get(metric) is not None]
                if values:
                    summary.append({
                        "target_prompt_tokens": target_prompt,
                        "requested_generation_tokens": n_predict,
                        "metric": metric,
                        "mean": statistics.fmean(values),
                        "stdev": statistics.stdev(values) if len(values) > 1 else 0.0,
                        "n": len(values),
                    })
    summary_path = args.output.with_name(args.output.stem + ".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
