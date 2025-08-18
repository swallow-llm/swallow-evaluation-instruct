#!/bin/bash
set -e

MODEL_NAME="tokyotech-llm/Llama-3.1-Swallow-8B-Instruct-v0.5"
PROVIDER_PREFIX="hosted_vllm/"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR/.."

uv run --isolated --locked --extra aggregate_results \
    python scripts/aggregate_results.py \
        --model_name "${PROVIDER_PREFIX}${MODEL_NAME}"
        # --raw_results_dir "./lighteval/outputs/results" \
        # --aggregated_outputs_dir "./aggregated_results"