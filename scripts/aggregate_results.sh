#!/bin/bash
set -e

# このスクリプトはクローンしたリポジトリのルートディレクトリで実行してください．

MODEL_NAME="tokyotech-llm/Llama-3.1-Swallow-8B-Instruct-v0.5"
PROVIDER_PREFIX="hosted_vllm/"

uv run --isolated --locked --extra aggregate_results \
    python scripts/aggregate_results.py \
        --model_name "${PROVIDER_PREFIX}${MODEL_NAME}"
        # --raw_results_dir "./lighteval/outputs/results" \
        # --aggregated_outputs_dir "./aggregated_results"