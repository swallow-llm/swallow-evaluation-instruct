#!/bin/bash
set -euo pipefail

export VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_USE_V1=0 # vLLM V0モードを指定する設定

MODEL_NAME="tokyotech-llm/Llama-3.1-Swallow-8B-Instruct-v0.5"
TASK_ID="swallow|gpqa:diamond"

cd swallow-evaluation-instruct

# 評価
echo "🐦 Evaluation has started"
uv run --isolated --locked --extra lighteval \
    lighteval vllm \
        "pretrained=$MODEL_NAME,dtype=bfloat16" \
        "${TASK_ID}|0|0" \
        --use-chat-template \
        --output-dir ./lighteval/outputs


# 結果の集計
echo "🐦 Aggregating results has started"
uv run --isolated --locked --extra aggregate_results \
    python scripts/aggregate_results.py \
        --model_name "$MODEL_NAME"