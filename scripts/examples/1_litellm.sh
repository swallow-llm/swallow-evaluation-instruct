#!/bin/bash
set -euo pipefail

# このスクリプトはクローンしたリポジトリのルートディレクトリで実行してください．

MODEL_NAME="openai/o3-2025-04-16" 
BASE_URL="https://api.openai.com/v1/" # OpenAI API の URL
API_KEY="{OpenAIのAPI Key}" 
TASK_ID="swallow|gpqa:diamond"


# 評価
echo "🐦 Evaluation has started"
uv run --isolated --locked --extra lighteval \
    lighteval endpoint litellm \
        "model=$MODEL_NAME,base_url=$BASE_URL,api_key=$API_KEY" \
        "${TASK_ID}|0|0" \
        --use-chat-template \
        --output-dir ./lighteval/outputs \
        --save-details


# 結果の集計
echo "🐦 Aggregating results has started"
uv run --isolated --locked --extra aggregate_results \
    python scripts/aggregate_results.py \
        --model_name "$MODEL_NAME"