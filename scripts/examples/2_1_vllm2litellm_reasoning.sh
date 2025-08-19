#!/bin/bash
set -euo pipefail

# このスクリプトはクロンしたリポジトリのルートディレクトリで実行してください．

MODEL_ID="Qwen/Qwen3-4B" 
# vLLMでセルフホストする場合のプロバイダ名は "hosted_vllm" とします
MODEL_NAME="hosted_vllm/${MODEL_ID}"
TASK_ID="swallow|humaneval"

source scripts/examples/utils.sh


# vLLM serve の起動
echo "🐦 vLLM serve has started"
uv run --isolated --locked --extra vllm \
    vllm serve "$MODEL_ID" \
        --host localhost \
        --port 25819 \
        --reasoning-parser qwen3 \
        --max-model-len 32768 &
VLLM_SERVER_PID=$!

BASE_URL="http://localhost:25819/v1"
# HuggingFace のモデルをローカルでサーブする場合には "http://localhost:(ポート番号)/v1" を指定．


# vLLMが起動するまで待機
wait_for_vllm_server "$BASE_URL" "$VLLM_SERVER_PID"


# 評価
echo "🐦 Evaluation has started"
uv run --isolated --locked --extra lighteval \
    lighteval endpoint litellm \
        "model=$MODEL_NAME,base_url=$BASE_URL,generation_parameters={temperature:0.2,top_p:0.95}" \
        "${TASK_ID}|0|0" \
        --use-chat-template \
        --output-dir ./lighteval/outputs \
        --save-details


# 結果の集計
echo "🐦 Aggregating results has started"
uv run --isolated --locked --extra aggregate_results \
    python scripts/aggregate_results.py \
        --model_name "$MODEL_NAME"