#!/bin/bash
set -euo pipefail

MODEL_NAME="Qwen/Qwen3-4B" 
PROVIDER_NAME="hosted_vllm" # vLLMのプロバイダ名は "hosted_vllm" です
TASK_ID="swallow|humaneval"

cd swallow-evaluation-instruct
source scripts/examples/utils.sh


# vLLM serve の起動
echo "🐦 vLLM serve has started"
uv run --isolated --locked --extra vllm \
    vllm serve "$MODEL_NAME" \
        --host localhost \
        --port 25818 \
        --reasoning-parser qwen3 \
        --max-model-len 32768 &
VLLM_SERVER_PID=$!

BASE_URL="http://localhost:25818/v1"
# HuggingFace のモデルをローカルでサーブする場合には "http://localhost:(ポート番号)/v1" を指定．

wait_for_vllm_server "$BASE_URL" "$VLLM_SERVER_PID" 900


# 評価
echo "🐦 Evaluation has started"
uv run --isolated --locked --extra lighteval \
    lighteval endpoint litellm \
        "model=$PROVIDER_NAME/$MODEL_NAME,base_url=$BASE_URL,generation_parameters={temperature:0.2,top_p:0.95}" \
        "${TASK_ID}|0|0" \
        --use-chat-template \
        --output-dir ./lighteval/outputs


# 結果の集計
echo "🐦 Aggregating results has started"
uv run --isolated --locked --extra aggregate_results \
    python scripts/aggregate_results.py \
        --model_name "$PROVIDER_NAME/$MODEL_NAME"