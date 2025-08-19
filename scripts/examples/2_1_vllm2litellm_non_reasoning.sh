#!/bin/bash
set -euo pipefail

# このスクリプトはクローンしたリポジトリのルートディレクトリで実行してください．

MODEL_ID="tokyotech-llm/Llama-3.1-Swallow-8B-Instruct-v0.5"
MODEL_NAME="hosted_vllm/${MODEL_ID}"
TASK_ID="swallow|japanese_mt_bench"

export OPENAI_API_KEY="{LLM-as-a-Judgeに使うOpenAI API Key}" 
source scripts/examples/utils.sh


# vLLM serve の起動
echo "🐦 vLLM serve has started"
uv run --isolated --locked --extra vllm \
    vllm serve "$MODEL_ID" \
        --host localhost \
        --port 25819 &
VLLM_SERVER_PID=$!

BASE_URL="http://localhost:25819/v1"
# HuggingFace のモデルをローカルでサーブする場合には "http://localhost:(ポート番号)/v1" を指定．


# vLLMが起動するまで待機
wait_for_vllm_server "$BASE_URL" "$VLLM_SERVER_PID"


# 評価
echo "🐦 Evaluation has started"
uv run --isolated --locked --extra lighteval \
    lighteval endpoint litellm \
        "model=$MODEL_NAME,base_url=$BASE_URL" \
        "${TASK_ID}|0|0" \
        --use-chat-template \
        --system-prompt "あなたは誠実で優秀な日本人のアシスタントです。" \
        --output-dir ./lighteval/outputs \
        --save-details


# 結果の集計
echo "🐦 Aggregating results has started"
uv run --isolated --locked --extra aggregate_results \
    python scripts/aggregate_results.py \
        --model_name "$MODEL_NAME"