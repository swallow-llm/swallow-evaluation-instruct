#!/bin/bash
set -euo pipefail

MODEL_NAME="tokyotech-llm/Llama-3.1-Swallow-8B-Instruct-v0.5"
TASK_ID="swallow|gpqa:diamond"

# API_KEY="" 
# OpenAIの推論APIや，それと互換性のあるNVIDIA NIM，DeepInfra の推論APIを使用する場合には対応するAPIキーを指定．

# SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
# cd "$SCRIPT_DIR/.."
cd /home/acg16653re/github/swallow-evaluation-instruct-private
CUDA_VISIBLE_DEVICES=0

source scripts/utils.sh

uv run --isolated --locked --extra vllm \
    vllm serve $MODEL_NAME \
        --host localhost \
        --port 8628 &
        # --reasoning-parser (vLLM 公式の reasoning parser はここで指定)
VLLM_SERVER_PID=$!

BASE_URL="http://localhost:8628/v1"
# HuggingFace のモデルをローカルでサーブする場合には "http://localhost:(ポート番号)/v1" を指定．
# 各種推論APIを用いる場合には，OpenAIであれば "https://api.openai.com/v1" を，
# DeepInfraであれば "https://api.deepinfra.com/v1/openai" のように適当なURLを指定する．

wait_for_vllm_server "$BASE_URL" "$VLLM_SERVER_PID" 900

uv run --isolated --locked --extra lighteval \
    lighteval endpoint litellm \
        "model=$MODEL_NAME,base_url=$BASE_URL" \
        "${TASK_ID}|0|0" \
        --use-chat-template \
        --output-dir ./lighteval/outputs \
        --max-samples 1