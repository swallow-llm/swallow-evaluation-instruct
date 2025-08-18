#!/bin/bash
set -euo pipefail

MODEL_NAME="tokyotech-llm/Llama-3.1-Swallow-8B-Instruct-v0.5"
TASK_ID="swallow|gpqa:diamond"

# API_KEY="" 
# OpenAIの推論APIや，それと互換性のあるNVIDIA NIM，DeepInfra の推論APIを使用する場合には対応するAPIキーを指定．

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR/.."

uv run --isolated --locked --extra vllm \
    vllm serve --model $MODEL_NAME \
        --host localhost \
        --port 8000 &
        # --reasoning-parser (vLLM 公式の reasoning parser はここで指定)

sleep 30    # サーバーが起動するまで待つ

BASE_URL="http://localhost:8000/v1"
# HuggingFace のモデルをローカルでサーブする場合には "http://localhost:(ポート番号)/v1" を指定．
# 各種推論APIを用いる場合には，OpenAIであれば "https://api.openai.com/v1" を，
# DeepInfraであれば "https://api.deepinfra.com/v1/openai" のように適当なURLを指定する．

uv run --isolated --locked --extra lighteval \
    lighteval endpoint litellm \
        "scripts/config.yaml" \     # ここを config ファイルへのパスに変更する
        "${TASK_ID}|0|0" \
        --use-chat-template \
        --output-dir ./lighteval/outputs