#!/bin/bash

# litellmのログレベルをWARNINGに設定
export LITELLM_LOG_LEVEL=WARNING
GPUS="0,1"
NUM_GPUS=2
TEMPERATURE=0.0
MAX_COMPLETION_TOKENS=8192
TASK_NAME="swallow|swallow_gpqa_ja|0|0"

# 引数の取得
MODEL_NAME=$1
PROVIDER=$2
API_KEY=$3
SYSTEM_MESSAGE=$4

# プロバイダーに応じた設定
case $PROVIDER in
    "openai")
        BASE_URL="https://api.openai.com/v1"
        RAW_OUTPUT_DIR="./outputs"
        OUTPUT_DIR="./results"
        MODEL_NAME_CONFIG=$MODEL_NAME
        RAW_DIR="$RAW_OUTPUT_DIR/results/$MODEL_NAME"
        mkdir -p "$RAW_DIR"
        ;;
    "deepinfra")
        BASE_URL="https://api.deepinfra.com/v1/openai"
        RAW_OUTPUT_DIR="./outputs"
        OUTPUT_DIR="./results"
        MODEL_NAME_CONFIG="deepinfra/$MODEL_NAME"
        RAW_DIR="$RAW_OUTPUT_DIR/results/deepinfra/$MODEL_NAME"
        mkdir -p "$RAW_DIR"
        ;;
    "vllm")
        BASE_URL="http://localhost:8000/v1"
        RAW_OUTPUT_DIR="./outputs"
        OUTPUT_DIR="./results"
        MODEL_NAME_CONFIG="hosted_vllm/$MODEL_NAME"
        RAW_DIR="$RAW_OUTPUT_DIR/results/hosted_vllm/$MODEL_NAME"
        mkdir -p "$RAW_DIR"
        export CUDA_VISIBLE_DEVICES=$GPUS
        ;;
    *)
        echo "Error: Invalid provider. Must be one of: openai, deepinfra, vllm"
        exit 1
        ;;
esac

# vllmの場合のサーバー起動とクリーンアップ
if [ "$PROVIDER" = "vllm" ]; then
    # 終了時のクリーンアップ関数
    cleanup() {
        if [ ! -z "$VLLM_PID" ]; then
            echo "Stopping vllm server..."
            kill $VLLM_PID
        fi
    }

    # スクリプト終了時にクリーンアップを実行
    trap cleanup EXIT

    # vllmサーバーをバックグラウンドで起動
    echo "Starting vllm server..."
    uv run vllm serve $MODEL_NAME --tensor-parallel-size $NUM_GPUS > "$RAW_DIR/vllm_server.log" 2>&1 &
    VLLM_PID=$!

    # サーバーが起動するまで待機
    echo "Waiting for vllm server to start..."
    while ! curl -s $BASE_URL > /dev/null; do
        sleep 1
    done
    echo "vllm server is ready"
fi

# YAMLファイルを作成
cat > "$RAW_DIR/model_config.yaml" << EOL
model:
    base_params:
        model_name: $MODEL_NAME_CONFIG
        base_url: $BASE_URL
EOL

# API_KEYが必要なプロバイダーの場合のみ追加
if [ "$PROVIDER" != "vllm" ]; then
    sed -i "/base_url:.*/a \        api_key: $API_KEY" "$RAW_DIR/model_config.yaml"
fi

# 生成パラメータを追加
cat >> "$RAW_DIR/model_config.yaml" << EOL
    generation_parameters:
        temperature: $TEMPERATURE
        max_completion_tokens: $MAX_COMPLETION_TOKENS
EOL

# 評価の実行
if [ -z "$SYSTEM_MESSAGE" ]; then
    uv run lighteval endpoint litellm \
        "$RAW_DIR/model_config.yaml" \
        "$TASK_NAME" \
        --use-chat-template \
        --output-dir $RAW_OUTPUT_DIR \
        --save-details
else
    echo "Using system message: $SYSTEM_MESSAGE"
    uv run lighteval endpoint litellm \
        "$RAW_DIR/model_config.yaml" \
        "$TASK_NAME" \
        --system-prompt "${SYSTEM_MESSAGE}" \
        --use-chat-template \
        --output-dir $RAW_OUTPUT_DIR \
        --save-details
fi

# 結果の集計
python scripts/aggregate_results.py \
    $MODEL_NAME \
    "$RAW_DIR" \
    "$OUTPUT_DIR/$MODEL_NAME" 
