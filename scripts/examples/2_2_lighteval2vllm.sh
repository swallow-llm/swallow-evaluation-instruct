#!/bin/bash
set -euo pipefail

# このスクリプトはクローンしたリポジトリのルートディレクトリで実行してください．

export VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_USE_V1=0 # vLLM V0モードを指定する設定

MODEL_NAME="tokyotech-llm/Llama-3.1-Swallow-8B-Instruct-v0.5"
TASK_ID="swallow|gpqa:diamond"

MODEL_ARGS="pretrained=$MODEL_NAME,dtype=bfloat16,generation_parameters={temperature:0.0}"
# MODEL_ARGSには dtype，tensor_parallel_size，max_model_length，gpu_memory_utlization，そして各種 generation_parameters なども指定できる．


# 評価
echo "🐦 Evaluation has started"
uv run --isolated --locked --extra lighteval \
    lighteval vllm \
        "$MODEL_ARGS" \
        "${TASK_ID}|0|0" \
        --use-chat-template \
        --output-dir ./lighteval/outputs \
        --system-prompt "あなたは誠実で優秀な日本人のアシスタントです。" \
        --save-details


# 結果の集計
echo "🐦 Aggregating results has started"
uv run --isolated --locked --extra aggregate_results \
    python scripts/aggregate_results.py \
        --model_name "$MODEL_NAME"