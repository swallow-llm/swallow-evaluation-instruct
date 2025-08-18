#!/bin/bash
set -euo pipefail

export VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_USE_V1=0 # vLLM V0モードを指定する設定

MODEL_NAME="tokyotech-llm/Llama-3.1-Swallow-8B-Instruct-v0.5"
TASK_ID="swallow|gpqa:diamond"

MODEL_ARGS="pretrained=$MODEL_NAME"
# ここで dtype，tensor_parallel_size，max_model_length，gpu_memory_utlization，そして各種 generation_parameters なども指定することができる．

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR/.."

uv run --isolated --locked --extra lighteval \
    lighteval vllm \
        $MODEL_ARGS \
        "${TASK_ID}|0|0" \
        --use-chat-template \
        --output-dir ./lighteval/outputs \
        # --reasoning-parser (reasoning parser はここで指定)
        # --system-prompt （System prompt はここで指定）