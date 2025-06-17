#!/bin/bash
#$ -cwd
#& -m abe
set -euo pipefail


# Extra Generation Parameters
## If not specified, DEFAULT_GEN_PARAMS is used.
## Note that max_new_tokens is automatically set to max_model_length if not specified.
DEFAULT_GEN_PARAMS="temperature:0.0"
declare -A GEN_PARAMS_LIST=(
    [japanese_mt_bench]=""
    [swallow_jhumaneval]="temperature:0.2,top_p:0.95"
)


# Load Args
TASK_NAME=$1
NODE_KIND=$2
MODEL_NAME=$3
REPO_PATH=$4
SYSTEM_MESSAGE=$5
PROVIDER=$6
MAX_MODEL_LENGTH=$7
MAX_COMPLETION_TOKENS=$8


# Setup
source "${REPO_PATH}/scripts/tsubame/utils.sh"
init_common $MODEL_NAME $NODE_KIND $REPO_PATH
serve_litellm $MODEL_NAME $PROVIDER $REPO_PATH
RAW_OUTPUTS_DIR="${REPO_PATH}/lighteval/outputs"
AGGREGATED_OUTPUTS_DIR="${REPO_PATH}/results/${MODEL_NAME}"


# Generation Parameters
## Set NUM_GPUS based on NODE_KIND
if [[ $NODE_KIND == "node_q" ]]; then
    NUM_GPUS=1
else if [[ $NODE_KIND == "node_f" ]]; then
    NUM_GPUS=4
else
    echo "❌ Unknown NODE_KIND: $NODE_KIND"
    exit 1
fi

## Set GPU_MEMORY_UTILIZATION
GPU_MEMORY_UTILIZATION=0.9

## Set GEN_PARAMS based on TASK_NAME
if [[ -v GEN_PARAMS_LIST[$TASK_NAME] ]]; then
    GEN_PARAMS=${GEN_PARAMS_LIST[$TASK_NAME]}
else
    GEN_PARAMS=${DEFAULT_GEN_PARAMS}
fi

## Set MAX_COMPLETION_TOKENS only if specified
if [[ $MAX_COMPLETION_TOKENS != "-1" ]]; then
    GEN_PARAMS="${GEN_PARAMS},max_new_tokens:${MAX_COMPLETION_TOKENS}"
fi

## Set MODEL_ARGS
MODEL_ARGS="pretrained=$MODEL_NAME,dtype=bfloat16,tensor_parallel_size=$NUM_GPUS,max_model_length=$MAX_MODEL_LENGTH,gpu_memory_utilization=$GPU_MEMORY_UTILIZATION,generation_parameters={${GEN_PARAMS}}"
echo "🤖 MODEL_ARGS: $MODEL_ARGS"


# Task Definition
TASK_DEF="swallow|${TASK_NAME}|0|0"
echo "📝 Task: ${TASK_DEF}"


# Run Evaluation
cd "${REPO_PATH}/lighteval"
start_time=$(date +%s)
uv run $UV_OPTIONS --extra lighteval \
 lighteval vllm $MODEL_ARGS $TASK_DEF \
    --system-prompt "${SYSTEM_MESSAGE}" \
    --use-chat-template \
    --output-dir "${RAW_OUTPUTS_DIR}" \
    --save-details
end_time=$(date +%s)
elapsed=$(( end_time - start_time ))
echo "⌚️ Elapsed time: ${elapsed} seconds"


# Aggregate Results
aggregate_result $MODEL_NAME $RAW_OUTPUTS_DIR $AGGREGATED_OUTPUTS_DIR