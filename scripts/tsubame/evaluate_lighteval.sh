#!/bin/bash
#$ -cwd
#& -m abe
set -euo pipefail


# Load Args
TASK_NAME=$1
NODE_KIND=$2
MODEL_NAME=$3
CUSTOM_SETTINGS=$4
REPO_PATH=$5
PROVIDER=$6


# Setup
source "${REPO_PATH}/scripts/tsubame/common_funcs.sh"
init_common $MODEL_NAME $NODE_KIND $REPO_PATH
get_generation_params "${CUSTOM_SETTINGS}" "${TASK_NAME}" "${REPO_PATH}" "${MODEL_NAME}"
echo "⚙️ Generation Parameters: ${GEN_PARAMS}"
RAW_OUTPUT_DIR="${REPO_PATH}/lighteval/outputs"


# Serve a LLM by using litellm
serve_litellm $MODEL_NAME $PROVIDER $REPO_PATH "${CUSTOM_SETTINGS_SUBDIR}" "${GEN_PARAMS}" $TASK_NAME $NUM_GPUS $GPU_MEMORY_UTILIZATION $MAX_MODEL_LENGTH
AGGREGATED_OUTPUTS_DIR="${REPO_PATH}/results/${MODEL_NAME_CONFIG}${CUSTOM_SETTINGS_SUBDIR}"


# Task Definition
TASK_DEF="swallow|${TASK_NAME}|0|0"
echo "📝 Task: ${TASK_DEF}"


# Run Evaluation
cd "${REPO_PATH}/lighteval"
echo "🏃 Run Evaluation..."
start_time=$(date +%s)
uv run $UV_OPTIONS --extra lighteval \
 lighteval endpoint litellm $MODEL_CONFIG_PATH $TASK_DEF \
    --system-prompt "${SYSTEM_MESSAGE}" \
    --use-chat-template \
    --output-dir "${RAW_OUTPUT_DIR}" \
    --output-subdir "${CUSTOM_SETTINGS_SUBDIR}" \
    --save-details
end_time=$(date +%s)
elapsed=$(( end_time - start_time ))
echo "⌚️ Elapsed time: ${elapsed} seconds"


# Aggregate Results
aggregate_result "${MODEL_NAME_CONFIG}" "${RAW_RESULT_DIR}" "${AGGREGATED_OUTPUTS_DIR}" "${REPO_PATH}" "${CUSTOM_SETTINGS_PATH}" "${CUSTOM_SETTINGS_NAME}" "${CUSTOM_SETTINGS_VERSION}"