#!/bin/bash
#$ -cwd
#& -m abe
set -euo pipefail


# Extra Generation Parameters
## If not specified, DEFAULT_GEN_PARAMS is used.
## Note that max_new_tokens is automatically set to max_model_length if not specified.
DEFAULT_GEN_PARAMS=$(cat <<'EOL'
        temperature: 0.0
EOL
)
declare -A GEN_PARAMS_LIST
GEN_PARAMS_LIST=(
    [japanese_mt_bench]=""
    [swallow_jhumaneval]=$(cat <<'EOL'
        temperature: 0.2
        top_p: 0.95
EOL
)
    [mifeval_ja]=$(cat <<'EOL'
        temperature: 0.0
        top_p: 1.0
EOL
)
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
source "${REPO_PATH}/scripts/tsubame/common_funcs.sh"
init_common $MODEL_NAME $NODE_KIND $REPO_PATH
RAW_OUTPUTS_DIR="${REPO_PATH}/lighteval/outputs"
AGGREGATED_OUTPUTS_DIR="${REPO_PATH}/results/${MODEL_NAME}"


# Generation Parameters
## Set GEN_PARAMS based on TASK_NAME
if [[ -v GEN_PARAMS_LIST[$TASK_NAME] ]]; then
    GEN_PARAMS=${GEN_PARAMS_LIST[$TASK_NAME]}
else
    GEN_PARAMS=${DEFAULT_GEN_PARAMS}
fi

## Set MAX_COMPLETION_TOKENS only if specified
if [[ "$MAX_COMPLETION_TOKENS" != "-1" ]]; then
    GEN_PARAMS+="        max_new_tokens: $MAX_COMPLETION_TOKENS
"
fi

## Set MODEL_ARGS
if [[ -n $GEN_PARAMS ]]; then
  GEN_PARAMS="    generation:
$GEN_PARAMS"
fi
echo "⚙️ Generation Parameters: ${GEN_PARAMS}"


# Serve a LLM by using litellm
serve_litellm $MODEL_NAME $PROVIDER $REPO_PATH "${GEN_PARAMS}" $TASK_NAME $NUM_GPUS $GPU_MEMORY_UTILIZATION $MAX_MODEL_LENGTH
case $PROVIDER in
    "openai") PROVIDER_SUBDIR="" ;;
    "deepinfra") PROVIDER_SUBDIR="deepinfra/" ;;
    "vllm") PROVIDER_SUBDIR="hosted_vllm/" ;;
    *) echo "❌ Unknown PROVIDER: $PROVIDER" && exit 1 ;;
esac
MODEL_CONFIG_PATH="${RAW_OUTPUTS_DIR}/results/${PROVIDER_SUBDIR}${MODEL_NAME}/model_config_${TASK_NAME}.yaml"
RESULTS_DIR="${RAW_OUTPUTS_DIR}/results/${PROVIDER_SUBDIR}${MODEL_NAME}"


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
    --output-dir "${RAW_OUTPUTS_DIR}" \
    --save-details
end_time=$(date +%s)
elapsed=$(( end_time - start_time ))
echo "⌚️ Elapsed time: ${elapsed} seconds"


# Aggregate Results
aggregate_result "${PROVIDER_SUBDIR}${MODEL_NAME}" "${RESULTS_DIR}" "${AGGREGATED_OUTPUTS_DIR}" "${REPO_PATH}"