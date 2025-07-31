#!/bin/bash
#$ -cwd
#% -m abe
#% -M your_email@address.here
# Replace % with $ if you want to receive emails when jobs start & finish, and errors occur.
set -euo pipefail


# Load Args
## Default Values
TASK_NAME=""; NODE_KIND=""; MODEL_NAME=""; REPO_PATH=""; SERVICE=""; CUSTOM_SETTINGS=""; PROVIDER=""; CUSTOM_JOB_ID=""; MAX_SAMPLES=""
STDOUT_STDERR_DIR=""; : "${CUDA_VISIBLE_DEVICES:=}"

## Parse Args
echo "🔧 Received args: $@"
while [[ $# -gt 0 ]]; do
  case "$1" in
    --task-name) TASK_NAME="$2";;
    --node-kind) NODE_KIND="$2";;
    --model-name) MODEL_NAME="$2";;
    --repo-path) REPO_PATH="$2";;
    --service) SERVICE="$2";;
    --provider) PROVIDER="$2";;
    --custom-settings) CUSTOM_SETTINGS="$2";;       # Optional
    --custom-job-id) CUSTOM_JOB_ID="$2";;           # Optional
    --max-samples) MAX_SAMPLES=$2;;               # Optional
    --stdout-stderr-dir) STDOUT_STDERR_DIR="$2";;   # Optional
    *) echo "💀 Error: Unknown option: $1" >&2;;
  esac
  shift 2
done


## Redirect stdout and stderr to files if specified
if [[ -n "$STDOUT_STDERR_DIR" ]]; then
  if [[ $SERVICE == "abci" ]]; then
    ### ABCI does not support streaming stdout and stderr into specific files, so we need to tee them by ourselves.
    exec > >(tee -a "${STDOUT_STDERR_DIR}/${PBS_JOBID}.OU") 2> >(tee -a "${STDOUT_STDERR_DIR}/${PBS_JOBID}.ER" >&2)
  fi
  if [[ $SERVICE == "local" ]]; then
    ### Local uses tmux session to run the job, so we need to tee the stdout and stderr in the session.
    source "${REPO_PATH}/scripts/qsub/conf/load_config.sh"
    JOB_NAME=$(script_task "${TASK_NAME}")
    exec > >(tee -a "${STDOUT_STDERR_DIR}/${JOB_NAME}.o${CUSTOM_JOB_ID}" ) 2> >(tee -a "${STDOUT_STDERR_DIR}/${JOB_NAME}.e${CUSTOM_JOB_ID}" >&2)
  fi
fi


## Check Required Args
if [[ -z "$TASK_NAME" ]] || [[ -z "$NODE_KIND" ]] || [[ -z "$MODEL_NAME" ]] || [[ -z "$REPO_PATH" ]] || [[ -z "$SERVICE" ]]; then
  echo "💀 Error: Missing required arguments. TASK_NAME: '${TASK_NAME}', NODE_KIND: '${NODE_KIND}', MODEL_NAME: '${MODEL_NAME}', REPO_PATH: '${REPO_PATH}', SERVICE: '${SERVICE}'" >&2
  exit 1
fi


# Setup
source "${REPO_PATH}/scripts/qsub/common_funcs.sh"
init_service "${SERVICE}" "${NODE_KIND}" "${CUDA_VISIBLE_DEVICES}" "${CUSTOM_JOB_ID}"
init_common "${REPO_PATH}"
get_generation_params "${CUSTOM_SETTINGS}" "${TASK_NAME}" "${REPO_PATH}" "${MODEL_NAME}" "${MAX_SAMPLES}"
echo "⚙️ Generation Parameters: ${GEN_PARAMS}"
RAW_OUTPUT_DIR="${REPO_PATH}/lighteval/outputs"


# Serve a LLM by using litellm
serve_litellm "${MODEL_NAME}" "${PROVIDER}" "${REPO_PATH}" "${CUSTOM_SETTINGS_SUBDIR}" "${GEN_PARAMS}" "${TASK_NAME}" "${NODE_KIND}" "${NUM_GPUS}" "${GPU_MEMORY_UTILIZATION}" "${MAX_MODEL_LENGTH}" "${REASONING_PARSER}"


# Task Definition
TASK_DEF="swallow|${TASK_NAME}|0|0"
echo "📝 Task: ${TASK_DEF}"


# Run Evaluation
cd "${REPO_PATH}/lighteval"
echo "🏃 Run Evaluation..."
echo "🔍 Optional Arguments for Lighteval: ${OPTIONAL_ARGS_FOR_LIGHTEVAL[@]}"
start_time=$(date +%s)
uv run $UV_OPTIONS --extra lighteval \
 lighteval endpoint litellm $MODEL_CONFIG_PATH $TASK_DEF \
    --use-chat-template \
    --output-dir "${RAW_OUTPUT_DIR}" \
    --output-subdir "${CUSTOM_SETTINGS_SUBDIR}" \
    --save-details \
    "${OPTIONAL_ARGS_FOR_LIGHTEVAL[@]}"
end_time=$(date +%s)
elapsed=$(( end_time - start_time ))
echo "⌚️ Elapsed time: ${elapsed} seconds"


# Stop VLLM Server
if [[ -n $VLLM_SERVER_PID ]]; then
    stop_vllm_server "${VLLM_SERVER_PID}"
fi


# Aggregate Results
aggregate_result "${MODEL_NAME_CONFIG}" "${RAW_RESULT_DIR}" "${AGGREGATED_OUTPUTS_DIR}" "${REPO_PATH}" "${CUSTOM_SETTINGS_PATH}" "${CUSTOM_SETTINGS_NAME}" "${CUSTOM_SETTINGS_VERSION}"
