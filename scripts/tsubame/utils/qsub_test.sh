#!/usr/bin/env bash
# How-to-use: bash scripts/tsubame/utils/qsub_test.sh
## This script is only for testing or debugging the evaluation scripts on an interactive node (not as a batch job) on TSUBAME.
## If you want to check setup-scripts on a login node, to avoid using many resources, you MUST comment out the following blocks:
## - `#Start vllm server in background` and `#Wait for server to start` in `evaluate_lighteval.sh`
## - `#Run evaluation` in `qsub_all.sh`
set -euo pipefail

########################################################

# Set Args
## Common Settings
NODE_KIND="node_q"
MODEL_NAME=""
SYSTEM_MESSAGE=""

## Special Settings
PROVIDER="vllm"             # Default: vllm. A provider to host the model. [vllm, openai, deepinfra]
PRIORITY="-5"               # Default: -5. A priority of the job. Note that double priority is double cost. [-5, -4, -3]
MAX_MODEL_LENGTH="-1"       # Default: -1 (Auto: min(32768, model_max_length)). Specify only if auto-detection is not working.
MAX_COMPLETION_TOKENS="-1"  # Default: -1 (Auto). Specify only if facing some critical issues (e.g. repetition).

## Task Settings
lang="ja"
task="humaneval"

########################################################

# Load task-definition
source "$(dirname "$0")/conf/load_config.sh"

# Load .env and define dirs
source "$(dirname "$0")/../../.env"
RESULTS_DIR="${REPO_PATH}/results/${MODEL_NAME}"
SCRIPTS_DIR="${REPO_PATH}/scripts/tsubame"

result_dir=$(task_result "${lang}_${task}")
task_name=$(task_script "${lang}_${task}")
task_framework=$(task_framework "${lang}_${task}")

OUTDIR="${RESULTS_DIR}/${result_dir}"
mkdir -p "${OUTDIR}"

bash "${SCRIPTS_DIR}/evaluate_${task_framework}.sh" \
    "${task_name}" "${NODE_KIND}" "${MODEL_NAME}" "${REPO_PATH}" "${SYSTEM_MESSAGE}" "${PROVIDER}" "${MAX_MODEL_LENGTH}" "${MAX_COMPLETION_TOKENS}"