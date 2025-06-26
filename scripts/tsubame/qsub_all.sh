#!/usr/bin/env bash
# How-to-use: bash scripts/tsubame/qsub_all.sh
set -euo pipefail

########################################################

# Set Args
## Common Settings
NODE_KIND="node_"
MODEL_NAME=""
SYSTEM_MESSAGE=""

## Special Settings
PROVIDER="vllm"             # Default: vllm. A provider to host the model. [vllm, openai, deepinfra]
PRIORITY="-5"               # Default: -5. A priority of the job. Note that double priority is double cost. [-5, -4, -3]
MAX_MODEL_LENGTH="-1"       # Default: -1 (Auto: min(32768, model_max_length)). Specify only if auto-detection is not working.
MAX_COMPLETION_TOKENS="-1"  # Default: -1 (Auto). Specify only if facing some critical issues (e.g. repetition).

########################################################

# Load task-definition
source "$(dirname "$0")/conf/load_config.sh"

# Define a qsub-command
QSUB_BASE="qsub -g tga-okazaki -l ${NODE_KIND}=1 -p ${PRIORITY}"

# Load .env and define dirs
source "$(dirname "$0")/../../.env"
RESULTS_DIR="${REPO_PATH}/results/${MODEL_NAME}"
SCRIPTS_DIR="${REPO_PATH}/scripts/tsubame"

# Define qsub-function
qsub_task() {
  # Get args
  local lang=$1 task=$2

  # Get task-specific args
  result_dir=$(task_result "${lang}_${task}")
  h_rt=$(hrt "${NODE_KIND}" "${lang}_${task}")
  task_name=$(task_script "${lang}_${task}")
  task_framework=$(task_framework "${lang}_${task}")
  [[ -z $task_name || -z $h_rt || -z $result_dir || -z $task_framework ]] && { echo "❌ unknown task ${lang}_${task}"; exit 1; }

  # Set an outdir 
  OUTDIR="${RESULTS_DIR}/${result_dir}"
  mkdir -p "${OUTDIR}"

  # Submit a job
  ${QSUB_BASE[@]} -N "${lang}_${task}" -l h_rt="${h_rt}" -o "${OUTDIR}" -e "${OUTDIR}" "${SCRIPTS_DIR}/evaluate_${task_framework}.sh" \
    "${task_name}" "${NODE_KIND}" "${MODEL_NAME}" "${REPO_PATH}" "${SYSTEM_MESSAGE}" "${PROVIDER}" "${MAX_MODEL_LENGTH}" "${MAX_COMPLETION_TOKENS}"
}

########################################################

# Submit tasks
echo "🚀 Submitting tasks..."

## Japanese
qsub_task ja gpqa
qsub_task ja jemhopqa_cot
qsub_task ja math_100
qsub_task ja mmlu
qsub_task ja mmlu_prox
qsub_task ja mtbench
qsub_task ja wmt20_en_ja
qsub_task ja wmt20_ja_en
qsub_task ja humaneval
qsub_task ja mifeval

## English
qsub_task en gpqa
qsub_task en hellaswag
qsub_task en mtbench
qsub_task en gpqa_diamond
qsub_task en math_500
qsub_task en aime_2024_2025
qsub_task en livecodebench_v5_v6


## Optional
qsub_task ja jemhopqa
qsub_task en mmlu_prox