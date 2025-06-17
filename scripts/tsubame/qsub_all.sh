#!/usr/bin/env bash
# How-to-use: bash scripts/tsubame/qsub_all.sh $NODE_KIND[node_q/node_f] $MODE_NAME_PATH ($PREDOWNLOAD_MODEL[true/false]) ($PRIORITY[-5/-4/-3])
set -euo pipefail

########################################################

# Set Args
## Common Settings
NODE_KIND="node_q"
MODEL_NAME="tokyotech-llm/Llama-3.1-Swallow-8B-Instruct-v0.3"
SYSTEM_MESSAGE="あなたは誠実で優秀な日本人のアシスタントです。"

## Special Settings
PROVIDER="vllm"             # Default: vllm. Specify only if you want to use a different provider. (e.g. openai, deepinfra)
MAX_MODEL_LENGTH="8192"     # Default: 32768. Modify only if the model does not support 32768. (e.g. 8192 for Llama-3.1 and Gemma-2.0) 
MAX_COMPLETION_TOKENS="-1"  # Default: -1 (Not specified). Specify only if facing some critical issues (e.g. repetition)
PRIORITY="-5"               # Default: -5. Specify only if you want to prioritize your job. ([low] -5, -4, -3 [high])

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
  [[ -z $task_name || -z $h_rt || -z $result_dir ]] && { echo "❌ unknown task ${lang}_${task}"; exit 1; }

  # Set an outdir 
  OUTDIR="${RESULTS_DIR}/${result_dir}"
  mkdir -p "${OUTDIR}"

  # Submit a job
  ${QSUB_BASE[@]} -l h_rt="${h_rt}" -o "${OUTDIR}" -e "${OUTDIR}" "${SCRIPTS_DIR}/evaluate_${task_framework}.sh" \
    "${task_name}" "${NODE_KIND}" "${MODEL_NAME}" "${REPO_PATH}" "${SYSTEM_MESSAGE}" "${PROVIDER}" "${MAX_MODEL_LENGTH}" "${MAX_COMPLETION_TOKENS}"
}

########################################################

# Submit tasks
echo "🚀 Submitting tasks..."

## Japanese
# qsub_task ja gpqa
# qsub_task ja jemhopqa
# qsub_task ja math_100
# qsub_task ja mmlu
# qsub_task ja mmlu_prox
# qsub_task ja mtbench
# qsub_task ja wmt20_en_ja
# qsub_task ja wmt20_ja_en
qsub_task ja humaneval

## English
# qsub_task en gpqa
# qsub_task en hellaswag
# qsub_task en mmlu_prox
# qsub_task en mtbench

## Optional
# qsub_task ja jemhopqa_cot