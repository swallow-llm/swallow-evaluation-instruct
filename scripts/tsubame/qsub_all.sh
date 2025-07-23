#!/usr/bin/env bash
# How-to-use: bash scripts/tsubame/qsub_all.sh
set -euo pipefail

########################################################

# Set Args
## Common Settings
NODE_KIND=""                # A node kind to use. Tsubame: ["node_q", "node_f", "cpu_16"], Local: ["cpu", "gpu_*" (*: GPU number)]
MODEL_NAME=""               # A model name (HuggingFace ID) to use.

## Special Settings
PROVIDER="vllm"             # Default: "vllm". A provider to host the model. ["vllm", "openai", "deepinfra"]
CUSTOM_SETTINGS=""          # Default: "". A custom setting name to use. (e.g. "reasoning", "coding", "flashattn_incompatible")
PREDOWNLOAD_MODEL="true"    # Default: "true". A pre-download a model before qsub.

## Environmental Settings
SERVICE=""                  # A service to use. ["tsubame", "local"]
PRIORITY="-5"               # Default: "-5". A priority of the job. Note that double priority is double cost. ["-5", "-4", "-3"] (Only for TSUBAME)
CUDA_VISIBLE_DEVICES=""     # Default: "". A CUDA_VISIBLE_DEVICES to use. [e.g. "0,1"] (Only for local)

########################################################

# Load task-definition
source "$(dirname "$0")/conf/load_config.sh"

# Load .env and define dirs
source "$(dirname "$0")/../../.env"
case $CUSTOM_SETTINGS in
    "") CUSTOM_SETTINGS_SUBDIR="" ;;
    *) CUSTOM_SETTINGS_SUBDIR="/${CUSTOM_SETTINGS}" ;;
esac
case $PROVIDER in
    openai) PROVIDER_SUBDIR="" ;;
    vllm) PROVIDER_SUBDIR="hosted_vllm/" ;;
    deepinfra) PROVIDER_SUBDIR="deepinfra/" ;;
    *) echo "❌ unknown provider ${PROVIDER}"; exit 1 ;;
esac
RESULTS_DIR="${REPO_PATH}/results/${PROVIDER_SUBDIR}${MODEL_NAME}${CUSTOM_SETTINGS_SUBDIR}"
SCRIPTS_DIR="${REPO_PATH}/scripts/tsubame"

# Optional Args
OPTIONAL_ARGS=""
if [[ -n "${CUSTOM_SETTINGS}" ]]; then
  OPTIONAL_ARGS="${OPTIONAL_ARGS} --custom-settings '${CUSTOM_SETTINGS}'"
fi
if [[ -n "${PROVIDER}" ]]; then
  OPTIONAL_ARGS="${OPTIONAL_ARGS} --provider '${PROVIDER}'"
fi

# Pre-download the model
if [ ${PREDOWNLOAD_MODEL} = "true" ]; then
  if [ ${PROVIDER} != "vllm" ]; then
    echo "☠️ Error: Pre-downloading is only supported for vLLM. Please set PREDOWNLOAD_MODEL=false."
    exit 1
  fi
  source "${REPO_PATH}/.common_envs/bin/activate"
  echo "🤖 Downloading ${MODEL_NAME} ..."
  huggingface-cli download $MODEL_NAME --cache-dir $HUGGINGFACE_CACHE --token $HF_TOKEN
  deactivate
  echo "✅ \`${MODEL_NAME}\` was successfully downloaded at \`${HUGGINGFACE_CACHE}\`."
else
  echo "⏭️ Skipping pre-downloading model."
fi

# Define qsub-function
last_submit_time=""
source "${REPO_PATH}/scripts/tsubame/common_funcs.sh"
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

  # Safety check for local jobs
  if [[ "${SERVICE}" == "local" && -n "${last_submit_time}" ]]; then
    if [[ $(( (date +%s) - last_submit_time )) -lt 30 ]]; then
      echo "💀 Error: Local jobs cannot be submitted continuously. Please reset CUDA_VISIBLE_DEVICES appropriately and submit one task at a time."
      exit 1
    fi
  fi

  # Submit a job
  local job_name="${lang}_${task}"
  case $SERVICE in
    "tsubame")
      qsub -g tga-okazaki -l ${NODE_KIND}=1 -p ${PRIORITY}-N "${job_name}" -l h_rt="${h_rt}" -o "${OUTDIR}" -e "${OUTDIR}" "${SCRIPTS_DIR}/evaluate_${task_framework}.sh" \
        --task-name "${task_name}" --node-kind "${NODE_KIND}" --model-name "${MODEL_NAME}" --repo-path "${REPO_PATH}" --service "${SERVICE}" ${OPTIONAL_ARGS}
      ;;

    "local")
      get_random_job_id
      local session_name="${job_name}_${JOB_ID}"
      tmux new-session -d -s "${session_name}" bash -c "
        set -euo pipefail
        export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} && bash "${SCRIPTS_DIR}/evaluate_${task_framework}.sh" \
          --task-name '${task_name}' --node-kind '${NODE_KIND}' --model-name '${MODEL_NAME}' --repo-path '${REPO_PATH}' --service '${SERVICE}' --custom-job-id '${JOB_ID}' ${OPTIONAL_ARGS} \
          > '${OUTDIR}/${job_name}.o${JOB_ID}' 2> '${OUTDIR}/${job_name}.e${JOB_ID}'
      "
      echo "✅ Local job ${job_name} was successfully submitted to tmux session ${session_name}."
  esac
  last_submit_time=$(date +%s)
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
qsub_task en hellaswag
qsub_task en mtbench
qsub_task en gpqa_diamond
qsub_task en math_500
qsub_task en aime_2024_2025
qsub_task en livecodebench_v5_v6
qsub_task en mmlu
qsub_task en mmlu_pro
qsub_task en mmlu_prox

## Optional
# qsub_task ja jemhopqa
