#!/bin/bash
#$ -cwd
#& -m abe
#$ -N JHumanEval
set -e

# module load
. /etc/profile.d/modules.sh
module load cuda/12.1.0
module load cudnn/9.0.0

# receive args
USER_CONFIG_PATH=$1
RAW_OUTPUTS_DIR=$2
AGGREGATED_OUTPUTS_DIR=$3
MODEL_NAME=$4
SYSTEM_MESSAGE=${5:-""}
MAX_NEW_TOKEN=${6:-""}  # Specify only when facing some critical issues (e.g. repetition)
MAX_MODEL_LENGTH=32768  # Though the default value is 32768, some models may not support it. Then change it to its max_position_embeddings (e.g. 8192 for Llama-3.1 and Gemma-2.0).

# load and set env
source $USER_CONFIG_PATH
export REPO_PATH=$REPO_PATH
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export HUGGINGFACE_HUB_CACHE=$HUGGINGFACE_CACHE
export HF_HOME=$HUGGINGFACE_CACHE
export HF_TOKEN=$HF_TOKEN

# load venv
source "$REPO_PATH/lighteval/.venv/bin/activate"

# model setting
## temperature and top_p are set to 0.2 and 0.95, respectively, regarding to "評価方法_swallow-eval"
NUM_GPUS=1
TEMPERATURE=0.2
TOP_P=0.95
GPU_MEMORY_UTILIZATION=0.9
if [ -n "$MAX_NEW_TOKEN" ]; then
    echo ">> Using max_new_tokens=$MAX_NEW_TOKEN"
    GEN_PARAMS="max_new_tokens:$MAX_NEW_TOKEN,temperature:$TEMPERATURE,top_p:$TOP_P"
else
    echo ">> Using default max_new_tokens"
    GEN_PARAMS="temperature:$TEMPERATURE,top_p:$TOP_P"
fi
MODEL_ARGS_L="pretrained=$MODEL_NAME,dtype=bfloat16,tensor_parallel_size=$NUM_GPUS,max_model_length=$MAX_MODEL_LENGTH,gpu_memory_utilization=$GPU_MEMORY_UTILIZATION,generation_parameters={${GEN_PARAMS}}"

# task definition
TASK_DEF="swallow|swallow_jhumaneval|0|0"

# evaluate
cd "${REPO_PATH}/lighteval"
echo "Task: ${TASK_DEF}"
start_time=$(date +%s)
lighteval vllm $MODEL_ARGS_L $TASK_DEF \
    --system-prompt "${SYSTEM_MESSAGE}" \
    --use-chat-template \
    --output-dir "${RAW_OUTPUTS_DIR}" \
    --save-details
end_time=$(date +%s)
elapsed=$(( end_time - start_time ))
echo "Elapsed time: ${elapsed} seconds"

# aggregate
cd ${REPO_PATH}
python scripts/aggregate_results.py $MODEL_NAME "${RAW_OUTPUTS_DIR}/results/${MODEL_NAME}" $AGGREGATED_OUTPUTS_DIR
