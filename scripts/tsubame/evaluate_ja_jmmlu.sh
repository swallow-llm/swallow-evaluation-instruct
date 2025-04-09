#!/bin/bash
#$ -cwd

# module load
. /etc/profile.d/modules.sh
module load cuda/12.1.0
module load cudnn/9.0.0

REPO_PATH=$1
HUGGINGFACE_CACHE=$2
OUTPUT_DIR=$3
MODEL_NAME=$4
SYSTEM_MESSAGE=$5
MAX_CONTEXT_WINDOW=$6

export VLLM_WORKER_MULTIPROC_METHOD=spawn
export HUGGINGFACE_HUB_CACHE=$HUGGINGFACE_CACHE
export HF_HOME=$HUGGINGFACE_CACHE

NUM_GPUS=1
GPU_MEMORY_UTILIZATION=0.9

source "${REPO_PATH}/.venv_jmmlu/bin/activate"

JMMLU_SUBSETS=(
    'japanese_history' 
    'miscellaneous' 
    'security_studies' 
    'virology' 
    'nutrition' 
    'human_sexuality' 
    'college_mathematics' 
    'econometrics' 
    'computer_security' 
    'clinical_knowledge' 
    'machine_learning' 
    'high_school_chemistry' 
    'human_aging' 
    'logical_fallacies' 
    'sociology' 
    'high_school_european_history' 
    'high_school_statistics' 
    'high_school_physics' 
    'high_school_microeconomics' 
    'college_physics' 
    'anatomy' 
    'high_school_psychology' 
    'business_ethics' 
    'professional_psychology' 
    'college_medicine' 
    'elementary_mathematics' 
    'moral_disputes' 
    'marketing' 
    'high_school_macroeconomics' 
    'world_religions'
    'conceptual_physics'
    'professional_medicine'
    'prehistory'
    'high_school_mathematics'
    'international_law'
    'philosophy'
    'management'
    'high_school_computer_science'
    'medical_genetics'
    'college_computer_science'
    'public_relations'
    'professional_accounting'
    'abstract_algebra'
    'global_facts'
    'college_biology'
    'high_school_geography'
    'world_history'
    'high_school_biology'
    'college_chemistry'
    'electrical_engineering'
    'astronomy'
    'jurisprudence'
    'formal_logic'
)
TASK_LIST=""
for i in "${!JMMLU_SUBSETS[@]}"; do
    part="custom|swallow_jmmlu:${JMMLU_SUBSETS[$i]}|0|0"
    if [ $i -ne 0 ]; then
        TASK_LIST="${TASK_LIST},${part}"
    else
        TASK_LIST="${part}"
    fi
done
echo -e "Task list are followed:\n${TASK_LIST}\n"


TEMPERATURE=0.0
MODEL_ARGS_L="pretrained=$MODEL_NAME,dtype=bfloat16,tensor_parallel_size=$NUM_GPUS,max_model_length=$MAX_CONTEXT_WINDOW,gpu_memory_utilization=$GPU_MEMORY_UTILIZATION,generation_parameters={max_new_tokens:$MAX_CONTEXT_WINDOW,temperature:$TEMPERATURE}"
cd "${REPO_PATH}/lighteval"
start_time=$(date +%s)
lighteval vllm $MODEL_ARGS_L $TASK_LIST \
    --custom-tasks "${REPO_PATH}/lighteval/src/lighteval/tasks/swallow/jmmlu.py" \
    --system-prompt $SYSTEM_MESSAGE \
    --use-chat-template \
    --output-dir $OUTPUT_DIR \
    --save-details
end_time=$(date +%s)
execution_time=$((end_time - start_time))
echo "Total Time: ${execution_time} seconds"