PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_USE_V1=0
export CUDA_VISIBLE_DEVICES="4,5"
export NUM_GPUS=2
MODEL=$1
SYSTEM_PROMPT=$2
MAX_MODEL_LENGTH=8192
MAX_NEW_TOKENS=2048
TEMPERATURE=0.0

MODEL_ARGS="pretrained=$MODEL,dtype=bfloat16,tensor_parallel_size=$NUM_GPUS,max_model_length=$MAX_MODEL_LENGTH,gpu_memory_utilization=0.8,generation_parameters={max_new_tokens:$MAX_NEW_TOKENS,temperature:$TEMPERATURE}"
RAW_OUTPUT_DIR="./outputs"
OUTPUT_DIR="./results"

uv run lighteval vllm $MODEL_ARGS "swallow|mmlu_prox_japanese|0|0" \
    --system-prompt "${SYSTEM_PROMPT}" \
    --output-dir $RAW_OUTPUT_DIR \
    --use-chat-template \
    --save-details

python scripts/aggregate_results.py \
    $MODEL \
    "$RAW_OUTPUT_DIR/results/$MODEL" \
    "$OUTPUT_DIR/$MODEL"