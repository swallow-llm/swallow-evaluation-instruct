PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True 
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_USE_V1=0
export CUDA_VISIBLE_DEVICES=6,7
export NUM_GPUS=2
MODEL=$1
SYSTEM_MESSAGE=$2
MAX_MODEL_LENGTH=32768
TEMPERATURE=0.0

MODEL_ARGS="pretrained=$MODEL,dtype=bfloat16,tensor_parallel_size=$NUM_GPUS,max_model_length=$MAX_MODEL_LENGTH,gpu_memory_utilization=0.8,generation_parameters={temperature:$TEMPERATURE}"
RAW_OUTPUT_DIR="./outputs"
OUTPUT_DIR="./results"

# SYSTEM_MESSAGEが指定されていない場合は、使わない
if [ -z "$SYSTEM_MESSAGE" ]; then
    uv run lighteval vllm $MODEL_ARGS "swallow|swallow_gpqa_en|0|0" \
        --use-chat-template \
        --output-dir $RAW_OUTPUT_DIR \
        --save-details
else
    echo "Using system message: $SYSTEM_MESSAGE"
    uv run lighteval vllm $MODEL_ARGS "swallow|swallow_gpqa_en|0|0" \
        --system-prompt "${SYSTEM_PROMPT}" \
        --use-chat-template \
        --output-dir $RAW_OUTPUT_DIR \
        --save-details
fi

python scripts/aggregate_results.py \
    $MODEL \
    "$RAW_OUTPUT_DIR/results/$MODEL" \
    "$OUTPUT_DIR/$MODEL"
