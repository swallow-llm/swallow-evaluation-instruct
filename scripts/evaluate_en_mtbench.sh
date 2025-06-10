PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_USE_V1=0
export CUDA_VISIBLE_DEVICES=2,4
export NUM_GPUS=2
MODEL=$1
SYSTEM_MESSAGE=$2
MAX_MODEL_LENGTH=32768
# MAX_NEW_TOKENS=8192

# MODEL_ARGS="pretrained=$MODEL,dtype=bfloat16,tensor_parallel_size=$NUM_GPUS,max_model_length=$MAX_MODEL_LENGTH,gpu_memory_utilization=0.8,generation_parameters={max_new_tokens:$MAX_MODEL_LENGTH}"
MODEL_ARGS="pretrained=$MODEL,dtype=bfloat16,tensor_parallel_size=$NUM_GPUS,max_model_length=$MAX_MODEL_LENGTH,gpu_memory_utilization=0.8"
RAW_OUTPUT_DIR="./outputs"
AGGREGATED_OUTPUTS_DIR="./results/${MODEL}"

# SYSTEM_MESSAGEが指定されていない場合は、使わない
if [ -z "$SYSTEM_MESSAGE" ]; then
    uv run lighteval vllm $MODEL_ARGS "swallow|english_mt_bench|0|0" \
        --use-chat-template \
        --output-dir $RAW_OUTPUT_DIR \
        --save-details
else
    echo "Using system message: $SYSTEM_MESSAGE"
    uv run lighteval vllm $MODEL_ARGS "swallow|english_mt_bench|0|0" \
        --use-chat-template \
        --output-dir $RAW_OUTPUT_DIR \
        --system-prompt "$SYSTEM_MESSAGE" \
        --save-details
fi
