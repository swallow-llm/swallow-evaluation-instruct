PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_USE_V1=0
export CUDA_VISIBLE_DEVICES=1,2
export NUM_GPUS=2
MODEL=$1
MAX_MODEL_LENGTH=$2
SYSTEM_MESSAGE=$3

MODEL_ARGS="pretrained=$MODEL,dtype=bfloat16,tensor_parallel_size=$NUM_GPUS,max_model_length=$MAX_MODEL_LENGTH,gpu_memory_utilization=0.8,generation_parameters={max_new_tokens:$MAX_MODEL_LENGTH}"
RAW_OUTPUT_DIR="./outputs"
AGGREGATED_OUTPUTS_DIR="./results/${MODEL}"

# GPQA JA
cd lighteval

# SYSTEM_MESSAGEが指定されていない場合は、使わない
if [ -z "$SYSTEM_MESSAGE" ]; then
    uv run lighteval vllm $MODEL_ARGS "swallow|japanese_mt_bench|0|0" \
        --use-chat-template \
        --output-dir $RAW_OUTPUT_DIR \
        --save-details
else
    echo "Using system message: $SYSTEM_MESSAGE"
    uv run lighteval vllm $MODEL_ARGS "swallow|japanese_mt_bench|0|0" \
        --use-chat-template \
        --output-dir $RAW_OUTPUT_DIR \
        --system-prompt "$SYSTEM_MESSAGE" \
        --save-details
fi