PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True 
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_USE_V1=0
export CUDA_VISIBLE_DEVICES=4,5
export NUM_GPUS=2
MODEL=$1
MAX_MODEL_LENGTH=16384
TEMPERATURE=0.0

MODEL_ARGS="pretrained=$MODEL,dtype=bfloat16,tensor_parallel_size=$NUM_GPUS,max_model_length=$MAX_MODEL_LENGTH,gpu_memory_utilization=0.8,generation_parameters={max_new_tokens:$MAX_MODEL_LENGTH,temperature:$TEMPERATURE}"
OUTPUT_DIR=data/evals/$MODEL

# MATH-500
lighteval vllm $MODEL_ARGS "swallow|math_100_japanese|0|0" \
    --use-chat-template \
    --output-dir $OUTPUT_DIR \
    --save-details
