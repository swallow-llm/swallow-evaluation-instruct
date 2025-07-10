PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True 
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_USE_V1=0
export CUDA_VISIBLE_DEVICES=4
export NUM_GPUS=1
MODEL=google/gemma-2-2b
MAX_MODEL_LENGTH=8192
TEMPERATURE=0.0

MODEL_ARGS="pretrained=$MODEL,dtype=bfloat16,tensor_parallel_size=$NUM_GPUS,max_model_length=$MAX_MODEL_LENGTH,gpu_memory_utilization=0.8,generation_parameters={max_new_tokens:$MAX_MODEL_LENGTH,temperature:$TEMPERATURE}"
OUTPUT_DIR=data/evals/$MODEL

# mmlu:business_ethics
lighteval vllm $MODEL_ARGS "helm|mmlu:business_ethics|2|0" \
    --output-dir $OUTPUT_DIR \
    --max-samples 10 \
    --save-details
