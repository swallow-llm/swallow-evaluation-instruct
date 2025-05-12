PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True 
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_USE_V1=0
export CUDA_VISIBLE_DEVICES=1,2
export NUM_GPUS=1
MODEL=tokyotech-llm/Llama-3.1-Swallow-8B-Instruct-v0.3
MAX_MODEL_LENGTH=8192
MAX_NEW_TOKENS=2048
TEMPERATURE=0.0

MODEL_ARGS="pretrained=$MODEL,dtype=bfloat16,tensor_parallel_size=$NUM_GPUS,max_model_length=$MAX_MODEL_LENGTH,gpu_memory_utilization=0.8,generation_parameters={max_new_tokens:$MAX_NEW_TOKENS,temperature:$TEMPERATURE}"
OUTPUT_DIR=data/evals/$MODEL

# mmlu:business_ethics
lighteval vllm $MODEL_ARGS "swallow|jemhopqa|0|0" \
--output-dir $OUTPUT_DIR \
--use-chat-template \
--save-details
