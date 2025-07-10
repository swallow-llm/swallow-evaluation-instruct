PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True 
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_USE_V1=0
export CUDA_VISIBLE_DEVICES=0,1
export NUM_GPUS=2
MODEL=tokyotech-llm/Llama-3.1-Swallow-8B-Instruct-v0.5-stage1
MAX_MODEL_LENGTH=8192
TEMPERATURE=0.6
TOP_P=0.95

MODEL_ARGS="pretrained=$MODEL,dtype=bfloat16,tensor_parallel_size=$NUM_GPUS,max_model_length=$MAX_MODEL_LENGTH,gpu_memory_utilization=0.8,generation_parameters={max_new_tokens:$MAX_MODEL_LENGTH,temperature:$TEMPERATURE,top_p:$TOP_P}"
OUTPUT_DIR=data/evals/

# MATH-500
lighteval vllm $MODEL_ARGS "lighteval|math_500|0|0" \
    --use-chat-template \
    --output-dir $OUTPUT_DIR \
    --save-details \
    --system-prompt "detailed thinking on"
