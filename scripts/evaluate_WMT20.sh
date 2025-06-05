PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True 
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_USE_V1=0
export CUDA_VISIBLE_DEVICES=1,2
export NUM_GPUS=2
MODEL="tokyotech-llm/Llama-3.1-Swallow-8B-Instruct-v0.3"
# SYSTEM_PROMPT="\"You are a helpful assistant.\""
MAX_MODEL_LENGTH=8192
MAX_NEW_TOKENS=2048
TEMPERATURE=0.0

MODEL_ARGS="pretrained=$MODEL,dtype=bfloat16,tensor_parallel_size=$NUM_GPUS,max_model_length=$MAX_MODEL_LENGTH,gpu_memory_utilization=0.8,generation_parameters={max_new_tokens:$MAX_NEW_TOKENS,temperature:$TEMPERATURE}"
OUTPUT_DIR=data/evals

# WMT20 En-Ja
lighteval vllm $MODEL_ARGS "swallow|wmt20:en-ja|0|0" \
--output-dir $OUTPUT_DIR \
--max-samples 10 \
--use-chat-template \
--save-details

# WMT20 Ja-En
lighteval vllm $MODEL_ARGS "swallow|wmt20:ja-en|0|0" \
--output-dir $OUTPUT_DIR \
--max-samples 10 \
--use-chat-template \
--save-details
