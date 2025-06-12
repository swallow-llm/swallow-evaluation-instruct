PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True 
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_USE_V1=0
export CUDA_VISIBLE_DEVICES=4,5
export NUM_GPUS=2
MODEL="tokyotech-llm/Llama-3.1-Swallow-8B-Instruct-v0.3"
SYSTEM_PROMPT="あなたは誠実で優秀な日本人のアシスタントです。"
MAX_MODEL_LENGTH=8192
TEMPERATURE=0.0

MODEL_ARGS="pretrained=$MODEL,dtype=bfloat16,tensor_parallel_size=$NUM_GPUS,max_model_length=$MAX_MODEL_LENGTH,gpu_memory_utilization=0.7,generation_parameters={temperature:$TEMPERATURE}"
OUTPUT_DIR=data/evals

# WMT20 En-Ja
lighteval vllm $MODEL_ARGS "swallow|wmt20:en-ja|0|0" \
--output-dir $OUTPUT_DIR \
--use-chat-template \
--save-details \
--system-prompt $SYSTEM_PROMPT

# WMT20 Ja-En
lighteval vllm $MODEL_ARGS "swallow|wmt20:ja-en|0|0" \
--output-dir $OUTPUT_DIR \
--use-chat-template \
--save-details \
--system-prompt $SYSTEM_PROMPT
