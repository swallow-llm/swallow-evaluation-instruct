PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True 
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_USE_V1=0
export CUDA_VISIBLE_DEVICES=4,5
export NUM_GPUS=2
MODEL="google/gemma-2-2b-it"
# SYSTEM_PROMPT="あなたは誠実で優秀な日本人のアシスタントです。"
MAX_MODEL_LENGTH=8192
TEMPERATURE=0.0

MODEL_ARGS="pretrained=$MODEL,dtype=bfloat16,tensor_parallel_size=$NUM_GPUS,max_model_length=$MAX_MODEL_LENGTH,gpu_memory_utilization=0.7,generation_parameters={temperature:$TEMPERATURE}"
OUTPUT_DIR=data/evals

# mmlu:business_ethics
lighteval vllm $MODEL_ARGS "swallow|mifeval_ja|0|0" \
    --output-dir $OUTPUT_DIR \
    --save-details \
    --use-chat-template \
    --max-samples 10