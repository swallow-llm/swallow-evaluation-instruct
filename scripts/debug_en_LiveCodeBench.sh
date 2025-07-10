PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True 
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_USE_V1=0
export CUDA_VISIBLE_DEVICES=1
export NUM_GPUS=1
MODEL="deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
# SYSTEM_PROMPT="あなたは誠実で優秀な日本人のアシスタントです。"
MAX_MODEL_LENGTH=16384
TEMPERATURE=0.6
TOP_P=0.95

MODEL_ARGS="pretrained=$MODEL,dtype=bfloat16,reasoning_parser=deepseek_r1,tensor_parallel_size=$NUM_GPUS,max_model_length=$MAX_MODEL_LENGTH,gpu_memory_utilization=0.9,generation_parameters={temperature:$TEMPERATURE,top_p:$TOP_P}"
OUTPUT_DIR=data/evals

# mmlu:business_ethics
lighteval vllm $MODEL_ARGS "swallow|lcb:codegeneration_v5|0|0" \
    --output-dir $OUTPUT_DIR \
    --save-details \
    --use-chat-template \
    --max-samples 10