PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True 
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_USE_V1=0
export CUDA_VISIBLE_DEVICES=4,6
export NUM_GPUS=2
MODEL="Qwen/Qwen3-8B"
MAX_MODEL_LENGTH=16384
# Officially recommended decoding parameters for Qwen3
TEMPERATURE=0.6
TOP_P=0.95

MODEL_ARGS="pretrained=$MODEL,dtype=bfloat16,tensor_parallel_size=$NUM_GPUS,max_model_length=$MAX_MODEL_LENGTH,gpu_memory_utilization=0.7,generation_parameters={temperature:$TEMPERATURE,top_p:$TOP_P}"
OUTPUT_DIR=data/evals

# AIME 24--25
lighteval vllm $MODEL_ARGS "swallow|aime|0|0" \
    --output-dir $OUTPUT_DIR \
    --save-details \
    --use-chat-template \
    --max-samples 10

# GPQA Diamond
lighteval vllm $MODEL_ARGS "swallow|gpqa:diamond|0|0" \
    --output-dir $OUTPUT_DIR \
    --save-details \
    --use-chat-template \
    --max-samples 10

# MATH-500
lighteval vllm $MODEL_ARGS "swallow|math_500|0|0" \
    --output-dir $OUTPUT_DIR \
    --save-details \
    --use-chat-template \
    --max-samples 10

# LiveCodeBench
TEMPERATURE=0.2
TOP_P=0.95
MODEL_ARGS="pretrained=$MODEL,dtype=bfloat16,tensor_parallel_size=$NUM_GPUS,max_model_length=$MAX_MODEL_LENGTH,gpu_memory_utilization=0.7,generation_parameters={temperature:$TEMPERATURE,top_p:${TOP_P}}"
lighteval vllm $MODEL_ARGS "swallow|lcb:codegeneration_v6|0|0" \
    --output-dir $OUTPUT_DIR \
    --save-details \
    --use-chat-template \
    --max-samples 10