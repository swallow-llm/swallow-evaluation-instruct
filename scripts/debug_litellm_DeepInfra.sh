# Llama 4 Maverick を DeepInfra API で評価する
API_KEY="DeepInfraのAPI Key"
API_KEY="gvvZIq9k15zd2GShkiol65rVtDDFuQQO"
MODEL_NAME="openai/meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8"
BASE_URL="https://api.deepinfra.com/v1/openai" # OpenAI API の URL
OUTPUT_DIR=data/evals/

# export LITELLM_LOG="INFO"

# LiveCodeBench test-run
lighteval endpoint litellm \
    "model=$MODEL_NAME,api_key=$API_KEY,base_url=$BASE_URL,generation_parameters={temperature:0.2,top_p:0.95,max_n:4}" \
    "swallow|lcb:codegeneration_v6|0|0" \
    --use-chat-template \
    --output-dir $OUTPUT_DIR \
    --max-samples 2

# GPQA test-run
lighteval endpoint litellm \
    "model=$MODEL_NAME,api_key=$API_KEY,base_url=$BASE_URL" \
    "swallow|gpqa:diamond|0|0" \
    --use-chat-template \
    --output-dir $OUTPUT_DIR \
    --max-samples 2