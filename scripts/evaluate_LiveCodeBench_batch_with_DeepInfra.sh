#!/bin/bash
set -e

export REQUEST_TIMEOUT=6000 # litellm default timeout [sec]
export LITELLM_CONCURRENT_CALLS=500

BASE_URL="https://api.deepinfra.com/v1/openai"
PROVIDER="deepinfra"
API_KEY="gvvZIq9k15zd2GShkiol65rVtDDFuQQO"

# SYSTEM_PROMPT="あなたは誠実で優秀な日本人のアシスタントです。"
MAX_MODEL_LENGTH=16384
TEMPERATURE=0.6
TOP_P=0.95

#―― ここに評価したいモデルを並べる ―――――――――――――――――――――――――
# 左から順に HF ID, reasoning_parser, system_message, max-model-len である．
ENTRIES=(
  "Qwen/Qwen3-14B,qwen3,,"
  "google/gemma-3-12b-it,,,"
  "microsoft/phi-4-reasoning-plus,deepseek_r1,,"
  # "deepseek-ai/DeepSeek-R1-Distill-Llama-8B,deepseek_r1,,"
  # "deepseek-ai/DeepSeek-R1-Distill-Llama-8B,,,"
  # "meta-llama/Meta-Llama-3.1-8B-Instruct,,,"
  # "Qwen/Qwen2.5-7B-Instruct,,,"
  # "tokyotech-llm/Llama-3.1-Swallow-8B-Instruct-v0.5,,あなたは誠実で優秀な日本人のアシスタントです。,8192"
)

# 左から順に HF ID, reasoning_parser, system_message, max-model-len である．
for ENTRY in "${ENTRIES[@]}"; do
    MODEL=$(echo "$ENTRY" | cut -d',' -f1)
    REASONING_PARSER=$(echo "$ENTRY" | cut -d',' -f2)
    SYSTEM_MESSAGE=$(echo "$ENTRY" | cut -d',' -f3)
    ENTRY_MAX_MODEL_LEN=$(echo "$ENTRY" | cut -d',' -f4)
    # 出力ディレクトリをモデルごとに分ける（/ や : を _ に変換）
    SANITIZED_NAME=$(echo "$MODEL" | tr '/:' '__')
    OUTPUT_DIR="data/evals/"

    echo "▶︎ Evaluating $MODEL …"
    
    # lighteval実行

    SYSTEM_MESSAGE_ARG=""
    if [[ -n "$SYSTEM_MESSAGE" ]]; then
        SYSTEM_MESSAGE_ARG="--system-prompt \"${SYSTEM_MESSAGE}\""
    fi

    set +e
    lighteval endpoint litellm \
      "model=$PROVIDER/$MODEL,api_key=$API_KEY,base_url=$BASE_URL,generation_parameters={temperature:$TEMPERATURE,top_p:$TOP_P,max_n:4}" \
      "swallow|lcb:codegeneration_v5|0|0" \
      $SYSTEM_MESSAGE_ARG \
      --save-details \
      --use-chat-template \
      --output-dir "$OUTPUT_DIR" \
      --output-subdir ""
      # --max-samples 3
    EXIT_CODE=$?
    set -e

    if [[ $EXIT_CODE -ne 0 ]]; then
        echo "lighteval failed (exit code $EXIT_CODE)"
        exit $EXIT_CODE
    fi

    echo "Finished evaluating $MODEL"
done
