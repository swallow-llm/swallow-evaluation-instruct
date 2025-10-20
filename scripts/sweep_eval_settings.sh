#!/bin/bash
set -e

PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_USE_V1=1
export CUDA_VISIBLE_DEVICES=4,5
export NUM_GPUS=2

export REQUEST_TIMEOUT=600000 # litellm default timeout [sec]
export LITELLM_CONCURRENT_CALLS=70

HOST=192.168.1.108
PORT=9003
BASE_URL="http://$HOST:$PORT/v1"
API_KEY="dummy"

# SYSTEM_PROMPT="あなたは誠実で優秀な日本人のアシスタントです。"
MAX_MODEL_LENGTH=32768
MAX_NUM_SEQS=32

#―― 探索したい生成設定を列挙する（設定ごとに試行される） ―――――――――――
GENERATION_CONFIGS=(
  "temperature=0.2;top_p=0.95"
  "temperature=0.4;top_p=0.95"
  "temperature=0.6;top_p=0.95"
  "temperature=0.8;top_p=0.95"
  "temperature=1.0;top_p=0.95"
)

#―― 評価したいベンチマークを列挙する ――――――――――――――――――――
BENCHMARKS=(
  # "swallow|math_500_N16|0|0"
  "swallow|aime_N16|0|0"
)

#―― ここに評価したいモデルを並べる ―――――――――――――――――――――――――
# 左から順に HF ID, reasoning_parser, system_message, max-model-len である．
ENTRIES=(
  "Qwen/Qwen3-8B-Base,qwen3,,"
  #"google/gemma-3-12b-it,,,"
  #"deepseek-ai/DeepSeek-R1-Distill-Llama-8B,deepseek_r1,,"
  #"meta-llama/Meta-Llama-3.1-8B-Instruct,,,"
  #"Qwen/Qwen2.5-7B-Instruct,,,"
  "Qwen/Qwen3-8B,qwen3,,"
  #"Qwen/Qwen3-0.6B,qwen3,,"
  #"tokyotech-llm/Llama-3.1-Swallow-8B-Instruct-v0.5,,あなたは誠実で優秀な日本人のアシスタントです。,8192"
  #"llm-jp/llm-jp-3.1-13b-instruct4,,,4096"
  #"openai/gpt-oss-20b,,,"
  #"tokyotech-llm/Qwen3-Swallow-8B-v0.1-LR1.5E-5-iter0025000,qwen3,,"
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

    # vllm serveをバックグラウンドで起動
    REASONING_PARSER_ARG=""
    if [[ -n "$REASONING_PARSER" ]]; then
        REASONING_PARSER_ARG="--reasoning-parser $REASONING_PARSER"
    fi
    # max-model-lenをENTRYごとに切り替え
    MAX_MODEL_LEN_ARG="--max-model-len ${MAX_MODEL_LENGTH}"
    if [[ -n "$ENTRY_MAX_MODEL_LEN" ]]; then
        MAX_MODEL_LEN_ARG="--max-model-len ${ENTRY_MAX_MODEL_LEN}"
    fi
    vllm serve "$MODEL" \
      --tensor-parallel-size=$NUM_GPUS \
      $MAX_MODEL_LEN_ARG \
      --host "$HOST" \
      --port "$PORT" \
      $REASONING_PARSER_ARG \
      --gpu-memory-utilization 0.9 \
      > "vllm_${SANITIZED_NAME}.log" 2>&1 &
      # --max-num-seqs ${MAX_NUM_SEQS} \
    VLLM_PID=$!

    # ポートが開くまで最大300秒待機（10秒間隔で30回）
    for i in {1..60}; do
        if nc -z "$HOST" "$PORT"; then
            echo "vllm serve is up."
            break
        fi
        echo "Waiting for vllm serve to start... ($i/60)"
        sleep 10
    done

    if ! nc -z "$HOST" "$PORT"; then
        echo "Error: vllm serve did not start on $HOST:$PORT"
        kill $VLLM_PID
        wait $VLLM_PID 2>/dev/null
        exit 1
    fi

    # lighteval実行
    SYSTEM_MESSAGE_ARG=""
    if [[ -n "$SYSTEM_MESSAGE" ]]; then
        SYSTEM_MESSAGE_ARG="--system-prompt \"${SYSTEM_MESSAGE}\""
    fi

    set +e

    FAILURE=0
    for GENERATION_CONFIG in "${GENERATION_CONFIGS[@]}"; do
        IFS=';' read -r -a PARAMS <<< "$GENERATION_CONFIG"

        GENERATION_PARAMETERS="{"
        HUMAN_READABLE=()
        for (( i = 0; i < ${#PARAMS[@]}; i++ )); do
            PAIR="${PARAMS[$i]}"
            KEY="${PAIR%%=*}"
            VALUE="${PAIR#*=}"
            HUMAN_READABLE+=("${KEY}=${VALUE}")
            GENERATION_PARAMETERS+="${KEY}:${VALUE}"
            if (( i < ${#PARAMS[@]} - 1 )); then
                GENERATION_PARAMETERS+=",";
            fi
        done
        GENERATION_PARAMETERS+="}"

        echo "  ↳ Generation settings: ${HUMAN_READABLE[*]}"

        for BENCHMARK in "${BENCHMARKS[@]}"; do
            echo "    ↳ Benchmark $BENCHMARK"
            lighteval endpoint litellm \
              "model=hosted_vllm/$MODEL,api_key=$API_KEY,base_url=$BASE_URL,generation_parameters=$GENERATION_PARAMETERS" \
              "$BENCHMARK" \
              $SYSTEM_MESSAGE_ARG \
              --save-details \
              --use-chat-template \
              --output-dir "$OUTPUT_DIR"
            EXIT_CODE=$?
            if [[ $EXIT_CODE -ne 0 ]]; then
                FAILURE=$EXIT_CODE
                break 2
            fi
        done
    done

    set -e

    # vllm serveを必ず終了
    kill $VLLM_PID
    wait $VLLM_PID 2>/dev/null

    if [[ $FAILURE -ne 0 ]]; then
        echo "lighteval failed (exit code $FAILURE)"
        exit $FAILURE
    fi

    echo "Finished evaluating $MODEL"
done
