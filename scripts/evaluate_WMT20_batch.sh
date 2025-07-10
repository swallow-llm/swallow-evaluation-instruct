#―― 共通環境変数 ――――――――――――――――――――――――――――――――――――
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_USE_V1=0
export CUDA_VISIBLE_DEVICES=4
export NUM_GPUS=1

MAX_MODEL_LENGTH=8192
MAX_NEW_TOKENS=2048
TEMPERATURE=0.0
TOP_P=1.0

#―― ここに評価したいモデルを並べる ―――――――――――――――――――――――――
MODELS=(
  "google/gemma-3-12b-it"
  "google/gemma-3-27b-it"
  "microsoft/phi-4"
  "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
  "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B"
  "meta-llama/Meta-Llama-3.1-8B-Instruct"
  "Qwen/Qwen2.5-7B-Instruct"
  "Qwen/QwQ-32B"
  "tokyotech-llm/Llama-3.1-Swallow-8B-Instruct-v0.3"
)

MODELS=(  
  "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
  "tokyotech-llm/Llama-3.1-Swallow-8B-Instruct-v0.3"
)

#―― 評価ループ ―――――――――――――――――――――――――――――――――――――――
for MODEL in "${MODELS[@]}"; do
  # 出力ディレクトリをモデルごとに分ける（/ や : を _ に変換）
  SANITIZED_NAME=$(echo "$MODEL" | tr '/:' '__')
  OUTPUT_DIR="data/evals/"

  MODEL_ARGS="pretrained=${MODEL},dtype=bfloat16,tensor_parallel_size=${NUM_GPUS},enforce_eager=true,max_model_length=${MAX_MODEL_LENGTH},gpu_memory_utilization=0.7,generation_parameters={temperature:${TEMPERATURE},top_p:${TOP_P},max_new_tokens:${MAX_NEW_TOKENS}}"

  echo "▶︎ Evaluating $MODEL …"
  lighteval vllm "${MODEL_ARGS}" "swallow|wmt20:en-ja|0|0,swallow|wmt20:ja-en|0|0" \
    --output-dir "${OUTPUT_DIR}" \
    --use-chat-template \
    --save-details
    # --max-samples 10
done

# lighteval vllm "${MODEL_ARGS}" "swallow|wmt20:en-ja|0|0" \

echo "✅ すべてのモデルの評価が完了しました。"