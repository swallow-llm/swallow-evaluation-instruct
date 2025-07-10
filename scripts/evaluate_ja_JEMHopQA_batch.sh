#―― 共通環境変数 ――――――――――――――――――――――――――――――――――――
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_USE_V1=0
export CUDA_VISIBLE_DEVICES=0,1
export NUM_GPUS=2

MAX_MODEL_LENGTH=8192
MAX_NEW_TOKENS=2048
TEMPERATURE=0.0

#―― ここに評価したいモデルを並べる ―――――――――――――――――――――――――
MODELS=(
  "tokyotech-llm/Llama-3.1-Swallow-8B-Instruct-v0.5"
)

#―― 評価ループ ―――――――――――――――――――――――――――――――――――――――
for MODEL in "${MODELS[@]}"; do
  # 出力ディレクトリをモデルごとに分ける（/ や : を _ に変換）
  SANITIZED_NAME=$(echo "$MODEL" | tr '/:' '__')
  OUTPUT_DIR="data/evals/"

  MODEL_ARGS="pretrained=${MODEL},dtype=bfloat16,tensor_parallel_size=${NUM_GPUS},max_model_length=${MAX_MODEL_LENGTH},gpu_memory_utilization=0.8,generation_parameters={max_new_tokens:${MAX_NEW_TOKENS},temperature:${TEMPERATURE}}"

  echo "▶︎ Evaluating $MODEL …"
  lighteval vllm "${MODEL_ARGS}" "swallow|jemhopqa_cot|0|0" \
    --output-dir "${OUTPUT_DIR}" \
    --use-chat-template \
    --save-details
done

echo "✅ すべてのモデルの評価が完了しました。"