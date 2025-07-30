
#!/bin/bash

# ============================================================
# How to use:
# 1. 下記の設定を変更する。

#    - MODEL_ID: モデルIDを指定する。   :  e.g. "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B"
#    - SETTING_NAME: セッティング名を指定する。  : e.g. "reasoning", "default", "" <- 空文字列も可能
#    - HOSTING: ホスティングの種類を指定する。   : e.g. "hosted_vllm", "openai", "deepinfra"
HOSTING="hosted_vllm"
MODEL_ID="meta-llama/Llama-3.3-70B-Instruct"
SETTING_NAME=""



# 2. このスクリプトを実行する。 -> bash runtime.sh

# ========== Script to extract runtime from qsub jobs =========
cd "$(dirname "$0")"


# .envからREPO_PATHを取得
set -a
if [ -f "../../../.env" ]; then
  source "../../../.env"
fi
set +a



python3 extract_runtime.py \
  --repo_path "$REPO_PATH" \
  --hosting "$HOSTING" \
  --model_id "$MODEL_ID" \
  --setting_name "$SETTING_NAME" \
  # --return_seconds \


