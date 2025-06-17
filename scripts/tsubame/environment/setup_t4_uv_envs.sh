#!/bin/bash
set -euo pipefail

# ----------------------------------------------------------------------
# 1. Load user-specific configuration
# ----------------------------------------------------------------------
CONFIG_FILE="$(dirname "$0")/../../../.env"
if [[ -f "$CONFIG_FILE" ]]; then
    source "$CONFIG_FILE" || {
        echo "❌ Failed to source ${CONFIG_FILE}"
        exit 1
    }
else
    echo "❌ Config file not found: ${CONFIG_FILE}"
    exit 1
fi


# ----------------------------------------------------------------------
# 2. Ensure required environment variables are set & customised
# ----------------------------------------------------------------------
declare -A defaults=(
  ["REPO_PATH"]="/gs/fs/tga-okazaki/{your_name}/swallow-evaluation-private"
  ["HUGGINGFACE_CACHE"]="/gs/bs/tga-okazaki/{your_name}/cache/huggingface"
  ["UV_CACHE"]="/gs/bs/tga-okazaki/{your_name}/cache/uv"
  ["OPENAI_API_KEY"]="sk-iloveswallow"
  ["HF_TOKEN"]="hf_iloveswallow"
)
for var in "${!defaults[@]}"; do
  val="${!var:-}"
  if [[ -z "$val" ]]; then
    # 環境変数が空文字の場合はエラー
    echo "❌ Environment variable $var is not set."
    exit 1
  fi
  if [[ "$val" == "${defaults[$var]}" ]]; then
    # 環境変数がデフォルト文字のままの場合はエラー
    echo "❌ Environment variable $var is still using default value (${defaults[$var]})."
    exit 1
  fi
done


# ----------------------------------------------------------------------
# 3. Install uv & create shared virtual-env
# ----------------------------------------------------------------------
cd $REPO_PATH

export UV_CACHE_DIR=$UV_CACHE
echo "💰 Set UV_CACHE_DIR as \`${UV_CACHE_DIR}.\`"

echo "🌐 Installing uv..."
curl -LsSf https://astral.sh/uv/install.sh | sh

echo "📦 Installing Python 3.10.14 via uv..."
uv python install 3.10.14
uv python pin 3.10.14

echo "🔧 Creating shared virtual environment..."
uv venv "${REPO_PATH}/.common_envs"
source "${REPO_PATH}/.common_envs/bin/activate"

echo "📥 Installing utilities..."
uv pip install --upgrade pip setuptools wheel
uv pip install pre-commit huggingface_hub[cli]
deactivate

echo "🔗 Added virtual-env bin dir to PATH in .bashrc"
echo 'export PATH="'"${REPO_PATH}"'/.common_envs/bin:$PATH"' >> "$HOME/.bashrc"

echo "✅ Environment was successfully created!"
