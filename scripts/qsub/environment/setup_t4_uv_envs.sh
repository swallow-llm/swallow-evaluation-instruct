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

if command -v uv &> /dev/null; then
    echo "🛠️ uv is already installed: $(command -v uv) (version: $(uv --version))"
else
    echo "🌐 uv is not found. Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh || echo "❌ Failed to install uv."
    export PATH="$HOME/.local/bin:$PATH"
    hash -r
    command -v uv &>/dev/null || echo "❌ uv still not found on PATH after installation"
    echo "🛠️ uv is successfully installed: $(command -v uv) (version: $(uv --version))"
fi

if uv python list --only-installed | grep -q "3\.10\.14"; then
    echo "🛠️ Python 3.10.14 is already installed under uv."
else
    echo "📦 Installing Python 3.10.14 via uv..."
    uv python install 3.10.14
    uv python pin 3.10.14
    echo "🛠️ Python 3.10.14 is successfully installed under uv."
fi

echo "🔧 Creating shared virtual environment..."
uv venv "${REPO_PATH}/.common_envs"
source "${REPO_PATH}/.common_envs/bin/activate"

echo "📥 Installing utilities..."
uv pip install --upgrade pip setuptools wheel
uv pip install pre-commit huggingface_hub[cli]
uv pip install pandas pyarrow fastparquet ipykernel

echo "🤗 Logging in to HuggingFace..."
hf auth login --token $HF_TOKEN
deactivate

echo "🔗 Added virtual-env bin dir to PATH in .bashrc (if not already present)"
grep -qxF 'export PATH="'"${REPO_PATH}"'/.common_envs/bin:$PATH"' "$HOME/.bashrc" || echo 'export PATH="'"${REPO_PATH}"'/.common_envs/bin:$PATH"' >> "$HOME/.bashrc"
grep -qxF 'export PATH="$HOME/.local/bin:$PATH"' "$HOME/.bashrc" || echo 'export PATH="$HOME/.local/bin:$PATH"' >> "$HOME/.bashrc"


# ----------------------------------------------------------------------
# 4. Manage execution permissions (Only for ABCI)
# ----------------------------------------------------------------------
if [[ -d /groups/gag51395 ]]; then
  echo "🔧 Managing execution permissions..."
  files_to_chmod=(
      "$REPO_PATH/scripts/qsub/common_funcs.sh"
      "$REPO_PATH/scripts/qsub/evaluate_lighteval.sh"
      "$REPO_PATH/scripts/qsub/qsub_all.sh"
  )
  for file in "${files_to_chmod[@]}"; do
    chgrp ${ABCI_GROUP} "$file"
    chmod 750 "$file"
    echo "- Set execution permission for $file (chgrp: ${ABCI_GROUP}, chmod: 750)"
  done
fi


echo "✅ Environment was successfully created!"
