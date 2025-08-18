#!/bin/bash
set -e

# 1. uv のインストール
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"


# 2. uv 内への python のインストール
uv python install 3.10.14
uv python pin 3.10.14


# 3. 共通パッケージのインストール
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR/.."

uv venv ".common_envs"
source ".common_envs/bin/activate"

uv pip install --upgrade pip setuptools wheel
uv pip install huggingface_hub[cli]

# HuggingFace のログインを行う場合は，自身の huggingface token を書き，コメントアウトを外してから実行すること．
# hf auth login --token (ここに huggingface token を書く)

deactivate


# 4. パスの追加
echo 'export PATH="/.common_envs/bin:$PATH"' >> "$HOME/.bashrc"
echo 'export PATH="$HOME/.local/bin:$PATH"' >> "$HOME/.bashrc"