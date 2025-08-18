<!--
完成したらREADMEにマージします．
-->

# 大規模言語モデル評価フレームワーク swallow-evaluation-instruct Ver. 202508

このリポジトリでは，[Swallowプロジェクト](https://swallow-llm.github.io/) にて事後学習ずみモデルの評価を想定して開発した包括的評価フレームワーク swallow-evaluation-instruct（以下，"本フレームワーク"）を配布しています．  

swallow-evaluation-instruct は，HuggingFace社が開発した評価フレームワーク [lighteval](https://github.com/huggingface/lighteval) (v0.8.0) (© 2024 Hugging Face) をフォークして，日本語および英語のベンチマークの追加および利便性の改善をおこなったものです．
LLMの研究開発にご活用ください．

## 環境構築
本フレームワークでは環境管理に [uv](https://docs.astral.sh/uv/) を使用することを想定しています．
従って，環境構築は以下の流れで行ってください．

### 1. uv のインストール
まず uv をインストールしてください．
```sh
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"
```

### 2. uv 内への python のインストール
次に uv 内に python をインストールしてください．本フレームワークでは python [v3.10.14](https://www.python.org/downloads/release/python-31014/) を想定しています．
```sh
uv python install 3.10.14
uv python pin 3.10.14
```

### 3. 共通パッケージのインストール
本フレームワークでは評価の実行ごとに uv の仮想環境を一時的に構築することを想定していますが，
最低限の共通パッケージについては `.venv` の中にインストールします．
```sh
cd swallow-evaluation-instruct

uv venv ".common_envs"
source ".common_envs/bin/activate"

uv pip install --upgrade pip setuptools wheel
uv pip install huggingface_hub[cli]
```

ここでインストールした `huggingface_hub` を用いて huggingface へのログインを済ませておくのをお勧めします．
これにより，ライセンスの承諾が必要なモデルなどを評価することができるようになります．
```sh
hf auth login --token (ここに huggingface token を書く)
deactivate
```

### 4. パスの追加
最後に `~/.bashrc` に必要なパスを追加して uv に関する初期設定は終了です．
```sh
echo 'export PATH="/.common_envs/bin:$PATH"' >> "$HOME/.bashrc"
echo 'export PATH="$HOME/.local/bin:$PATH"' >> "$HOME/.bashrc"
```

## 実行方法
本フレームワークでは推論をvLLM [v0.9.2](https://github.com/vllm-project/vllm/releases/tag/v0.9.2)，評価をlighteval [v0.8.0](https://github.com/huggingface/lighteval/releases/tag/v0.8.0) で行っています．
そのため，vLLMを先にserveしてから評価する方法（推奨）と，lightevalの中で直接vLLMを呼び出す方法の二つが可能です．

### １）vLLM serve → lighteval（推奨）
vLLM serveコマンドで推論APIを立ててからlitellm経由でAPIを呼び出すことにより，[推論型モデルサポート](https://docs.vllm.ai/en/stable/features/reasoning_outputs.html)のようなvLLMの豊富な機能を活用しながら評価を実行できます．  

例えば，[tokyotech-llm/Llama-3.1-Swallow-8B-Instruct-v0.5](https://huggingface.co/tokyotech-llm/Llama-3.1-Swallow-8B-Instruct-v0.5)について[swallow|gpqa:diamond](BENCHMARKS#gpqadiamond)のタスクで評価したい場合は以下のように実行することができます．

```sh
MODEL_NAME="tokyotech-llm/Llama-3.1-Swallow-8B-Instruct-v0.5"
TASK_ID="swallow|gpqa:diamond"

# API_KEY="" 
# OpenAIの推論APIや，それと互換性のあるNVIDIA NIM，DeepInfra の推論APIを使用する場合には対応するAPIキーを指定．

cd swallow-evaluation-instruct-private

uv run --isolated --locked --extra vllm \
    vllm serve --model $MODEL_NAME \
        --host localhost \
        --port 8000 \
        # --reasoning-parser (vLLM 公式の reasoning parser はここで指定)

BASE_URL="http://localhost:8000/v1"
# HuggingFace のモデルをローカルでサーブする場合には "http://localhost:(ポート番号)/v1" を指定．
# 各種推論APIを用いる場合には，OpenAIであれば "https://api.openai.com/v1" を，
# DeepInfraであれば "https://api.deepinfra.com/v1/openai" のように適当なURLを指定する．

uv run --isolated --locked --extra lighteval \
    lighteval endpoint litellm \
        "model=$MODEL_NAME,base_url=$BASE_URL" \ # 必要なら api_key=$API_KEY を追加
        "${TASK_ID}|0|0" \
        --use-chat-template \
        --output-dir ./lighteval/outputs
```


### 2）lighteval → vLLM （非推奨）
[標準的な lighteval の実行方法](https://huggingface.co/docs/lighteval/quicktour)に則って，vLLMを直接起動して動かすことも可能です．  
ただし，**OpenAIなどの推論APIが使用できない**点や，**vLLM V0モードのみをサポート**（Ref. [vLLM V1](https://docs.vllm.ai/en/stable/usage/v1_guide.html)）している点，**vLLM実行時引数のサポートが不完全**である点から，先に紹介している[vLLMをserveしてから評価する方法](#１vllm-serve--lighteval推奨)を推奨します．

例えば，[tokyotech-llm/Llama-3.1-Swallow-8B-Instruct-v0.5](https://huggingface.co/tokyotech-llm/Llama-3.1-Swallow-8B-Instruct-v0.5)について[swallow|gpqa:diamond](BENCHMARKS#gpqadiamond)のタスクで評価したい場合は以下のように実行することができます．

```sh
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_USE_V1=0 # vLLM V0モードを指定する設定

MODEL_NAME="tokyotech-llm/Llama-3.1-Swallow-8B-Instruct-v0.5"
TASK_ID="swallow|gpqa:diamond"

MODEL_ARGS="pretrained=$MODEL_NAME"
# ここで dtype，tensor_parallel_size，max_model_length，gpu_memory_utlization，そして各種 generation_parameters なども指定することができる．

cd swallow-evaluation-instruct-private

uv run --isolated --locked --extra lighteval \
    lighteval vllm \
        $MODEL_ARGS \
        "${TASK_ID}|0|0" \
        --use-chat-template \
        --output-dir ./lighteval/outputs \
        # --reasoning-parser (reasoning parser はここで指定)
        # --system-prompt （System prompt はここで指定）
```


### 評価の実行に関するTips
#### 1. vLLM で公式にサポートされていない Reasoning parser を使う方法．．
reasoning-parser を指定することにより，モデルの出力から推論過程が取り除かれた最終出力部分のみを採点の対象とすることができ，推論型モデルを適切に評価することができます．（Ref. [vLLM Doc: Reasoning Outputs](https://docs.vllm.ai/en/stable/features/reasoning_outputs.html)）

本フレームワークでは[vLLM を予め serve する方法](#１vllm-serve--lighteval推奨)では vLLM==0.9.2，[lighteval から直接 vLLM を呼び出す方法](#2lighteval--vllm-非推奨)では vLLM==0.9.1 が使用されるため，https://github.com/vllm-project/vllm/tree/v0.9.2/vllm/reasoning から使用することができる vLLM 公式の reasoning parser を確認することができます．

ここで，[nvidia/Llama-3.1-Nemotron-Nano-8B-v1](https://huggingface.co/nvidia/Llama-3.1-Nemotron-Nano-8B-v1) のように vLLM に公式でサポートされていないが，その think タグが `<thinkg>, </think>` である推論型モデルについては， reasoning parser に `deepseek_r1_markup` を指定することで，適当に推論過程の除去が行われます．


#### 2. OpenAI APIや互換性のあるAPIを使用する場合の注意点．
本フレームワークでは [litellm](https://github.com/BerriAI/litellm) をバックエンドとして指定することにより，OpenAIのように推論APIだけが提供されているモデルも評価することができます．
また，NVIDIA NIM や DeepInfra のようなOpenAI互換の推論APIも対応しています．
ただしAPIプロバイダ固有の仕様（並列リクエスト数など）によりエラーが起きることがあるので注意してください．


#### 3. より詳細な生成条件を与える方法．（vLLM serve → lighteval 用）
vLLM serve → lighteval の方法で評価を行う場合，上記の例では以下のように評価を実行しています．
```sh
lighteval endpoint litellm \
    "model=$MODEL_NAME,api_key=$API_KEY,base_url=$BASE_URL" \
    "${TASK_ID}|0|0" \
    --use-chat-template \
    --output-dir ./lighteval/outputs
```

ここで `"model=$MODEL_NAME,api_key=$API_KEY,base_url=$BASE_URL"` の部分を config ファイル（.yaml）へのパスに置き換えることで，以下のようにより詳細な生成条件を見やすい形で指定することができます．

```yaml
# config.yaml
model:
    base_params:
        model_name: hosted_vllm/tokyotech-llm/Llama-3.1-Swallow-8B-Instruct-v0.5
        base_url: http://localhost:8000/v1

    generation:
        temperature: 0.6
        max_new_tokens: 8092
        top_p: 0.9
        max_n: 1
```

```sh
lighteval endpoint litellm \
    "config.yaml" \ # ここを config ファイルへのパスに変更する
    "${TASK_ID}|0|0" \
    --use-chat-template \
    --output-dir ./lighteval/outputs
```
