<!--
完成したらREADMEにマージします．
-->

# 大規模言語モデル評価フレームワーク swallow-evaluation-instruct Ver. 202508

このリポジトリでは，推論型モデルのような事後学習ずみモデルの評価を想定して[Swallowプロジェクト](https://swallow-llm.github.io/)にて開発した包括的評価フレームワーク swallow-evaluation-instruct（以下，"本フレームワーク"）を配布しています．  

swallow-evaluation-instruct は，HuggingFace社が開発した評価フレームワーク [lighteval](https://github.com/huggingface/lighteval) (v0.8.0) (© 2024 Hugging Face) をフォークして，日本語・英語ベンチマークの追加および利便性の改善をおこなったものです．
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

新しいモデルに対応する必要がある場合は，vLLMやLiteLLM, transformersなど推論に関するパッケージを適宜更新してください．  
ただしパッケージの更新成否および更新した場合の動作は保証しておりません．  

```
uv lock --upgrade-package vllm litellm transformers
```


### 4. パスの追加
最後に `~/.bashrc` に必要なパスを追加して uv に関する初期設定は終了です．

```sh
echo 'export PATH="/.common_envs/bin:$PATH"' >> "$HOME/.bashrc"
echo 'export PATH="$HOME/.local/bin:$PATH"' >> "$HOME/.bashrc"
```

## 実行方法
[lighteval](https://github.com/huggingface/lighteval)はvLLMをはじめとする複数のバックエンドに対応していますが，本フレームワークではOpenAI API互換クライアントであるLiteLLMを使用してChat Completion APIを呼び出す方式を主にサポートしています．  
オープンLLMを各自の計算環境で評価する場合は，vLLMでOpenAI互換APIをホスティングしてからLiteLLMバックエンドを使用してAPIを呼び出す方式を推奨します．

[lighteval公式ドキュメント](https://huggingface.co/docs/lighteval/v0.8.0/en/index)も参照してください．  

### 1. LiteLLMバックエンドで実行

`lighteval endpoint litellm {MODEL_ARGS} {TASK_ID} [OPTIONS]` で，OpenAI互換の Chat Completion API を提供するモデルを評価することができます．OpenAI o3 で GPQA (Diamond) ベンチマークを評価する例を以下に示します．  

```sh
MODEL_NAME="openai/o3-2025-04-16" 
BASE_URL="https://api.openai.com/v1/" # OpenAI API の URL
API_KEY="{OpenAIのAPI Key}"
TASK_ID="swallow|gpqa:diamond"

uv run --isolated --locked --extra lighteval \
    lighteval endpoint litellm \
        "model=$MODEL_NAME,base_url=$BASE_URL,api_key=$API_KEY" \
        "${TASK_ID}|0|0" \
        --use-chat-template \
        --output-dir ./lighteval/outputs
```

MODEL_ARGS には `model` や `base_url` を指定します．modelパラメータは `{プロバイダ名}/{MODEL ID}` という表記にします（例："openai/gpt-4o-2024-08-06"）．
MODEL_ARGSのかわりにYAML設定ファイルパスを指定することもできます．詳細は後述します．  

TASK_ID はベンチマークの識別子です．swallow-evaluation-instruct ではlighteval公式実装に加えて，Swallowチームが実装したベンチマークを指定できます．  
詳細は [Swallowチームが実装したベンチマーク一覧](./BENCHMARKS.md) を参照してください．

OpenAI互換APIを提供するDeepInfraやGoogle AI Studioなどのプロバイダ（[LiteLLM Supported Providers](https://docs.litellm.ai/docs/providers)）についても同様のコマンドで評価できます．  
ただしプロバイダやモデルによってはエラーが起きる場合があります．[Tips](./TIPS.md)

### 2. vLLMでホスティング → LiteLLMバックエンドで実行

[vLLM serveコマンド](https://docs.vllm.ai/en/v0.9.2/serving/openai_compatible_server.html)でOpenAI互換APIを立ててからLiteLLM経由でAPIを呼び出すことにより，[推論型モデルサポート](https://docs.vllm.ai/en/stable/features/reasoning_outputs.html)などのvLLMの豊富な機能を活用しながら評価を実行できます．  
Qwen3 で HumanEval ベンチマークを評価する例を以下に示します．  

```sh
MODEL_NAME="hosted_vllm/Qwen/Qwen3-4B" # vLLMのプロバイダ名は "hosted_vllm" です
TASK_ID="swallow|humaneval"

uv run --isolated --locked --extra vllm \
    vllm serve $MODEL_NAME \
        --host localhost \
        --port 8000 \
        --reasoning-parser qwen3 \
        --max-model-len 32768 &

BASE_URL="http://localhost:8000/v1"
# HuggingFace のモデルをローカルでサーブする場合には "http://localhost:(ポート番号)/v1" を指定．

uv run --isolated --locked --extra lighteval \
    lighteval endpoint litellm \
        "model=$MODEL_NAME,base_url=$BASE_URL,generation_parameters={temperature:0.2,top_p:0.95}" \
        "${TASK_ID}|0|0" \
        --use-chat-template \
        --output-dir ./lighteval/outputs
```

`vllm serve` の引数 `--reasoning-parser` を指定することで，深い推論過程（reasoning_content）および最終出力（content）に分離したモデルの出力を受け取ることができます．  
本フレームワークは最終出力から回答を抽出して正誤判定する仕様にしています（[評価方針](./EVALUATION_PRINCIPLE.md)）ので **推論型モデルの場合は必ず `--reasoning-parser` を指定してください．**

MODEL_ARGS の generation_parameters にはtemperatureのような文生成条件を指定できます．詳細は後述します．  

非推論型モデルの場合は `--reasoning-parser` が不要なのでシンプルな実行時引数になります．  
[tokyotech-llm/Llama-3.1-Swallow-8B-Instruct-v0.5](https://huggingface.co/tokyotech-llm/Llama-3.1-Swallow-8B-Instruct-v0.5)で 日本語MT-Benchを評価する例を以下に示します．

```sh
MODEL_NAME="tokyotech-llm/Llama-3.1-Swallow-8B-Instruct-v0.5"
TASK_ID="swallow|japanese_mt_bench"

export OPENAI_API_KEY="{LLM-as-a-Judgeに使うOpenAI API Key}" 

uv run --isolated --locked --extra vllm \
    vllm serve $MODEL_NAME \
        --host localhost \
        --port 8000 &

BASE_URL="http://localhost:8000/v1"



uv run --isolated --locked --extra lighteval \
    lighteval endpoint litellm \
        "model=hosted_vllm/$MODEL_NAME,base_url=$BASE_URL" \
        "${TASK_ID}|0|0" \
        --use-chat-template \
        --system-prompt "あなたは誠実で優秀な日本人のアシスタントです。" \
        --output-dir ./lighteval/outputs
```

`--system-prompt` には，いわゆるシステムメッセージを指定できます．  
システムメッセージで推論の有無や深さを制御するモデルや，推奨システムメッセージがある場合に使用します．

### 3. [非推奨] lightevalからvLLMを直接起動する
[lighteval公式ドキュメント](https://huggingface.co/docs/lighteval/quicktour)で説明されているとおり `lighteval vllm MODEL_ARGS` によってvLLMを直接起動して動かすことも可能です．  
ただしこの方式は vLLM V0エンジンのみをサポート（Ref. [vLLM V1](https://docs.vllm.ai/en/stable/usage/v1_guide.html)）していること，およびvLLM実行時引数のサポートが不完全であることから，先に紹介している[vLLMでホスティングしてから評価する方式](#2-vllmでホスティング--litellmバックエンドで実行)を推奨します．

[tokyotech-llm/Llama-3.1-Swallow-8B-Instruct-v0.5](https://huggingface.co/tokyotech-llm/Llama-3.1-Swallow-8B-Instruct-v0.5)で GPQA (Diamond) ベンチマークを評価する例を以下に示します．

```sh
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_USE_V1=0 # vLLM V0エンジンを指定

MODEL_NAME="tokyotech-llm/Llama-3.1-Swallow-8B-Instruct-v0.5"
TASK_ID="swallow|gpqa:diamond"

MODEL_ARGS="pretrained=$MODEL_NAME,dtype=bfloat16,generation_parameters={temperature:0.0}"
# MODEL_ARGSには dtype，tensor_parallel_size，max_model_length，gpu_memory_utlization，そして各種 generation_parameters なども指定できる．

uv run --isolated --locked --extra lighteval \
    lighteval vllm \
        $MODEL_ARGS \
        "${TASK_ID}|0|0" \
        --use-chat-template \
        --output-dir ./lighteval/outputs \
        --system-prompt "あなたは誠実で優秀な日本人のアシスタントです。"
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

#### 4. 主要な環境変数
* `OPENAI_API_KEY`：MT-Bench(日英) で LLM-as-a-judge として OpenAI のモデル（gpt-4o-2024-08-06）を使うために使用します．
* `LITELLM_CONCURRENT_CALLS`：LiteLLM 内で推論APIを呼ぶときの最大並列数．大きくすると処理速度は向上するかもしれませんが，推論APIの挙動が不安定になることもああります．


#### 5. 主要な lighteval 実行時引数
lighteval を用いた評価は以下のフォーマットで実行しています：
```sh
lighteval endpoint litellm {MODEL_ARGS} {TASK_ID} [OPTIONS]
```

本節では上記のうち `[OPTIONS]` と `{MODEL_ARGS}` について，主要なもの・Swallow独自(*)なものを説明します．
なお，`{MODEL_ARGS}` は [config の書き方](https://huggingface.co/docs/lighteval/v0.8.0/en/use-litellm-as-backend)に倣って base_params と generation に分けて説明をします．

> (*)：該当の引数には「（独自）」と記しております．

##### I. `[OPTIONS]`

* --system-prompt：モデルに与える system prompt．
* --save-details：評価の詳細（プロンプト・応答・メトリクスなど）を .parquet で出力させるオプション．
* --max-samples：評価対象のサンプル数．動作確認のためにサンプル数を絞って実行することなどができます．

##### II. `MODEL_ARGS` - base_params
* `model`：評価に用いるモデル名．プロバイダー名が先頭に付く．（例：`hosted_vllm`）
* `base_url`：プロバイダーに対応するURL．（例：vLLM をセルフホストする場合：`http://localhost:8000/v1`）
* `api_key`：プロバイダーに対応するAPIキー．（例：OpenAI の場合：`sk-...`）
* `reasoniing_parser`（独自）：lighteval内（≠ vLLM内）で reasoning parser を使う場合用の引数．Swallow独自のパーサー（`deepseek_r1_markup`）はこちらからしか指定できないです．

##### III. `MODEL_ARGS` - generation
* temperature：モデルに与える tempearture．
* top_p：モデルに与える top_p．
* max_n（独自）：一度の推論APIの呼び出しにおいて生成させる応答数の最大数．
* max_new_tokens：モデルに与える max_new_tokens．