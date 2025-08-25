<!--
完成したらREADMEにマージします．
-->

# 大規模言語モデル評価フレームワーク swallow-evaluation-instruct Ver. 202508

このリポジトリでは，推論型モデルのような事後学習済みモデルの評価を想定して[Swallowプロジェクト](https://swallow-llm.github.io/)にて開発した包括的評価フレームワーク swallow-evaluation-instruct（以下，"本フレームワーク"）を配布しています．  

swallow-evaluation-instruct は，HuggingFace社が開発した評価フレームワーク [lighteval](https://github.com/huggingface/lighteval) (v0.8.0) (© 2024 Hugging Face) をフォークして，[日本語・英語ベンチマークの追加](./BENCHMARKS.md)および利便性の改善をおこなったものです．この場をお借りしてlighteval開発者およびベンチマーク開発者の皆様にお礼申し上げます．  

（事後学習を施していない）事前学習済みモデルの評価をお考えの方は [swallow-evaluation](https://github.com/swallow-llm/swallow-evaluation) をご検討ください．

[Swallow LLM Leaderboard v2](https://swallow-llm.github.io/swallow-leaderboard-v2.ja.html)には，本フレームワークを用いてさまざまな事後学習済みモデルを評価した結果を掲載しています．ぜひご覧ください．

- [大規模言語モデル評価フレームワーク swallow-evaluation-instruct Ver. 202508](#大規模言語モデル評価フレームワーク-swallow-evaluation-instruct-ver-202508)
  - [以前のバージョンをお探しの方へ](#以前のバージョンをお探しの方へ)
  - [環境構築](#環境構築)
    - [1. uv のインストール](#1-uv-のインストール)
    - [2. uv 内への python のインストール](#2-uv-内への-python-のインストール)
    - [3. 共通パッケージのインストール](#3-共通パッケージのインストール)
    - [4. パスの追加](#4-パスの追加)
  - [実行方法](#実行方法)
    - [1. OpenAI互換の推論APIを提供するモデルの評価](#1-openai互換の推論apiを提供するモデルの評価)
    - [2. オープンモデルを自身の計算環境で評価](#2-オープンモデルを自身の計算環境で評価)
      - [\[推奨\] 2.1. vLLMでホスティング -\> LiteLLMバックエンドで実行](#推奨-21-vllmでホスティング---litellmバックエンドで実行)
      - [\[非推奨\] 2.2. lightevalからvLLMを直接起動する](#非推奨-22-lightevalからvllmを直接起動する)
    - [3. MODEL\_ARGS のかわりに設定ファイルを使う方法](#3-model_args-のかわりに設定ファイルを使う方法)
    - [4. 評価結果の出力先](#4-評価結果の出力先)
  - [詳細な設定](#詳細な設定)
    - [環境変数](#環境変数)
    - [lighteval 実行時引数](#lighteval-実行時引数)
      - [`[OPTIONS]`](#options)
      - [`MODEL_ARGS` - base\_params](#model_args---base_params)
      - [`MODEL_ARGS` - generation](#model_args---generation)
    - [vLLM serve 実行時引数](#vllm-serve-実行時引数)
  - [ライセンス](#ライセンス)
  - [謝辞](#謝辞)
  - [開発者](#開発者)
  - [関連資料](#関連資料)
  - [引用について](#引用について)

## 以前のバージョンをお探しの方へ
以前のバージョンをご利用になりたい方は[Releases](https://github.com/swallow-llm/swallow-evaluation-instruct/releases)を参照してください．

## 環境構築
本フレームワークでは環境管理に [uv](https://docs.astral.sh/uv/) を使用することを想定しています．
環境構築は以下の流れで行ってください． \
なお，以下の一連の操作を一つのシェルスクリプトにまとめたものが [./scripts/setup_env.sh](./scripts/setup_env.sh) にありますので，適宜ご活用ください．


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
これにより，ライセンスの承諾が必要なモデルなどを評価できるようになります．
```sh
hf auth login --token (ここに huggingface token を書く)
deactivate
```

新しいモデルに対応する必要がある場合は，[vLLM](https://github.com/vllm-project/vllm)や[LiteLLM](https://github.com/BerriAI/litellm), [transformers](https://github.com/huggingface/transformers)など推論に関するパッケージを適宜更新してください．ただし，パッケージの更新成否および更新した場合の動作は保証しておりません．  

```
uv lock --upgrade-package vllm litellm transformers
```


### 4. パスの追加
最後に `~/.bashrc` に必要なパスを追加して uv に関する初期設定は終了です．

```sh
echo "export PATH=\"$(pwd)/.common_envs/bin:\$PATH\"" >> "$HOME/.bashrc"
echo 'export PATH="$HOME/.local/bin:$PATH"' >> "$HOME/.bashrc"
```

## 実行方法
本フレームワークでは，OpenAI API互換のChat Completion API（以下，"推論API"）を提供するモデルの評価とvLLMでのホスティングが可能なオープンモデルの評価をサポートしています．

<!-- 
lightevalはvLLMをはじめとする複数のバックエンドに対応していますが，本フレームワークではOpenAI API互換クライアントのLiteLLMを使用してChat Completion API（以下，"推論API"）を呼び出す方式を主にサポートしています．オープンLLMを各自の計算環境で評価する場合は，vLLMで推論APIをホスティングしてからLiteLLMバックエンドを使用し，APIを呼び出す方式を推奨します．
-->

以下に，各方式の実行方法を説明します．コマンド例と同等のシェルスクリプトを [./scripts/examples](./scripts/examples) に格納しています．

### 1. OpenAI互換の推論APIを提供するモデルの評価

`lighteval endpoint litellm {MODEL_ARGS} {TASK_ID} [OPTIONS]` で，OpenAI互換の推論APIを提供するモデルを評価できます．OpenAI o3 で GPQA (Diamond) ベンチマークを評価する例を以下に示します．  

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
        --output-dir ./lighteval/outputs \
        --save-details
```

MODEL_ARGS には `model` や `base_url` を指定します．modelパラメータは `{プロバイダ名}/{MODEL ID}` という表記にします（例："openai/gpt-4o-2024-08-06"）．
MODEL_ARGSのかわりにYAML設定ファイルのパスを指定できます．詳細は後述します．  

TASK_ID はベンチマークの識別子です．swallow-evaluation-instruct ではlighteval公式実装に加えて，Swallowチームが実装したベンチマークを指定できます．詳細は [Swallowチームが実装したベンチマーク一覧](./BENCHMARKS.md) を参照してください．

OpenAI互換の推論APIを提供するDeepInfraやGoogle AI Studioなどのプロバイダ（[LiteLLM Supported Providers](https://docs.litellm.ai/docs/providers)）についても同様のコマンドで評価できます．ただし，プロバイダやモデルによってはエラーが起きる場合があります（[Tips](./TIPS.md)参照）．

### 2. オープンモデルを自身の計算環境で評価

オープンモデルを各自の計算環境で評価する場合は，vLLMで推論APIをホスティングしてからLiteLLMバックエンドを使用し，APIを呼び出す方式を推奨します．

#### [推奨] 2.1. vLLMでホスティング -> LiteLLMバックエンドで実行

[vLLM serveコマンド](https://docs.vllm.ai/en/v0.9.2/serving/openai_compatible_server.html)でOpenAI互換APIを起動し，そのAPIをLiteLLM経由で呼び出すことにより，[推論型モデルサポート](https://docs.vllm.ai/en/stable/features/reasoning_outputs.html)などのvLLMの豊富な機能を活用しながら評価を実行できます．

**推論型モデルの評価実行例**

Qwen3 で HumanEval ベンチマークを評価する例を以下に示します．
例では1つのシェルスクリプト内でvLLMのホスティング（`vLLM serve`）とLiteLLMバックエンドによる評価の実行を行っていますが，それぞれを分離して明示的に別のプロセスで行うことも可能です．

```sh
MODEL_ID="Qwen/Qwen3-4B"
# vLLMでセルフホストする場合のプロバイダ名は "hosted_vllm" とします
MODEL_NAME="hosted_vllm/${MODEL_ID}"
TASK_ID="swallow|humaneval"
# vLLMのセルフホストの状況がVLLM_LOG_FILEに出力されます
VLLM_LOG_FILE="./vllm.log"

setsid uv run --isolated --locked --extra vllm \
    vllm serve "$MODEL_ID" \
        --host localhost \
        --port 8000 \
        --reasoning-parser qwen3 \
        --max-model-len 32768 >"$VLLM_LOG_FILE" 2>&1 &

VLLM_PID=$!
VLLM_PGID=$(ps -o pgid= "$VLLM_PID" | tr -d ' ')
trap 'kill -TERM -'"$VLLM_PGID"' 2>/dev/null; sleep 2; kill -KILL -'"$VLLM_PGID"' 2>/dev/null || true' EXIT INT TERM

BASE_URL="http://localhost:8000/v1"
# HuggingFace のモデルをローカルでサーブする場合には "http://localhost:(ポート番号)/v1" を指定．

# vLLMが起動するまで待機
for i in {1..60}; do
    if nc -z localhost 8000; then
        echo "vllm serve is up."
        break
    fi
    echo "Waiting for vllm serve to start... ($i/60)"
    sleep 30
done

uv run --isolated --locked --extra lighteval \
    lighteval endpoint litellm \
        "model=$MODEL_NAME,base_url=$BASE_URL,generation_parameters={temperature:0.2,top_p:0.95}" \
        "${TASK_ID}|0|0" \
        --use-chat-template \
        --output-dir ./lighteval/outputs \
        --save-details
```

`vllm serve` の引数 `--reasoning-parser` を指定することで，推論過程（reasoning_content）および最終出力（content）が分離されたモデルの出力を受け取ることができます．本フレームワークはモデルの最終出力から回答を抽出して正誤判定する仕様となっていますので **推論型モデルの場合は必ず `--reasoning-parser` を指定してください．**

MODEL_ARGS の generation_parameters にはtemperatureのような文生成条件を指定できます．詳細は後述します．**本フレームワークではデフォルトの文生成条件を定義していませんので，モデルやベンチマークごとに適切な条件を指定してください**（参考：[Swallowチームが実装したベンチマーク一覧](./BENCHMARKS.md)）．

**非推論型モデルの評価実行例**

非推論型モデルの場合についても説明します．この場合は `--reasoning-parser` が不要なのでシンプルな実行時引数になります．  
[tokyotech-llm/Llama-3.1-Swallow-8B-Instruct-v0.5](https://huggingface.co/tokyotech-llm/Llama-3.1-Swallow-8B-Instruct-v0.5)で 日本語MT-Benchを評価する例を以下に示します．

```sh
MODEL_ID="tokyotech-llm/Llama-3.1-Swallow-8B-Instruct-v0.5"
MODEL_NAME="hosted_vllm/${MODEL_ID}"
TASK_ID="swallow|japanese_mt_bench"
# vLLMのセルフホストの状況がVLLM_LOG_FILEに出力されます
VLLM_LOG_FILE="./vllm.log"

export OPENAI_API_KEY="{LLM-as-a-Judgeに使うOpenAI API Key}" 

setsid uv run --isolated --locked --extra vllm \
    vllm serve $MODEL_ID \
        --host localhost \
        --port 8000 >"$VLLM_LOG_FILE" 2>&1 &

VLLM_PID=$!
VLLM_PGID=$(ps -o pgid= "$VLLM_PID" | tr -d ' ')
trap 'kill -TERM -'"$VLLM_PGID"' 2>/dev/null; sleep 2; kill -KILL -'"$VLLM_PGID"' 2>/dev/null || true' EXIT INT TERM

BASE_URL="http://localhost:8000/v1"

# vLLMが起動するまで待機
for i in {1..60}; do
    if nc -z localhost 8000; then
        echo "vllm serve is up."
        break
    fi
    echo "Waiting for vllm serve to start... ($i/60)"
    sleep 30
done

uv run --isolated --locked --extra lighteval \
    lighteval endpoint litellm \
        "model=$MODEL_NAME,base_url=$BASE_URL" \
        "${TASK_ID}|0|0" \
        --use-chat-template \
        --system-prompt "あなたは誠実で優秀な日本人のアシスタントです。" \
        --output-dir ./lighteval/outputs \
        --save-details
```

`--system-prompt` には，いわゆるシステムメッセージを指定できます．システムメッセージで推論の有無や深さを制御するモデルや，推奨システムメッセージがデフォルトと異なる場合に使用します．

#### [非推奨] 2.2. lightevalからvLLMを直接起動する
[lighteval公式ドキュメント](https://huggingface.co/docs/lighteval/quicktour)で説明されているとおり `lighteval vllm MODEL_ARGS` によってvLLMを直接起動して実行することも可能です．ただし，この方式は vLLM V0エンジンのみをサポート（[vLLM V1](https://docs.vllm.ai/en/stable/usage/v1_guide.html)を参照）していること，およびvLLM実行時引数のサポートが不完全であることから，先に紹介している[vLLMでホスティングしてから評価する方式](#推奨-21-vllmでホスティング---litellmバックエンドで実行)を推奨します．

[tokyotech-llm/Llama-3.1-Swallow-8B-Instruct-v0.5](https://huggingface.co/tokyotech-llm/Llama-3.1-Swallow-8B-Instruct-v0.5)で GPQA (Diamond) ベンチマークを評価する例を以下に示します．

```sh
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_USE_V1=0 # vLLM V0エンジンを指定

MODEL_NAME="tokyotech-llm/Llama-3.1-Swallow-8B-Instruct-v0.5"
TASK_ID="swallow|gpqa:diamond"

MODEL_ARGS="pretrained=$MODEL_NAME,dtype=bfloat16,generation_parameters={temperature:0.0}"
# MODEL_ARGSには tensor_parallel_size，max_model_length，gpu_memory_utilization，generation_parameters なども指定できる．

uv run --isolated --locked --extra lighteval \
    lighteval vllm \
        $MODEL_ARGS \
        "${TASK_ID}|0|0" \
        --use-chat-template \
        --system-prompt "あなたは誠実で優秀な日本人のアシスタントです。" \
        --output-dir ./lighteval/outputs \
        --save-details
```

### 3. MODEL_ARGS のかわりに設定ファイルを使う方法

`"model=$MODEL_NAME,api_key=$API_KEY,base_url=$BASE_URL"` のような lighteval の実行時引数 MODEL_ARGS は，以下に例示するようなYAML形式の設定ファイル（.yaml）に置き換えることができます．
モデルIDやAPIのエンドポイントなどは `base_params` 以下に書き，temperatureやtop_pなどの文生成条件は `generation` 以下に書きます（参考：[設定ファイルの書き方](https://huggingface.co/docs/lighteval/v0.8.0/en/use-litellm-as-backend)）．

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
    "config.yaml" \ # MODEL_ARGSのかわりに設定ファイルのパスを書く
    "${TASK_ID}|0|0" \
    --use-chat-template \
    --system-prompt "あなたは誠実で優秀な日本人のアシスタントです。" \
    --output-dir ./lighteval/outputs \
    --save-details
```

### 4. 評価結果の出力先

評価した結果のスコアは標準出力に表示されるほかに，評価設定および結果が `--output-dir` で指定したディレクトリ配下にモデル名とともにJSON形式で保存されます（例：`./lighteval/outputs/results/openai/o3-2025-04-16/results_{タイムスタンプ}.json`）．
またlighteval実行時引数 `--save-details` を付けた場合は，各設問に対するプロンプトやモデルの応答文などの詳細がParquet形式で保存されます（[Tips](./TIPS.md)参照）．

## 詳細な設定

本フレームワークの動作に影響する環境変数，評価に影響するlightevalの実行時引数や文生成条件，およびvLLMでモデルをホスティングする際の主な設定項目を説明します．

### 環境変数

主な環境変数は以下のとおりです．  

* `OPENAI_API_KEY`：MT-Bench(日英) で LLM-as-a-judge として OpenAI のモデルを呼び出すために使用します．  
* `LITELLM_CONCURRENT_CALLS`：LiteLLMが推論APIを呼ぶときの最大並列数．大きくすると処理速度は向上するかもしれませんが，推論APIの挙動が不安定になることもあります．  

### lighteval 実行時引数

lightevalの実行時引数 `lighteval endpoint litellm {MODEL_ARGS} {TASK_ID} [OPTIONS]` のうち，本節では `TASK_ID`，`[OPTIONS]` および `{MODEL_ARGS}` の主要な項目およびSwallow独自(*)の項目を説明します．なお `{MODEL_ARGS}` は [設定ファイルの書き方](https://huggingface.co/docs/lighteval/v0.8.0/en/use-litellm-as-backend)に倣って，YAML形式ファイルの base_params と generation に分けて説明をします．

ここで説明しない実行時引数については [lighteval公式ドキュメント](https://huggingface.co/docs/lighteval/quicktour)を参照してください．

> (*)：該当の引数には「（独自）」と記しております．


#### `TASK_ID`

評価したいベンチマークのタスクIDを `{タスクID}|0|0` という形式で指定します（例：`swallow|gpqa:diamond|0|0`）．タスクID直後の数字はFew-shot数を表していますが，[Swallowチームが実装したベンチマーク](./BENCHMARKS.md)ではゼロショット設定を意味する `0|0` を指定することを推奨します．

#### `[OPTIONS]`

* `--system-prompt`：モデルに与えるシステムメッセージ．
* `--save-details`：評価の詳細（プロンプト・モデルの応答文・メトリクスなど）をParquet形式で保存するオプション．
* `--use-chat-template`：ユーザーの発話やシステムメッセージを対話形式に整形するテンプレートを適用します．**原則として必須です．**
* `--max-samples`：評価対象のサンプル数．動作確認のために数件だけ実行するような場合に便利です．
* `--output-dir`：評価結果の保存先．保存先を "DIR" とすると，評価結果は `DIR/results/モデル名/` 配下に，評価の詳細は `DIR/details/モデル名/` 配下にそれぞれタイムスタンプとともに保存されます．

#### `MODEL_ARGS` - base_params
* `model`：評価に用いるモデル名．プロバイダー名を先頭に付けてください．（例：`hosted_vllm`）
* `base_url`：プロバイダーに対応するURL．（例：vLLM をセルフホストする場合：`http://localhost:8000/v1`）
* `api_key`：プロバイダーに対応するAPIキー．（例：OpenAI の場合：`sk-...`）
* `reasoning_parser`（独自）：lighteval内（≠ vLLM内）で適用する reasoning parser の名前．vLLM非対応の推論型モデルを評価する場合，またはlightevalからvLLMを直接起動する場合のみ使用します．vLLMのreasoning parserに加えて，Swallow独自のパーサー（`deepseek_r1_markup`）を指定できます（参考：[Tips](./TIPS.md)）．

#### `MODEL_ARGS` - generation
* `temperature`：サンプリングの温度．
* `top_p`：核サンプリング（[Holtzman et al. (2020)](https://openreview.net/forum?id=rygGQyrFvH)）のパラメータ．
* `max_new_tokens`：出力トークン数の最大値．
* `reasoning_effort`（独自）：推論の深さ（例："middle"）．LiteLLMが対応しているOpenAI o系列などの推論型モデルで利用できます（参考：[Reasoning models](https://platform.openai.com/docs/guides/reasoning)）
* `max_n`（独自）：推論APIの1回の呼び出しにおいて生成させる応答数の最大値（いわゆる"n"の上限値）．OpenAIのように応答数を制限しているプロバイダは1を指定してください（参考：[Tips](./TIPS.md)）．

### vLLM serve 実行時引数
`vllm serve` コマンドの主な実行時引数は以下の通りです．

* `model`(位置引数)：評価に用いるモデル名．HuggingFace Model ID または Model Checkpoint のパスを指定します．
* `--reasoning-parser`：推論型モデルの出力を推論過程および最終出力に分離するparserの名前．**推論型モデルの場合は必ず指定してください．** 選択肢は公式ドキュメントを参照ください（[Reasoning Outputs](https://docs.vllm.ai/en/v0.9.2/features/reasoning_outputs.html)）．
* `--port`：セルフホストするためのポート番号．衝突すると serve に失敗します．
* `--hf-token`：HuggingFaceのトークン．モデルをロードするときに使用されます．
* `--tensor-parallel-size`：GPUの並列数．注意機構のヘッド数に対して約数でなければなりません．
* `--max-model-len`：モデルの最大コンテキスト長（入力と出力の和）．

その他の実行時引数については公式のドキュメント：[vLLM CLI Guide](https://github.com/vllm-project/vllm/blob/v0.9.2/docs/cli/README.md)をご参照ください．

## ライセンス

本フレームワークは MIT License で配布します．ベンチマークのライセンスは [Swallowチームが実装したベンチマーク一覧](./BENCHMARKS.md) を参照してください．

## 謝辞

本成果物 swallow-evaluation-instruct は，産総研政策予算プロジェクト「フィジカル領域の生成AI基盤モデルに関する研究開発」，文部科学省の補助事業「生成AIモデルの透明性・信頼性の確保に向けた研究開発拠点形成」，その他の支援によって構築されました．
またモデルの評価については，産総研及びAIST Solutionsが提供するABCI 3.0を「ABCI 3.0開発加速利用」を支援を受けて利用しました．
本研究は，東京科学大学のスーパーコンピュータ TSUBAME4.0 を利用して実施しました．

## 開発者

本フレームワークの開発者は [CONTRIBUTORS](./CONTRIBUTORS.md) を参照してください．

## 関連資料
本フレームワークの利用に関連する資料は以下の通りです．

* [Swallowチームが実装したベンチマーク一覧](./BENCHMARKS.md)
* [評価における課題と解決策](./TIPS.md)

## 引用について

本フレームワークを引用くださる場合は以下の書誌情報をお使いください．

```
@misc{swallow-evaluation-instruct-v202508,
  author       = {{Swallow LLM Team} and Mizuki, Sakae and Saito, Koshiro and Oi, Masanari and Ichinose, Tatsuya and Matsushita, Naoya and Miyamoto, Sora and Nguyen, Tien Dung and Moon, Sangwhan},
  title    = {大規模言語モデル評価フレームワーク swallow-evaluation-instruct},
  url = {https://github.com/swallow-llm/swallow-evaluation-instruct},
  howpublished = {\url{https://github.com/swallow-llm/swallow-evaluation-instruct}},
  year     = {2025},
  version  = {v202508}
}
```
