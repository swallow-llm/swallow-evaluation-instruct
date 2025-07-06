# swallow-evaluation-instruct-private

lighteval: [v0.8.0](https://github.com/huggingface/lighteval/releases/tag/v0.8.0)

## 評価マニュアル

Swallowプロジェクト，特にTSUBAMEでの評価作業に特化したマニュアルはリンク先を参照してください．Ref. [TSUBAME4を用いた評価方法](./README_t4.md)


## 実行方法
vLLMで推論APIを立ててからlitellm経由でAPIを呼び出すことにより，[推論型モデルサポート](https://docs.vllm.ai/en/stable/features/reasoning_outputs.html)のようなvLLMの豊富な機能を活用しながら評価を実行できます．  
実行例は以下の通り．  

```
MODEL="Qwen/Qwen3-0.6B"
API_KEY="DUMMY"

vllm serve --model $MODEL \
--host localhost \
--port 8000 \
--reasoning-parser qwen3

BASE_URL="http://localhost:8000/v1"

lighteval endpoint litellm \
"model=$MODEL,api_key=$API_KEY,base_url=$BASE_URL" \
"lighteval|gpqa:diamond|0|0" \
--use-chat-template \
--output-dir $OUTPUT_DIR
```

**推論型モデルの場合は vLLM起動時に に reasoning_parser を指定してください．**  
これにより出力から推論過程が取り除かれます．Ref. [vLLM Doc: Reasoning Outputs](https://docs.vllm.ai/en/stable/features/reasoning_outputs.html)

### OpenAIモデル等の評価
litellmをバックエンドとして指定することにより，OpenAIのように推論APIだけが提供されているモデルも評価できます．  
NVIDIA NIM や DeepInfra のようなOpenAI互換の推論APIも対応しています．  
ただしAPIプロバイダ固有の仕様（並列リクエスト数など）によりエラーが起きることがあるのでデバッグに注意してください．  
実行例は以下の通り．  

```
# GPT-4o の GPQA を OpenAI API で評価する例
API_KEY="APIキー"
MODEL_NAME="gpt-4o-2024-08-06"
BASE_URL="https://api.openai.com/v1" # OpenAI API の URL
OUTPUT_DIR=data/evals/

lighteval endpoint litellm \
    "model=$MODEL_NAME,api_key=$API_KEY,base_url=$BASE_URL" \
    "lighteval|gpqa:diamond|0|0" \
    --use-chat-template \
    --output-dir $OUTPUT_DIR
```

### vLLMを直接起動して評価
[標準的な lighteval の実行方法](https://huggingface.co/docs/lighteval/quicktour)に則って，vLLMを直接起動して動かすことも可能です．  
ただし**vLLM実行時引数のサポートが不完全なので，vLLMで推論APIを立ててからlitellmでAPIを呼び出す動かし方を推奨します．**  

```
MODEL="tokyotech-llm/Llama-3.1-Swallow-8B-Instruct-v0.3"
SYSTEM_PROMPT="あなたは誠実で優秀な日本人のアシスタントです。"
MAX_MODEL_LENGTH=8192
TEMPERATURE=0.0

MODEL_ARGS="pretrained=$MODEL,dtype=bfloat16,tensor_parallel_size=$NUM_GPUS,max_model_length=$MAX_MODEL_LENGTH,gpu_memory_utilization=0.8,generation_parameters={temperature:$TEMPERATURE}"

lighteval vllm $MODEL_ARGS "swallow|{ベンチマークのID}|0|0" \
--system-prompt "${SYSTEM_PROMPT}" \
--output-dir $OUTPUT_DIR \
--use-chat-template
```

## ベンチマーク一覧
ligiteval の `--tasks` として指定できるように [lighteval/tasks/swallow](./lighteval/src/lighteval/tasks/swallow/) 以下に各種ベンチマークを実装しています．  
いずれも自由に応答文を記述させて回答スパンを抽出するスタイルで実装しています．  
ベンチマーク一覧およびそのIDは以下の通り．  

### 日本語

* JEMHopQA: `swallow|jemhopqa, swallow|jemhopqa_cot`
* 日本語MT-Bench: `swallow|japanese_mt_bench`
* M-IFEval 日本語版: `swallow|mifeval_ja`
* JMMLU: `swallow|swallow_jmmlu`
* MMLU-ProX 日本語サブセット: `swallow|mmlu_prox_japanese`
* JHumanEval: `swallow|swallow_jhumaneval`
* MCLM MATH-100 日本語サブセット = MATH邦訳版: `swallow|math_100_japanese`
* BenchMAX Science Reasoning 日本語版 = GPQA邦訳版: `swallow|swallow_gpqa_ja`
* WMT20 En-Ja, Ja-En: `swallow|wmt20:en-ja, swallow|wmt20:ja-en`

### 英語
* HellaSwag: `swallow|hellaswag`
* 英語MT-Bench: `swallow|english_mt_bench`
* MMLU-Pro: `swallow|mmlu_pro_english`
* MMLU-ProX: `swallow|mmlu_prox_english`
* MMLU: `mmlu_english`
    * 既存実装 helm|mmlu は選択肢だけ出力する短答を想定した実装になっている（Ref. [コード](https://github.com/swallow-llm/swallow-evaluation-instruct-private/blob/main/lighteval/src/lighteval/tasks/default_tasks.py#L10310)）ので，"考えてから回答する"スタイルで実装し直したもの．  

MMLU, MMLU-Pro, MMLU-ProX はタスクIDが似ていますので取り違えに注意してください．  

以下のベンチマークはligiteval公式実装を微調整したものです．  

* GPQA: `swallow|gpqa:diamond`
* MATH-500: `swallow|math_500`
* AIME 2024--2025: `swallow|aime:24, lighteval|aime:25`
* LiveCodeBench v5 & v6 追加設問: `swallow|lcb:codegeneration_v5_v6`

## ベンチマークごとの詳細な評価設定
shot数，メトリック，CoT有無などの詳細な評価設定は，以下の資料を参照してください．  
* [表1. 評価タスクの一覧 Ver. 202504](https://docs.google.com/spreadsheets/d/1lMMaZmv6FwIZC6EArFLaApvc99gkuXh7uGb8AMnhzB4/edit?gid=1254224743#gid=1254224743&range=A9)
* [Instructモデルむけ評価方法および実装方法の検討](https://nlp-titech.slack.com/docs/T7EAFSVDY/F08F9ACBPL2)（岡崎研Slack Canvas）

## 評価指標一覧
[lightevalのMetricクラス](https://huggingface.co/docs/lighteval/metric-list)に準拠して，LLM-as-a-Judgeや自由記述式QAなどを定量化する評価指標を実装しています．  

### 自由記述式QA
* `JapaneseOpenQAExactMatchSamplingFunc` クラス
* 厳密な回答スパン抽出モード，ロバストな回答スパン抽出+文字列正規化モード(接頭辞 `quasi`) に対応  
* 3種類の評価指標に対応
    * 完全一致 : exact_match, quasi_exact_match
    * 文字F1: f1_score, f1_score_quasi
    * llm-jp-eval方式の文字F1: llmjpeval_f1_score, llmjpeval_f1_score_quasi

### LLM-as-a-Judge
* `JudgeLLMMTBenchSwallow` クラス
* `<think>...</think>` タグで囲まれた推論過程の除外に対応  
* 応答文を複数回サンプリングするN回平均値の算出に対応
* カテゴリ別スコアの算出に対応

### 機械翻訳
* `{Japanese}TranslationPreparator` クラスで前処理したのちに sacreBLEU で計算
* `日本語: ` や `English: ` のように所定のプレースホルダに後続する文字列を抽出することで「考えてから翻訳する」タイプのモデルに対応
* トークナイザが異なる2種類の日本語BLEUに対応
    * Janomeトークナイザ (≒MeCab+IPADIC) による標準的なBLEU
    * Nagisaトークナイザ によるJP LM Eval. Harness方式互換のBLEU
* chrFやTERもサポートしているが不採用  

### 


## [暫定] Pipenvによる仮想環境の管理

### 必要なもの
* pyenv または Python 3.11 ランタイム
* pipenv
* CUDA 12.1

### コマンド
`./lighteval/` にて

* 仮想環境の構築： `pipenv install --dev --skip-lock`
    * 本来は `--skip-lock` を指定せずにモジュール間の依存関係を完全に管理すべきなんですが，vLLM, pytorch, cuda の3つが絡むと容易に依存関係のresolveに失敗するみたいです．
* 仮想環境を有効化： `pipenv shell`
* extractive_match_metric の単体テストを実行： `pipenv run test_extractive_match`
* 仮想環境を抜ける： `exit`
* 仮想環境を削除する： `pipenv --rm`

もしflash-attention を入れる場合は，仮想環境を有効化してから `pip install flash-attn --no-build-isolation` を実行します．

さらに詳しい使用例は [Pipenv](https://pipenv.pypa.io/en/latest/) を参照するか，ChatGPTに聞いてみてください．  

## 評価スクリプトが使用するLLM評価フレームワークおよびそれらのライセンス・変更点
公開までに揃えること．

### M-IFEval 日本語サブセット
* バージョン: [M-IFEval](https://github.com/lightblue-tech/M-IFEval) 0bc3143
* ライセンス: Copyright 2025 Lightblue Inc, Apache License Version 2.0 ([LICENSE](https://github.com/lightblue-tech/M-IFEval/blob/main/LICENSE.txt))

#### 主な変更点
* <think>...</think>のような推論型モデルの推論過程を削除する処理を追加しました．  
* 言語制約の採点再現性を確保するために `langdetect` パッケージの乱数シードを固定しました．  
