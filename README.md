# swallow-evaluation-instruct-private

lighteval: [v0.8.0](https://github.com/huggingface/lighteval/releases/tag/v0.8.0)

## セットアップ
大井のメモ: 仮装環境はvenvではなく、モジュール間の依存関係の管理が可能なuvなどで管理したい

暫定的にPipenvでセットアップできるようにしてますが（後述），uvのほうがよいでしょう．  

## 実行方法
[標準的な lighteval の実行方法](https://huggingface.co/docs/lighteval/quicktour)で動かすことができます．  


```
MODEL="tokyotech-llm/Llama-3.1-Swallow-8B-Instruct-v0.3"
SYSTEM_PROMPT="あなたは誠実で優秀な日本人のアシスタントです。"
MAX_MODEL_LENGTH=8192
MAX_NEW_TOKENS=2048
TEMPERATURE=0.0

MODEL_ARGS="pretrained=$MODEL,dtype=bfloat16,tensor_parallel_size=$NUM_GPUS,max_model_length=$MAX_MODEL_LENGTH,gpu_memory_utilization=0.8,generation_parameters={max_new_tokens:$MAX_NEW_TOKENS,temperature:$TEMPERATURE}"

lighteval vllm $MODEL_ARGS "swallow|{ベンチマークのID}|0|0" \
--system-prompt "${SYSTEM_PROMPT}" \
--output-dir $OUTPUT_DIR \
--use-chat-template
```

## ベンチマーク一覧
ligiteval の `--tasks` として指定できるように [lighteval/tasks/swallow](./lighteval/src/lighteval/tasks/swallow/) 以下に各種ベンチマークを実装しています．  
いずれも自由に応答文を記述させて回答スパンを抽出するスタイルで実装しています．  
実装中のベンチマーク一覧は以下の通り．  

### 日本語

* JEMHopQA: jemhopqa, jemhopqa_cot
* WMT20 En-Ja, Ja-En: WIP
* 日本語MT-Bench: japanese_mt_bench
* M-IFEval 日本語サブセット: WIP
* JMMLU: swallow_jmmlu
* MMLU-ProX: WIP
* JHumanEval: WIP
* MCLM MATH-100 日本語サブセット = MATH邦訳版: math_100_japanese
* BenchMAX Science Reasoning 日本語版 = GPQA邦訳版: swallow_gpqa_ja

### 英語
* HellaSwag: hellaswag
* 英語MT-Bench: WIP
* MMLU-Pro: WIP

## 評価指標一覧
[lightevalのMetricクラス](https://huggingface.co/docs/lighteval/metric-list)に準拠して，LLM-as-a-Judgeや自由記述式QAなどを定量化する評価指標を実装しています．  

### 自由記述式QA
* `JapaneseOpenQAExactMatchSamplingFunc` クラス
* 厳密な回答スパン抽出モード，ロバストな回答スパン抽出+文字列正規化モード(接頭辞 `quasi`) に対応  
* 3種類の評価指標に対応
    * 完全一致 (exact_match, quasi_exact_match)
    * 文字F1 (f1_score, f1_score_quasi)
    * llm-jp-eval方式の文字F1 (llmjpeval_f1_score, llmjpeval_f1_score_quasi)

### LLM-as-a-Judge
* `JudgeLLMMTBenchSwallow` クラス
* `<think>...</think>` タグで囲まれた推論過程の除外に対応  
* 応答文を複数回サンプリングするN回平均値の算出に対応
* カテゴリ別スコアの算出に対応

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
