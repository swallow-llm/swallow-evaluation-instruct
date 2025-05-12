# swallow-evaluation-instruct-private

lighteval: [v0.8.0](https://github.com/huggingface/lighteval/releases/tag/v0.8.0)

大井のメモ: 仮装環境はvenvではなく、モジュール間の依存関係の管理が可能なuvなどで管理したい

## ベンチマーク一覧
ligiteval の `--tasks` として使えるように [lighteval/tasks/swallow](./lighteval/src/lighteval/tasks/swallow/) 以下に各種ベンチマークを実装しています．  
いずれも自由に応答文を記述させて回答スパンを抽出するスタイルで実装しています．  
実装中のベンチマーク一覧は以下の通り．  

### 日本語

* JEMHopQA
* WMT20 En-Ja, Ja-En
* 日本語MT-Bench
* M-IFEval 日本語サブセット
* JMMLU
* MMLU-ProX
* JHumanEval
* MCLM MATH-100 日本語サブセット = MATH邦訳版
* BenchMAX Science Reasoning 日本語版 = GPQA邦訳版

### 英語
* HellaSwag
* 英語MT-Bench
* MMLU-Pro

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
