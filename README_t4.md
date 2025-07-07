# TSUBAME4を用いた評価方法

> 目次
> - [概要](#概要)
> - [更新履歴](#更新履歴)
> - [1. 環境構築](#1-環境構築)
>   - [1.1 評価者固有情報の登録](#11-評価者固有情報の登録)
>   - [1.2 評価者固有情報ファイルの名前変更](#12-評価者固有情報ファイルの名前変更)
>   - [1.3 環境構築スクリプトの実行](#13-環境構築スクリプトの実行)
> - [2. 評価の実行](#2-評価の実行)
>   - [2.1 モデルの設定](#21-モデルの設定)
>   - [2.2 タスクの設定](#22-タスクの設定)
>   - [2.3 評価の実行](#23-評価の実行)
>   - [2.4 評価結果の確認](#24-評価結果の確認)
>   - [2.5 評価結果の確認](#25-評価結果の確認)
>   - [2.6 評価ログの確認](#26-評価ログの確認)
>   - [2.7 評価詳細の確認（lightevalを用いた評価の場合のみ）](27-評価詳細の確認（lightevalを用いた評価の場合のみ）)
> - [3. Tips](#3-tips)
>   - [3.1 タスクを追加するときに](#31-タスクを追加するときに)
>   - [3.2 モデル固有の生成条件を追加するとき](#32-モデル固有の生成条件を追加するとき)
>   - [3.3 各モデル用のディレクトリについて](#33-各モデル用のディレクトリについて)
>   - [3.4 各モデル用のディレクトリについて](#34-特定のproviderでエラーが出る場合の対応)
>   - [3.5 並列応答数を強制的に1にする方法](#35-並列応答数を強制的に1にする方法)
>   - [3.6 評価がいつまでたっても終わらない場合](#36-評価がいつまでたっても終わらない場合)

## 概要
この資料は，岡崎研の評価チームが TSUBAME4 上で評価を行う際に参照することを想定した，内部向けのマニュアルである．

## 更新履歴
| 日付 | 内容 | 担当 |
| -- | -- | -- |
| 2025/06/19 | 初稿 | 齋藤 |
| 2025/07/01 | custom_settings とタスクの追加に関して追記 | 齋藤 |
| 2025/07/02 | Provider固有の問題への対処法、モデル固有の生成条件に関する注意、計算が終わらないときの対処法をTipsに追記 | 水木 |

## 1. 環境構築
### 1.1 評価者固有情報の登録
`.env_template` に記されている以下の変数について各評価者の環境に合わせて修正を行う．

| 変数名 | 説明 | 備考 | 
| -- | -- | -- |
| `GROUP_AND_NAME` | 以下の変数の設定を簡単にするための変数． | 必ずしも使う必要はない． |
| `REPO_PATH` | 評価レポジトリの絶対パス．| デバッグなどで複数のレポジトリを持っている場合は注意． |
| `HUGGINGFACE_CACHE` | HuggingFace のデータセットやモデルのインストールなどに関するキャッシュを保存しておくディレクトリ． | 必ず `/gs/bs/` 以下のディレクトリを指定すること． |
| `UV_CACHE` | UV のパッケージインストールなどに関するキャッシュを保存しておくディレクトリ．| 容量はそこまで大きくならないことが想定されるが，念の為 `/gs/bs/` 以下のディレクトリを指定するとよい．|
| `VLLM_CACHE` | VLLM の動作に関するキャッシュを保存しておくディレクトリ．| 容量がどれほど占められるかは把握していないが，念の為`/gs/bs/` 以下のディレクトリを指定するとよい．|
| `OPENAI_API_KEY` | MT-Bench の評価における LLM-as-a-judge で OpanAI のモデルを使用するときや，OpenAI のモデルを評価するよきに使用する． | Swallow project 用の API を指定すること．お金を消費するので要注意．|
| `DEEPINFRA_API_KEY` | DeepInfra のモデルを評価するよきに使用する． | Swallow project 用の API を指定すること．お金を消費するので要注意．|
| `HF_TOKEN` | HuggingFace のデータセットやモデルのインストール時に用いられるトークン．| Rate Limit の緩和や使用許可が必要なデータセット・モデルを使用する際に必要．|

### 1.2 評価者固有情報ファイルの名前変更
1.1 で評価者固有情報を記載したファイルの名前を`.env_template` から `.env` に変更しておく． \
これにより評価スクリプトから読み取られるようになり，また，`gitignore` の対象となる．
> ⚠️ 注意： \
> `.env` には　API キーが含まれているため決してパブリックに公開してはならない．

### 1.3 環境構築スクリプトの実行
1.1，1.2 で正しく評価者固有情報が登録できていることを確認した上で，以下のスクリプトを実行し，環境構築を行う．
```bash
# 以下は Saito の例
cd /gs/fs/tga-okazaki/saito/swallow-evaluation-instruct-private
bash scripts/tsubame/environment/setup_t4_uv_envs.sh
```

最終的に `"✅ Environment was successfully created!"` が表示されれば成功．


## 2. 評価の実行
### 2.1 モデルの設定
まず，評価を行うモデルについて以下の情報を`scripts/tsubame/qsub_all.sh`の `# Set Args` の欄に書き込む．

| 変数名 | 説明 | 備考 |
| -- | -- | -- |
| `NODE_KIND` | 使いたいTSUBAME4のノード．["node_q", "node_f", "cpu_16"] | 13B以下なら"node_q"，13B超なら"node_f"，OpenAIやDeepInfraのAPIを使うなら"cpu_16"を選ぶと良い．|
| `MODEL_NAME`| 評価するモデルのHuggingFaceID．| HuggingFaceのモデルカード上部にあるコピーボタンから取得できる．|
| `PROVIDER` | 評価するモデルを serve するための provider．| HuggingFaceのモデルであれば"vllm"（デフォルト），OpenAI のモデルなら"openai"を指定．Deepinfra を使う場合は"deepinfra"を指定する．|
| `PRIORITY` | 使いたいTSUBAME4における優先度．["-5", "-4", "-3"] | 数値が大きい方がジョブが流れやすくなる．しかし，それに応じて値段が2倍，4倍と高くなるので，指定する場合は要相談．|
| `MAX_MODEL_LENGTH` | 評価するモデルの生成時に渡す引数．入力と出力の合計の最大値であり，この大きさのKV CACHEが確保される．| モデルのconfigから自動で取得を行うので基本的に指定は不要．自動取得に失敗する場合のみ指定．|
| `MAX_COMPLETION_TOKENS` | 評価するモデルの生成時に渡す引数．出力の最大トークン数の制約である．| モデルの `MAX_MODEL_LENGTH` から自動計算されるので，基本的に指定は不要．必要な場合のみ指定．|

> 📝 Note: \
> 以下の変数は custom model setetings で設定するように変更した．(参照： [3.2 モデル固有の生成条件を追加するとき](#32-モデル固有の生成条件を追加するとき) ）
> `SYSTEM_MESSAGE`，`MAX_MODEL_LENGTH`，`MAX_NEW_TOKENS`

> 📝 Note： \
> openaiやdeepinfraのようなAPIを使う場合はTSUBAMEでなくてもよい．

### 2.2 タスクの設定
2.1 でモデルの設定を終えたら \
同ファイル（`scripts/tsubame/qsub_all.sh`）下部の `# Submit tasks` 以降を編集し， \
評価するタスクを指定する．具体的には，評価しないタスクについてコメントアウトをすれば良い．

### 2.3 評価の実行
2.1，2.2 でモデルとタスクの指定を正しく行ったことを確認したのち，以下のスクリプトで評価のジョブを投げることができる．
```sh
# 以下は Saito の例
cd /gs/fs/tga-okazaki/saito/swallow-evaluation-instruct-private
bash scripts/tsubame/qsub_all.sh
```

### 2.4 評価状況の確認
以下のスクリプトから評価状況を確認することができる．
```sh
# 以下は Saito の例
cd /gs/fs/tga-okazaki/saito/swallow-evaluation-instruct-private
bash scripts/tsubame/utils/save_and_check_qstat.sh
```

### 2.5 評価結果の確認
評価結果（`aggregated_results.json`）は \
各モデル用のディレクトリ（`results/{provider}/{model_publisher}/{model_name}/{custom_settings}`）以下に保存される． \
評価担当者は `aggregated_results.json` 内の `overall`の値を，指定された spreadsheet にコピーすれば良い．

ちなみに評価結果には評価メトリクスや評価タスクに加え，使用した custom_settings の詳細も記されている． \
もちろん custom_settings を使用していない場合には何も記されない．


### 2.6 評価ログの確認
評価のログ（標準出力 `.o` ファイル・標準エラー出力 `.e` ファイル）は \
各モデル用のディレクトリ（`results/{provider}/{model_publisher}/{model_name}/{custom_settings}`）以下に言語・タスクごとに保存される．


### 2.7 評価詳細の確認（lightevalを用いた評価の場合のみ）
評価結果の詳細は `lighteval/outputs/results` 以下に `.json` ファイルとして， \
モデルが生成した回答の詳細は `lighteval/outputs/outputs` 以下に `.pqt` ファイルとして保存されている． \
`.pqt`ファイルは pandas を用いて dataframe として開くことができる． \
（`scripts/utils/details_viewer.ipynb`参照）


## 3. その他
### 3.1 タスクを追加するときに
タスクを追加する際には `lighteval/src/ligtheval/tasks/swallow/` 以下にタスクの定義を書く． \
しかし，その操作はあくまで lighteval に対するの操作であり，swallow-evaluation-instruct 用には追加の操作が必要である． \
以下にそれをまとめる．

| カテゴリ | 必要性 | 操作対象 | 操作内容 |
| -- | -- | -- | -- |
| 結果集約（Aggregate）のための操作 | 必須 | `scripts/aggregate_utils/conf.py` | 追加したタスクに対応するメトリクスを定義する| 
| | 適宜 |`scripts/aggregate_utils/funcs.py` | 追加したタスクのメトリクスに必要な計算を追加することができる |
| | 適宜 |`scripts/aggregate_utils/white_lists.py` | 追加したタスクのメトリクスの計算に用いるタスクサブセットのサブセットを定義することができる |
| 評価実行のための操作 | 必須 | `scripts/tsubame/conf/tasks_runtime.csv` | 追加したタスクについて，`key`（"{言語}_{タスク名}"），`script`（定義したタスク名），`result_dir`（結果・ログの出力先），`framework`（フレームワーク），`hrt_q`（node_qでの想定所要時間），`hrt_f`（node_fでの想定所要時間），を定義する |
| | 必須 | `scripts/tsubame/qsub_all.sh` | 追加したタスクについて，`qsub_task {言語} {タスク名}` を末尾の適当な箇所に追加する．|
| | 必須 | `scripts/generation_settings/task_settings.csv` | 追加したタスク固有の生成条件を記す．デフォルトは`temperature=0.0`．キーとバリューは"="で繋ぎ，複数の条件を定義したい場合は","で繋ぐ．|


### 3.2 モデル固有の生成条件を追加するとき
モデルによっては意図した推論をさせるために temperature や system message を指定する必要がある． \
それらの指定は `scripts/generation_settings/custom_model_settings` 以下に \
model publisher ごと，model name ごとに .yaml ファイルで定義することができる．

例えば，`tokyotech-llm/Llama-3.1-Swallow-8B-Instruct-v0.3` についての指定を追加したい場合は \
`scripts/generation_settings/custom_model_settings/tokyotech-llm/Llama-3.1-Swallow-8B-Instruct-v0.3.yaml` \
に定義を行えば良い．

具体的な定義の仕方や定義できるパラメタについては \
`scripts/generation_settings/custom_model_settings/template.yaml` を参照されたい．

> ⚠️注意： \
> temperature を指定する custom_model_settings は，\
> デフォルトのtemperatureが0になっているベンチマークに対してのみ，適用すること． \
> MT-BenchやLiveCodeBenchのようにデフォルトのtemperatureが0でないベンチマークは，\
> 特別な要求がない限り，custom_model_settings を指定しないで実行すること．

> 🗒️Note: \
> 特定の推論が必要な場合，原則，評価依頼者が評価依頼時に生成条件を指定する． \
> しかし，場合によっては依頼者が勘違いや指定漏れをすることがありうる． \
> そのため以下のケースに該当するにもかかわらず生成条件の指定がない場合は，念の為，依頼者に対して再確認してほしい： 
> * モデルカードで推奨システムメッセージが明示されているにもかかわらず，指定されていない
> * モデルカードで推論モードをonにする条件が明示されているにもかかわらず，指定されていない

### 3.3 各モデル用のディレクトリについて
[2.5 評価結果の確認](#25-評価結果の確認)や[2.6 評価ログの確認](#26-評価ログの確認)で「各モデル用のディレクトリ」として \
`results/{provider}/{model_publisher}/{model_name}/{custom_settings}` というパスを記しているが，
このパスに含まれる各要素は以下の通りである．

| 表記 | 意味 |
| -- | -- |
| `{provider}` | モデルをserve（立てる）サービスの名前．vllmの場合は`hosted_vllm`，deepinfraの場合は`deepinfra`，openaiの場合は空文字となる．|
| `{model_publisher}` | Huggingface Model ID を"/"で区切った時の前半部分．モデルを提供する団体を指す．（e.g. `tokyotech-llm`）
| `{model_name}` | Huggingface Model ID を"/"で区切った時の後半部分．モデル名を指す．（e.g. `Llama-3.1-Swallow-8B-Instruct-v0.3`）
| `{custom_settings}` | モデル固有の特別な生成条件．指定していない場合は空文字となる． |

なお，`{provider}` と `{custom_settings}` については空文字となりうるが，その場合パス内で空文字は端折られる． \
（e.g. `results//tokyotech-llm/swallow/` -> `results/tokyotech-llm/swallow`）

### 3.4 特定のproviderでエラーが出る場合の対応

特定のproviderでエラーが出る場合は，provider固有の制限に抵触している可能性がある．\ 
たとえば `deepinfra/meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8` では並列応答数 `n` に5以上を指定するとエラーを起こす（2025年6月時点）．   \
このような場合は，Jupyter Notebookなどを用いてproviderに適当なリクエストを投げて，調査を行なってほしい．

具体例は以下の通り．  
```
# deepinfra/meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8 の例

import litellm

request_payload = {
    "model": "deepinfra/meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8",
    "messages": [
        {
            "role": "user",
            "content": (
                "こんにちは。なにかしゃべって"
            ),
        }
    ],
    "logprobs": None,
    "base_url": "https://api.deepinfra.com/v1/openai",
    "n": 2,
    "caching": False,
    "api_key": "DeepInfraのAPIキー",
    "max_completion_tokens": 512,
    "temperature": 1.0
}

responses = litellm.completion(**request_payload)
```

### 3.5 並列応答数 `n` を強制的に1にする方法

JHumanEval や LiveCodeBench のように "N回解いて正答率を測定する" ベンチマークでは，\
並列応答数 `n` をN（たとえばJHumanEvalならN=10）で設定している． \
providerが並列応答数を制限していてエラーになる場合などは generation_parameters に `max_n` の指定を追加してほしい．

具体例は以下の通り（上：generation_parametersに追記する例．下：手元で検証を行う例．）
```yaml
max_n_specified:
  max_n: "4"
  version: "1"
```

```
lighteval endpoint litellm \
    "model=$MODEL_NAME,api_key=$API_KEY,base_url=$BASE_URL,generation_parameters={temperature:0.2,top_p:0.95,max_n:4}" \
    "swallow|lcb:codegeneration_v5_6|0|0" \
    ...
```

### 3.6 評価がいつまでたっても終わらない場合

評価がいつまでたっても終わらない場合は，\
1 ) repetitionが多発してvLLMのスループットが極端に低下している または 2 ) 計算資源が足りない のどちらかが疑われる． \
このような場合はvLLMログを見てスループットの極端な低下やKV Cache不足のWARNING多発を確認してほしい．
- Ref. [vLLM Optimization and Tuning](https://docs.vllm.ai/en/latest/configuration/optimization.htm)

基本的に計算資源の増強で解決したい．\
しかし，それが難しい場合は，依頼者に報告したうえで，`MAX_MODEL_LENGTH` を8,192まで下げることで解決を図りたい．

vLLMログに `Aborted request` が出力されている，またはlightevalログに `Timeout` が出力されている場合は，推論に時間がかかりすぎてAPI呼び出しがタイムアウトしている．  
この場合は環境変数 `REQUEST_TIMEOUT` （単位は秒）に十分に大きな値を設定すること．
