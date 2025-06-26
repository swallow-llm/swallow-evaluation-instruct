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

## 概要
この資料は，岡崎研の評価チームが TSUBAME4 上で評価を行う際に参照することを想定した，内部向けのマニュアルである．

## 更新履歴
| 日付 | 内容 | 担当 |
| -- | -- | -- |
| 2025/06/19 | 初稿 | 齋藤 |

## 1. 環境構築
### 1.1 評価者固有情報の登録
`swallow-evaluation-instruction-private/.env_template` に記されている以下の変数について各評価者の環境に合わせて修正を行う．

| 変数名 | 説明 | 備考 | 
| -- | -- | -- |
| `GROUP_AND_NAME` | 以下の変数の設定を簡単にするための変数． | 必ずしも使う必要はない． |
| `REPO_PATH` | 評価レポジトリの絶対パス．| デバッグなどで複数のレポジトリを持っている場合は注意． |
| `HUGGINGFACE_CACHE` | HuggingFace のデータセットやモデルのインストールなどに関するキャッシュを保存しておくディレクトリ． | 必ず `/gs/bs/` 以下のディレクトリを指定すること． |
| `UV_CACHE` | UV のパッケージインストールなどに関するキャッシュを保存しておくディレクトリ．| 容量はそこまで大きくならないことが想定されるが，念の為 `/gs/bs/` 以下のディレクトリを指定するとよい．|
| `VLLM_CACHE` | VLLM の動作に関するキャッシュを保存しておくディレクトリ．| 容量がどれほど占められるかは把握していないが，念の為`/gs/bs/` 以下のディレクトリを指定するとよい．|
| `OPENAI_API_KEY` | MT-Bench の評価における LLM-as-a-judge で OpanAI のモデルを使用するときや，OpenAI のモデルを評価するよきに使用する． | Swallow project 用の API を指定すること．お金を消費するので要注意．|
| `HF_TOKEN` | HuggingFace のデータセットやモデルのインストール時に用いられるトークン．| Rate Limit の緩和や使用許可が必要なデータセット・モデルを使用する際に必要．|

### 1.2 評価者固有情報ファイルの名前変更
1.1 で評価者固有情報を記載したファイルの名前を`.env_template` から `.env` に変更しておく． \
これにより評価スクリプトから読み取られるようになり，また，`gitignore` の対象となる．
> ⚠️ 注意：`.env` には　API キーが含まれているため決してパブリックに公開してはならない．

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
まず，評価を行うモデルについて以下の情報を`swallow-evaluation-instruct-private/scripts/tsubame/qsub_all.sh`の `# Set Args` の欄に書き込む．

| 変数名 | 説明 | 備考 |
| -- | -- | -- |
| `NODE_KIND` | 使いたいTSUBAME4のノード．["node_q", "node_f"] | 13B以下なら"node_q"，13B超なら"node_f"を選ぶと良い．|
| `MODEL_NAME`| 評価するモデルのHuggingFaceID．| HuggingFaceのモデルカード上部にあるコピーボタンから取得できる．|
| `SYSTEM_MESSAGE` | 評価するモデルに渡すシステムメッセージ．| 必要な場合のみ指定．基本的に指定しなくて良い．|
| `PROVIDER` | 評価するモデルを serve するための provider．| HuggingFaceのモデルであれば"vllm"（デフォルト），OpenAI のモデルなら"openai"を指定．Deepinfra を使う場合は"deepinfra"を指定する．|
| `PRIORITY` | 使いたいTSUBAME4における優先度．["-5", "-4", "-3"] | 数値が大きい方がジョブが流れやすくなる．しかし，それに応じて値段が2倍，4倍と高くなるので，指定する場合は要相談．|
| `MAX_MODEL_LENGTH` | 評価するモデルの生成時に渡す引数．入力と出力の合計の最大値であり，この大きさのKV CACHEが確保される．| モデルのconfigから自動で取得を行うので基本的に指定は不要．自動取得に失敗する場合のみ指定．|
| `MAX_COMPLETION_TOKENS` | 評価するモデルの生成時に渡す引数．出力の最大トークン数の制約である．| モデルの `MAX_MODEL_LENGTH` から自動計算されるので，基本的に指定は不要．必要な場合のみ指定．|

### 2.2 タスクの設定
2.1 でモデルの設定を終えたら \
同ファイル（`swallow-evaluation-instruct-private/scripts/tsubame/qsub_all.sh`）下部の `# Submit tasks` 以降を編集し， \
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
各モデル用のディレクトリ（`swallow-evaluation-instruct-private/results/{model_publisher}/{model_name}`）以下に保存される． \
評価担当者は `aggregated_results.json` 内の `overall`の値を，指定された spreadsheet にコピーすれば良い．


### 2.6 評価ログの確認
評価のログ（標準出力 `.o` ファイル・標準エラー出力 `.e` ファイル）は \
各モデル用のディレクトリ（`swallow-evaluation-instruct-private/results/{model_publisher}/{model_name}`）以下の言語・タスクごとに保存される．


### 2.7 評価詳細の確認（lightevalを用いた評価の場合のみ）
評価結果の詳細は `swallow-evaluation-instruct-private/lighteval/outputs/results` 以下に `.json` ファイルとして， \
モデルが生成した回答の詳細は `swallow-evaluation-instruction-private/lighteval/outputs/outputs` 以下に `.pqt` ファイルとして保存されている． \
`.pqt`ファイルは pandas を用いて dataframe として開くことができる．（`swallow-evaluation-instruction-private/scripts/utils/details_viewer.ipynb`参照）


## 3. Tips
### 3.1 タスクを追加するときに
タスクを追加する際には `lighteval/src/ligtheval/tasks/swallow/` 以下にタスクの定義を書く． \
しかし，その操作はあくまで lighteval に対するの操作であり，swallow-evaluation-instruct 用には追加の操作が必要である． \
以下にそれをまとめる．

| カテゴリ | 必要性 | 操作対象 | 操作内容 |
| -- | -- | -- | -- |
| 結果集約（Aggregate）のための操作 | 必須 | `scripts/aggregate_utils/conf.py` | 追加したタスクに対応するメトリクスを定義する| 
| | 適宜 |`scripts/aggregate_utils/funcs.py` | 追加したタスクのメトリクスに必要な計算を追加することができる |
| | 適宜 |`scripts/aggregate_utils/white_lists.py` | 追加したタスクのメトリクスの計算に用いるタスクサブセットのサブセットを定義することができる |
| 評価実行のための操作 | 必須 | `scripts/tsubame/conf` | 追加したタスクについて，`key`（"{言語}_{タスク名}"），`script`（定義したタスク名），`result_dir`（結果・ログの出力先），`framework`（フレームワーク），`hrt_q`（node_qでの想定所要時間），`hrt_f`（node_fでの想定所要時間），を定義する |
| | 必須 | `scripts/tsubame/qsub_all.sh` | 追加したタスクについて，`qsub_task {言語} {タスク名}` を末尾の適当な箇所に追加する．|
| | 適宜 | `scripts/tsubame/qsub_all.sh` | 追加したタスクの生成条件を `GEN_PARAMS_LIST` に追加する．|