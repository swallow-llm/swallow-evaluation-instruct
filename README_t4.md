# TSUBAME4を用いた評価方法

> 目次
> - [概要](#概要)
> - [更新履歴](#更新履歴)

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
1.1 で評価者固有情報を記載したファイルの名前を`.env_template` から `.env` に変更しておく．
これにより評価スクリプトから読み取られるようになり，また，`gitignore` の対象となる．
> ⚠️ 注意：`.env` には　API キーが含まれているため決してパブリックに公開してはならない．

### 1.3 環境構築スクリプトの実行
1.1，1.2 で正しく評価者固有情報が登録できていることを確認した上で，以下のスクリプトを実行し，環境構築を行う．
```bash
# 以下は Saito の例
cd /gs/fs/tga-okazaki/saito/swallow-evaluation-instruction-private
bash scripts/tsubame/environment/setup_t4_uv_envs.sh
```

最終的に `"✅ Environment was successfully created!"` が表示されれば成功．


## 2. 評価の実行
### 2.1 モデルの設定
まず，評価を行うモデルについて以下の情報を`swallow-evaluation-instruction-private/scripts/tsubame/qsub_all.sh`の `# Set Args` の欄に書き込む．

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
2.1 でモデルの設定を終えたら同ファイル（`swallow-evaluation-instruction-private/scripts/tsubame/qsub_all.sh`）下部の `# Submit tasks` 以降を編集し，評価するタスクを指定する．具体的には，評価しないタスクについてコメントアウトをすれば良い．

### 2.3 評価の実行
2.1，2.2 でモデルとタスクの指定を正しく行ったことを確認したのち，以下のスクリプトで評価のジョブを投げることができる．
```sh
# 以下は Saito の例
cd /gs/fs/tga-okazaki/saito/swallow-evaluation-instruction-private
bash scripts/tsubame/qsub_all.sh
```

### 2.4 評価状況の確認
以下のスクリプトから評価状況を確認することができる．
```sh
# 以下は Saito の例
cd /gs/fs/tga-okazaki/saito/swallow-evaluation-instruction-private
bash scripts/tsubame/utils/save_and_check_qstat.sh
```

### 2.5 評価結果の確認
評価の結果は各モデル用のディレクトリ（`swallow-evaluation-instruction-private/results/{model_publisher}/{model_name}`）以下に`aggregated_results.json`という名前で保存される．
その中で，`overall`の値をコピーして指定された spreadsheet に貼り付ければ良い．


### 2.6 評価ログの確認
評価のログ（標準出力・標準エラー出力）は各モデル用のディレクトリ（`swallow-evaluation-instruction-private/results/{model_publisher}/{model_name}`）以下の言語・タスクごとのディレクトリにそれぞれ `.o` ファイル，`.e` ファイルとして保存される．


### 2.7 評価詳細の確認（lightevalを用いた評価の場合のみ）
評価結果の詳細は `swallow-evaluation-instruction-private/lighteval/outputs/results` 以下に `.json` ファイルとして，モデルが生成した回答の詳細は `swallow-evaluation-instruction-private/lighteval/outputs/outputs` 以下に `.pqt` ファイルとして保存されている．`.pqt`ファイルは pandas を用いて dataframe として開くことができる．（`swallow-evaluation-instruction-private/scripts/utils/details_viewer.ipynb`参照）