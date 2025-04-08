# swallow-evaluation-instruct-private

lighteval: [v0.8.0](https://github.com/huggingface/lighteval/releases/tag/v0.8.0)

大井のメモ: 仮装環境はvenvではなく、モジュール間の依存関係の管理が可能なuvなどで管理したい

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
