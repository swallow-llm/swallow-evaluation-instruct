# 評価における課題と解決策

この文書では，swallow-evaluation-instruct（以下，"本フレームワーク"）を用いて評価する際に遭遇しやすい課題およびその解決策を紹介します．  

### 評価結果の詳細を確認する

ベンチマークのスコアだけでなく各設問のプロンプトや出力などの評価結果の詳細を確認したい場合は，lightevalの引数 `--save-details` を指定して実行してください．評価結果の詳細は `{--output-dir引数}/details/{モデル名}/` 以下にParquet形式で保存されます．
評価結果の詳細は設問ごとに1行となる表形式で保存されており，Pandasなどで開くことができます（参考：[Saving and reading results](https://huggingface.co/docs/lighteval/v0.8.0/en/saving-and-reading-results)）．  
評価結果の詳細をPandasで開く例を以下に示します．

```python
import pandas as pd

path = "{output_dir}/details/{モデル名}/{タイムスタンプ}/details_{Task ID}_{タイムスタンプ}.parquet"
df = pd.read_parquet(path)
```

表に含まれる列はベンチマークにより異なりますが，基本的には "instruction" 列に設問，"gold" 列に正解，"predictions" 列にモデルの応答文，"specifics" 列（Dict形式）に応答文から抽出したモデルの回答が格納されます．

### vLLMが対応していない推論型モデルを評価する

まれにvLLMのreasoning parserが対応していない推論型モデルが存在します．たとえば [nvidia/Llama-3.1-Nemotron-Nano-8B-v1](https://huggingface.co/nvidia/Llama-3.1-Nemotron-Nano-8B-v1) は DeepSeek-R1 と同じく `<thinkg>, </think>` タグで推論過程をマークアップするモデルですが `vllm serve --reasoning-parser deepseek_r1` を実行するとエラーが生じます（vLLM v0.10.0で検証）．  

このようなモデルの場合は `vllm serve` 実行時引数から `--reasoning-parser` を削除して，かわりに lighteval 実行時引数の MODEL_ARGS で `reasoning_parser` を指定することにより，vLLMではなくlighteval内で推論過程と最終出力を分離することで評価を実行できます．  
DeepSeek-R1形式でマークアップするモデルの場合は `deepseek_r1_markup` を指定してください．

### プロバイダが提供する推論APIでエラーが起きる

本フレームワークは，OpenAI互換の推論APIを提供するプロバイダ（[LiteLLM Supported Providers](https://docs.litellm.ai/docs/providers)），たとえばGoogle AI StudioのGemini系列モデルの評価にも対応しています．しかしプロバイダやモデルに固有の仕様が原因でエラーが起きることがあり網羅的な検証はできないため，プロバイダが提供する推論APIでの動作は保証しておりません．  

プロバイダが提供する推論APIでありがちなエラーとして，リクエスト数の制限または，1回の呼び出しの応答数の制限に抵触するケースが挙げられます．
リクエスト数制限については環境変数 `LITELLM_CONCURRENT_CALLS` を小さくしてみてください．
応答数制限が問題になるのはそもそも LiveCodeBench のように設問ごとに複数の応答を要求するベンチマークのみですが，応答数制限に抵触してエラーになる場合は MODEL_ARGS:generation の `max_n` を1に設定してみてください．たとえばOpenAIは応答数が8までに制限されているモデルが多いので `max_n:1` の設定を推奨します（2025年8月時点）．

上記のケースで解決しない場合は，LiteLLMパッケージからプロバイダの推論APIを直接呼び出してみて問題を切り分けることをおすすめします（参考：[Chat Completions](https://docs.litellm.ai/docs/completion)）．

### 推論の深さを指定する

LiteLLMが対応しているOpenAI o系列などの推論型モデルでは，推論の深さを指定できます（参考：[Reasoning models](https://platform.openai.com/docs/guides/reasoning)）．
具体的には lighteval 実行時引数の MODEL_ARGS generation で `reasoning_effort` を指定してください．OpenAI o3の例を以下に示します．

```
# OpenAI o3 で reasoning effort: high を指定する例
API_KEY="{OpenAI APIキー}"
BASE_URL="https://api.openai.com/v1/"
MODEL="openai/o3-2025-04-16"

uv run --isolated --locked --extra lighteval \
lighteval endpoint litellm \
"model=$MODEL,api_key=$API_KEY,base_url=$BASE_URL,generation_parameters={reasoning_effort:\"high\"}" \
"swallow|gpqa:diamond|0|0" \
--output-dir ./lighteval/outputs
```

オープンウェイトの推論型モデルを自身の計算環境で評価する場合は，LiteLLMおよびvLLMの両方が当該モデルのreasoning_effort指定に対応している必要があります．

### 評価の進捗が遅い

LiveCodeBenchのように深い推論を必要とする高難易度のベンチマークを推論型モデルに解かせるケースでは出力が1万トークン以上になることが珍しくないため，ベンチマークやモデルによっては評価に数時間以上かかることがあります．

評価の進捗が想定よりも遅い場合は，推論APIを呼ぶときの最大並列数を増やすことで改善する場合があります．
プロバイダが提供する推論APIの場合はリクエスト数制限に注意しながら，オープンモデルを自身の計算環境で評価する場合はvLLMのスループットに注意しながら（参考： [vLLM Optimization and Tuning](https://docs.vllm.ai/en/latest/configuration/optimization.htm)） `LITELLM_CONCURRENT_CALLS` （デフォルト値は20）の値を大きくしてみてください．

動作が不安定なオープンモデルの場合は，コンテキスト長さの限界まで出力しつづけてしまって評価が進まないケースがあります．Swallowチームの経験では，延々と同じ文字列を反復する"繰り返し"が起きやすいモデルや，深い推論の途中で考え直しすぎて最終出力に到達しない推論型モデルが存在することを確認しています．
動作の不安定が疑われる場合は，lighteval実行時引数 `--save-details` を指定して評価結果の詳細からモデルの応答文を確認することをおすすめします．同時に `--max-samples` を指定して数件だけ観察するとよいでしょう．

長すぎる出力が確認された場合は `vllm serve` の実行時引数 `--max-model-len` の値を小さくしてコンテキスト長さを制限することで，いわば回答を強制的に打ち切ることで評価の進捗を加速できます．また推論型モデルの場合は，まれに temperature や top_p などを調整して推論が安定することがありますので，モデルカードや開発者コミュニティを参照することもおすすめします．いずれにせよ動作が不安定なモデルは性能の評価がむずかしいことにご注意ください．  

### [非推奨・動作保証外] 事前学習済みモデルをむりやり評価したい

[Swallowチームが実装したベンチマーク](./BENCHMARKS.md)はすべてゼロショット設定かつ"考えてから解く"ことができる事後学習済みモデルを想定した仕様になっていますので，"続きの単語を予測する"事前学習済みモデルの評価に使うことは非推奨かつ動作を保証しておりません（かわりに [swallow-evaluation](https://github.com/swallow-llm/swallow-evaluation) をご検討ください）．

非推奨を承知のうえで事前学習済みモデルをむりやり評価したい場合は `vllm serve` 実行時引数の `--chat-template` に `./resources/chat_template_base_model.jinja` を指定してください．このチャットテンプレートは設問のプロンプトを対話形式ではなくただの平文としてモデルに渡す働きをします．実行例を以下に示します．

```
# チャットテンプレートを適用
MODEL="Qwen/Qwen2.5-1.5B"

uv run --isolated --locked --extra vllm \
vllm serve $MODEL \
--chat-template ./resources/chat_template_base_model.jinja \
--host localhost \
--port 8000
```

`lighteval` コマンドは通常通りとなります．
