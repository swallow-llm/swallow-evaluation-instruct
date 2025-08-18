# 評価における課題と解決策

この文書では，評価作業で生じやすい課題への対処方法を


## vLLM で公式にサポートされていない Reasoning parser を使う方法．．
reasoning-parser を指定することにより，モデルの出力から推論過程が取り除かれた最終出力部分のみを採点の対象とすることができ，推論型モデルを適切に評価することができます．（Ref. [vLLM Doc: Reasoning Outputs](https://docs.vllm.ai/en/stable/features/reasoning_outputs.html)）

本フレームワークでは[vLLM を予め serve する方法](#１vllm-serve--lighteval推奨)では vLLM==0.9.2，[lighteval から直接 vLLM を呼び出す方法](#2lighteval--vllm-非推奨)では vLLM==0.9.1 が使用されるため，https://github.com/vllm-project/vllm/tree/v0.9.2/vllm/reasoning から使用することができる vLLM 公式の reasoning parser を確認することができます．

ここで，[nvidia/Llama-3.1-Nemotron-Nano-8B-v1](https://huggingface.co/nvidia/Llama-3.1-Nemotron-Nano-8B-v1) のように vLLM に公式でサポートされていないが，その think タグが `<thinkg>, </think>` である推論型モデルについては， reasoning parser に `deepseek_r1_markup` を指定することで，適当に推論過程の除去が行われます．

## OpenAI APIや互換性のあるAPIを使用する場合の注意点．
本フレームワークでは [litellm](https://github.com/BerriAI/litellm) をバックエンドとして指定することにより，OpenAIのように推論APIだけが提供されているモデルも評価することができます．
また，NVIDIA NIM や DeepInfra のようなOpenAI互換の推論APIも対応しています．
ただしAPIプロバイダ固有の仕様（並列リクエスト数など）によりエラーが起きることがあるので注意してください．
