# Swallowチームが実装したベンチマーク一覧

swallow-evaluation-instruct では，[lighteval公式実装](https://github.com/huggingface/lighteval/releases/tag/v0.8.0)に加えて，Swallowチームが実装したベンチマークを評価することができます．  
本文書ではSwallowチームが実装したベンチマークを紹介します．  

## ベンチマーク一覧の表示

Swallowチームが実装したベンチマークの一覧は `lighteval tasks list` コマンドの "swallow" suite に表示されます．

## 共通事項
* 明示なき限り，実験設定はゼロショットを推奨します．  
* MT-Benchを除きデコーディングパラメータは実行時に自由に指定できます．推奨設定がある場合は明記しています．  
* MT-Benchはマルチターン対話，それ以外はシングルターン対話で出題・回答する形式を採用しています．

## 日本語のベンチマーク

### JEMHopQA (v1.2)
知識量や推論能力を評価するための自由記述式質問応答です．

* タスク分類：マルチホップ質問応答
* 出典：[Ishii et al. (2024)](https://aclanthology.org/2024.lrec-main.831/)
* lightevalタスクID：`swallow|jemhopqa_cot`
* データセット：[tokyotech-llm/JEMHopQA](https://huggingface.co/datasets/tokyotech-llm/JEMHopQA)
* 設問数：120問
* Chain-of-Thought (CoT) プロンプト：あり
* 評価尺度：正規化後の文字F1 (f1_score_quasi)
* その他の評価尺度
    * 正規化前の文字F1：f1_score
    * 完全一致：exact_match, quasi_exact_match
    * llm-jp-eval (v1.4.1) 互換の文字F1：llmjpeval_f1_score, llmjpeval_f1_score_quasi
* 派生版
    * CoTプロンプトを付けない `swallow|jemhopqa` があります．  

### BenchMAX Science Reasoning
博士課程レベルの科学問題を集めたベンチマーク GPQA（Mainサブセット）の邦訳版です．

* タスク分類：科学知識に基づく質問応答
* 出典：[Huang et al. (2025)](https://arxiv.org/abs/2502.07346)
* lightevalタスクID：`swallow|swallow_gpqa_ja`
* データセット：[LLaMAX/BenchMAX_Science](https://huggingface.co/datasets/LLaMAX/BenchMAX_Science)
* 設問数：448問
* CoTプロンプト：あり
* 評価尺度：正解率

### M-IFEval Japaneseサブセット
「箇条書きにせよ」のような検証可能な指示を用いて対話における指示追従性を評価するベンチマーク IFEval [Zeng et al. (2024)](https://openreview.net/forum?id=tr0KidwPLc) の日本語ローカライズ版です．  
単なる邦訳ではなく「漢字にふりがなをつけよ」のような日本語の表記に特有の指示が含まれています．

* タスク分類：指示追従
* 設問数：172問，226指示
* 評価尺度：設問レベルの正解率 (instruct_level_strict_accuracy)
* その他の評価尺度

## 英語のベンチマーク

### HellaSwag
次に起こる出来事を予測する4択の選択式問題です．

* タスク分類：常識推論
* CoTプロンプト：なし

### LiveCodeBench v5--v6追加設問
競技プログラミングの設問を用いてコード生成能力を評価する問題です．リリースv5およびv6で追加された設問のみを使用します． 

* タスク分類：コード生成
* 出典：[Jain et al. (2025)](https://openreview.net/forum?id=chfJJYC3iL)
* lightevalタスクID：`swallow|lcb:codegeneration_v5_v6`
* データセット：
* 設問数：342問
* CoTプロンプト：なし
* 推奨設定：temperature=0.6, top-p=0.95
* 評価尺度：Pass@1, Pass@10 (N=10) [Chen et al. (2021)](https://arxiv.org/abs/2107.03374)
* 派生版：`swallow|lcb:codegeneration_{リリースID}` を指定することで評価を行うリリースを変更できます．  
  リリースIDの記法は LiveCodeBench公式実装 [LiveCodeBench/LiveCodeBench](https://github.com/LiveCodeBench/LiveCodeBench) を参照してください．
