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

### WMT20 英日翻訳
ニュース記事の翻訳を行うベンチマーク WMT20 [Barrault et al. (2020)](https://aclanthology.org/2020.wmt-1.1/) の英日翻訳サブセットです．

* タスク分類：機械翻訳
* lightevalタスクID：`swallow|wmt20:en-ja`
* データセット：[lighteval/sacrebleu_manual/wmt20/en-ja.jsonl](https://huggingface.co/datasets/lighteval/sacrebleu_manual/blob/main/wmt20/en-ja.jsonl)
* 設問数：1,000文
* CoTプロンプト：なし
* 評価尺度：BLEU（corpus BLEU；全文書のマイクロ平均）
  分かち書きは MeCab+IPADIC 互換の Janome を使用し，BLEU は sacreBLEU を用います．モデル出力から翻訳文を抽出できなかった場合は空文字として扱います．

### WMT20 日英翻訳
ニュース記事の翻訳を行うベンチマーク WMT20 [Barrault et al. (2020)](https://aclanthology.org/2020.wmt-1.1/) の日英翻訳サブセットです．

* タスク分類：機械翻訳
* lightevalタスクID：`swallow|wmt20:ja-en`
* データセット：[lighteval/sacrebleu_manual/wmt20/ja-en.jsonl](https://huggingface.co/datasets/lighteval/sacrebleu_manual/blob/main/wmt20/ja-en.jsonl)
* 設問数：993文
* CoTプロンプト：なし
* 評価尺度：BLEU（corpus BLEU；前処理・算出法は前項と同様）．

### MCLM MATH-100（日本語）
競技レベルの多言語数学ベンチマーク MCLM [Son et al. (2025)](https://aclanthology.org/2025.acl-long.699/) のうち，MATH-500 [Lightman et al. (2024)](https://openreview.net/forum?id=v8L0pN6EOi) をソースとするサブセット MT-MATH100 から，日本語の設問を抽出したものです．

* タスク分類：数学
* lightevalタスクID：`swallow|math_100_japanese`
* データセット：[amphora/MCLM](https://huggingface.co/datasets/amphora/MCLM)
* 設問数：99問
* CoTプロンプト：あり
* 評価尺度：正解率．

### JMMLU
一般教養を問う 4 値選択式ベンチマーク MMLU [Hendrycks et al.](https://openreview.net/forum?id=d7KBjmI3GmQ) の邦訳版です．
商用利用禁止の 3 科目を除く 53 科目について，科目別・カテゴリ別・全体の正解率を算出します．科目カテゴリは STEM／社会科学／人文科学／その他の 4 種類です．

* タスク分類：一般教養（多肢択一）
* 出典：[尹ら (2024)](https://www.anlp.jp/proceedings/annual_meeting/2024/pdf_dir/A7-5.pdf)
* lightevalタスクID：`swallow|swallow_jmmlu`
* データセット：[nlp-waseda/JMMLU](https://huggingface.co/datasets/nlp-waseda/JMMLU)
* 設問数：7,097問
* CoTプロンプト：あり
* 評価尺度：正解率（科目・カテゴリ・全体）．

### MMLU-ProX（日本語）
MMLU-Pro [Wang et al. (2024)](https://openreview.net/forum?id=y10DM6R2r3) をクリーニングして邦訳したベンチマークです．
出題形式は MMLU-Pro と同じく多肢選択式で，最大で 10 件の選択肢が提示されます．

* タスク分類：一般教養（多肢択一）
* 出典：[Xuan et al. (2025)](https://arxiv.org/abs/2503.10497)
* lightevalタスクID：`swallow|mmlu_prox_japanese`
* データセット：[li-lab/MMLU-ProX](https://huggingface.co/datasets/li-lab/MMLU-ProX)
* 設問数：11,759問
* CoTプロンプト：あり
* 評価尺度：正解率．

### JHumanEval
コード生成能力を評価するベンチマーク HumanEval [Chen et al. (2021)](https://arxiv.org/abs/2107.03374) の邦訳版です．

* タスク分類：コード生成
* 出典：[佐藤ら (2024)](https://www.anlp.jp/proceedings/annual_meeting/2024/pdf_dir/P10-9.pdf)
* lightevalタスクID：`swallow|swallow_jhumaneval`
* データセット：
* 設問数：164問
* CoTプロンプト：なし
* 推奨設定：temperature=0.2, top-p=0.95
* 評価尺度：Pass@1 (N=10)（[Chen et al. (2021)](https://arxiv.org/abs/2107.03374) の不偏推定式に従う）．

### Japanese MT-Bench
対話能力を評価するベンチマーク MT-Bench [Zheng et al. (2023)](https://openreview.net/forum?id=uccHPGDlao) の邦訳版です．

* タスク分類：オープンエンド対話
* 出典：[wandb-japan, llm-leaderboad](https://wandb.ai/wandb-japan/llm-leaderboard/artifacts/dataset/mtbench_en_referenceanswer/v0)（**これで良いのか自信がないです**）
* データセット：[wandb-japan/llm-leaderboard](https://wandb.ai/wandb-japan/llm-leaderboard/artifacts/dataset/mtbench_en_referenceanswer/v0)  
  * 採点プロンプト：`mtbench_ja_prompt:v1`
  * 設問：`mtbench_ja_question:v4`
  * 模範解答：`mtbench_ja_referenceanswer:v2` を Swallow チームで独自に校閲したデータ
* lightevalタスクID：`swallow|japanese_mt_bench`
* 設問数：80問×2ターン
* CoTプロンプト：なし
* 評価尺度：5 回の試行（応答）を LLM-as-a-Judge により 1〜10 のスケールで採点し，平均値を採用．審判（judge）は `gpt-4o-2024-08-06` を用います．

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
