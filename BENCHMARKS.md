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
* データセット：[tokyotech-llm/JEMHopQA](https://huggingface.co/datasets/tokyotech-llm/JEMHopQA), [オリジナル](https://github.com/aiishii/JEMHopQA)
* ライセンス：CC BY-SA 4.0
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
* ライセンス：CC BY 4.0
* 設問数：448問
* CoTプロンプト：あり
* 評価尺度：正解率

### M-IFEval Japaneseサブセット
「箇条書きにせよ」のような検証可能な指示を用いて対話における指示追従性を評価するベンチマーク IFEval [Zeng et al. (2024)](https://openreview.net/forum?id=tr0KidwPLc) の日本語ローカライズ版です．  
単なる邦訳ではなく「漢字にふりがなをつけよ」のような日本語の表記に特有の指示が含まれています．

* タスク分類：オープンエンド対話の指示追従
* 出典：[Dussolle et al. (2025)](https://aclanthology.org/2025.findings-naacl.344/), [実装](https://github.com/lightblue-tech/M-IFEval)
* lightevalタスクID：`swallow|mifeval_ja`
* データセット：[tokyotech-llm/M-IFEval-Ja](https://huggingface.co/datasets/tokyotech-llm/M-IFEval-Ja), [オリジナル](https://github.com/lightblue-tech/M-IFEval)
* ライセンス：Apache License Version 2.0 [(LICENSE)](./lighteval/src/lighteval/tasks/swallow/mifeval_ja/LICENSE.txt)
* 設問数：172問，226指示
* 評価尺度：指示レベルの正解率 (instruct_level_strict_accuracy)
* その他の評価尺度
    * 設問レベルの正解率：prompt_level_strict_accuracy
    * 正規化後の指示レベルの正解率：instruct_level_loose_accuracy
    * 正規化後の設問レベルの正解率：prompt_level_loose_accuracy
* その他：言語判定器の初期化を除き，出典の実装を忠実に再現しています．  

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
* データセット：[kogi-jwu/jhumaneval](https://huggingface.co/datasets/kogi-jwu/jhumaneval)
* 設問数：164問
* CoTプロンプト：なし
* 推奨設定：temperature=0.2, top-p=0.95 [Chen et al. (2021)](https://arxiv.org/abs/2107.03374)
* 評価尺度：Pass@1, Pass@10 (N=10)（[Chen et al. (2021)](https://arxiv.org/abs/2107.03374) の不偏推定式に従う）．

### Japanese MT-Bench
オープンエンド対話における有用性を評価するベンチマーク MT-Bench [Zheng et al. (2023)](https://openreview.net/forum?id=uccHPGDlao) の日本語ローカライズ版です．

* タスク分類：オープンエンド対話
* 出典：Stability AI Japan, [Japanese MT-Bench](https://github.com/Stability-AI/FastChat)
* lightevalタスクID：`swallow|japanese_mt_bench`
* データセット：[tokyotech-llm/swallow_japanese_mt_bench](https://huggingface.co/datasets/tokyotech-llm/swallow_japanese_mt_bench), オリジナルのデータセットは以下の通りです．    
  * 設問：[wandb-japan/llm-leaderboard](https://wandb.ai/wandb-japan/llm-leaderboard/artifacts/dataset/mtbench_en_referenceanswer/v0), `mtbench_ja_question:v4`
  * 採点プロンプト：[wandb-japan/llm-leaderboard](https://wandb.ai/wandb-japan/llm-leaderboard/artifacts/dataset/mtbench_en_referenceanswer/v0), `mtbench_ja_prompt:v1`
  * 模範解答：[wandb-japan/llm-leaderboard](https://wandb.ai/wandb-japan/llm-leaderboard/artifacts/dataset/mtbench_en_referenceanswer/v0), `mtbench_ja_referenceanswer:v2` を Swallow チームで独自に校閲したデータ
* ライセンス：Apache License Version 2.0
* 設問数：80問×2ターン
* CoTプロンプト：なし
* 評価尺度：5 回の試行（応答）を LLM-as-a-Judge により 1〜10 のスケールで採点した平均値を10で割ります．審判（judge）は `gpt-4o-2024-08-06` を用います．  
  カテゴリ×ターン別・カテゴリ別・ターン別・全設問 の4区分それぞれについてスコアの平均値を報告します．  
    * カテゴリ×ターン別（例）：judge_score_writing_turn_1_avg
    * カテゴリ別（例）：judge_score_roleplay_avg
    * ターン別：judge_score_overall_turn_{1,2}_avg
    * 全設問：judge_score_overall_avg

## 英語のベンチマーク

### HellaSwag
次に起こる出来事を予測する4択の選択式問題です．

* タスク分類：常識推論
* CoTプロンプト：なし

### LiveCodeBench
競技プログラミングの設問を用いたコード生成能力を評価するベンチマークです．  
リーク対策のためにデータセットが定期的に更新されており，Swallowリーダーボードではv5およびv6で追加された設問（リリースID： `v5_v6`）を使用しています．  

* タスク分類：コード生成
* 出典：[Jain et al. (2025)](https://openreview.net/forum?id=chfJJYC3iL)
* lightevalタスクID：`swallow|lcb:codegeneration_v5_v6`
* データセット：[livecodebench/code_generation_lite](https://huggingface.co/datasets/livecodebench/code_generation_lite)
* 設問数：342問（リリースv5・v6追加設問）
* CoTプロンプト：なし
* 推奨設定：temperature=0.6, top-p=0.95
* 評価尺度：Pass@1, Pass@10 (N=10) [Chen et al. (2021)](https://arxiv.org/abs/2107.03374)
* 派生版：`swallow|lcb:codegeneration_{リリースID}` を指定することで評価を行うリリースを変更できます．  
  リリースIDの記法は LiveCodeBench公式リポジトリ [LiveCodeBench/LiveCodeBench](https://github.com/LiveCodeBench/LiveCodeBench) を参照してください．

### MMLU
一般教養を問う 4 値選択式の英語設問で構成されるベンチマークです．
JMMLU と同じく，商用利用禁止の 3 科目を除く 53 科目について，科目別・カテゴリ別・全体の正解率を算出します．科目カテゴリは STEM／社会科学／人文科学／その他の 4 種類です．

* タスク分類：一般教養
* 出典：[Hendrycks et al.](https://openreview.net/forum?id=d7KBjmI3GmQ)
* lightevalタスクID：`swallow|mmlu_english`
* データセット：[lighteval/mmlu](https://huggingface.co/datasets/lighteval/mmlu)
* 設問数：14,042問
* CoTプロンプト：あり
* 評価尺度：正解率．

### MMLU-Pro
MMLU をクリーニングし，高難易度の設問を追加したベンチマークです．
出題形式は多肢選択式で，最大で 10 件の選択肢が提示されます．

* タスク分類：一般教養
* 出典：[Wang et al. (2024)](https://openreview.net/forum?id=y10DM6R2r3)
* lightevalタスクID：`swallow|mmlu_pro_english`
* データセット：[TIGER-Lab/MMLU-Pro](https://huggingface.co/datasets/TIGER-Lab/MMLU-Pro)
* 設問数：12,032問
* CoTプロンプト：あり
* 評価尺度：正解率．

### GPQA（Diamond）
博士課程レベルの科学問題を集めたベンチマーク GPQA のうち，高品質かつ高難易度な設問を抽出した Diamond サブセットです．
出題形式は多肢選択式です．

* タスク分類：科学
* 出典：[Rein et al. (2024)](https://openreview.net/forum?id=Ti67584b98)
* lightevalタスクID：`swallow|gpqa:diamond`
* データセット：[Idavidrein/gpqa](https://huggingface.co/datasets/Idavidrein/gpqa)
* 設問数：198問
* CoTプロンプト：あり
* 評価尺度：正解率．

### MATH-500
数学能力を問うベンチマークです．
高校の競技数学レベルの問題で構成された MATH データセット [Hendrycks et al. (2021)](https://openreview.net/forum?id=7Bywt2mQsCe) の test スプリットからランダムに抽出された 500 問で構成されます．

* タスク分類：数学
* 出典：[Lightman et al. (2024)](https://openreview.net/forum?id=v8L0pN6EOi)
* lightevalタスクID：`swallow|math_500`
* データセット：[HuggingFaceH4/MATH-500](https://huggingface.co/datasets/HuggingFaceH4/MATH-500)
* 設問数：500問
* CoTプロンプト：あり
* 評価尺度：正解率．

### AIME 24–25
高難易度な数学能力を評価するベンチマークです．  
AIME（American Invitational Mathematics Exam）の 2024 年および 2025 年の設問で構成されます．

* タスク分類：数学
* 出典：[Art of Problem Solving Wiki](https://artofproblemsolving.com/wiki/)
* lightevalタスクID：`swallow|aime`
* データセット
  * 2024 年：[HuggingFaceH4/aime_2024](https://huggingface.co/datasets/HuggingFaceH4/aime_2024)
  * 2025 年：[yentinglin/aime_2025](https://huggingface.co/datasets/yentinglin/aime_2025)
* 設問数：60問
* CoTプロンプト：あり
* 評価尺度：正解率．

### HumanEval
コード生成能力を評価するベンチマークです．

* タスク分類：コード生成
* 出典：[Chen et al. (2021)](https://arxiv.org/abs/2107.03374)
* lightevalタスクID：`swallow|humaneval`
* データセット：[openai/openai_humaneval](https://huggingface.co/datasets/openai/openai_humaneval)
* ライセンス：MIT License
* 設問数：164問
* CoTプロンプト：なし
* 推奨設定：temperature=0.2, top-p=0.95 [Chen et al. (2021)](https://arxiv.org/abs/2107.03374)
* 評価尺度：Pass@1, Pass@10 (N=10)（[Chen et al. (2021)](https://arxiv.org/abs/2107.03374) の不偏推定式に従う）．

### HumanEval+
コード生成能力を評価するベンチマーク HumanEval の設問はそのままで，単体テストを増強したベンチマークです．  

* タスク分類：コード生成
* 出典：[Liu et al. (2023)](https://papers.nips.cc/paper_files/paper/2023/hash/43e9d647ccd3e4b7b5baab53f0368686-Abstract-Conference.html)
* lightevalタスクID：`swallow|humanevalplus`
* データセット：[evalplus/humanevalplus](https://huggingface.co/datasets/evalplus/humanevalplus)
* ライセンス：Apache License Version 2.0
* 設問数：164問
* CoTプロンプト：なし
* 推奨設定：temperature=0.2, top-p=0.95 [Chen et al. (2021)](https://arxiv.org/abs/2107.03374)
* 評価尺度：Pass@1, Pass@10 (N=10)（[Chen et al. (2021)](https://arxiv.org/abs/2107.03374) の不偏推定式に従う）．

### MT-Bench
オープンエンド対話における有用性（usefulness, helpfulness）を評価するベンチマークです．  

* タスク分類：オープンエンド対話
* 出典：[Zheng et al. (2023)](https://openreview.net/forum?id=uccHPGDlao)
* lightevalタスクID：`swallow|english_mt_bench`
* データセット：[tokyotech-llm/swallow_english_mt_bench](https://huggingface.co/datasets/tokyotech-llm/swallow_english_mt_bench)
    * オリジナル [FastChat/fastchat/llm_judge/data/mt_bench](https://github.com/lm-sys/FastChat/tree/main/fastchat/llm_judge/data/mt_bench) の複製です
* ライセンス：Apache License Version 2.0
* 設問数：80問×2ターン
* CoTプロンプト：なし
* 評価尺度：5 回の試行（応答）を LLM-as-a-Judge により 1〜10 のスケールで採点した平均値を10で割ります．審判（judge）は `gpt-4o-2024-08-06` を用います．  
  カテゴリ×ターン別・カテゴリ別・ターン別・全設問 の4区分それぞれについてスコアの平均値を報告します．  
    * カテゴリ×ターン別（例）：judge_score_writing_turn_1_avg
    * カテゴリ別（例）：judge_score_roleplay_avg
    * ターン別：judge_score_overall_turn_{1,2}_avg
    * 全設問：judge_score_overall_avg
