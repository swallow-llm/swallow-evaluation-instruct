# Swallowチームが実装したベンチマーク一覧

swallow-evaluation-instruct では，[lighteval公式実装](https://github.com/huggingface/lighteval/releases/tag/v0.8.0)に加えて，Swallowチームが実装したベンチマークを評価することができます．  
本文書ではSwallowチームが新規に実装またはlighteval公式実装を改変したベンチマークを紹介します． 

## ベンチマーク一覧の表示

Swallowチームが実装したベンチマークの一覧は `lighteval tasks list` コマンドの "swallow" suite に表示されます．

## 共通事項
* MT-Benchはマルチターン対話，それ以外はシングルターン対話で出題・回答する形式を採用しています．
* MT-Benchを除き，デコーディングパラメータは実行時に自由に指定できます．推奨設定がある場合は明記しています．  
  なおSwallowリーダーボードの場合，コンテキスト長さは最長32,768トークン，推奨設定を書いていないベンチマークは貪欲法（temperature=0）を採用しています．  
* 深い推論を妨げる可能性があるため，タスクIDの後ろに0をつけてゼロショットで評価することを推奨します．  
  例： `swallow|swallow_jmmlu|0|0`

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
* 実装の出典：Dussolleらの実装 [lightblue-tech/M-IFEval](https://github.com/lightblue-tech/M-IFEval) を忠実に再現したうえで，言語判定器の初期化処理を追加しています．

### WMT20 英日翻訳
ニュース記事の翻訳を行うベンチマーク WMT20 [Barrault et al. (2020)](https://aclanthology.org/2020.wmt-1.1/) の英日翻訳サブセットです．

* タスク分類：機械翻訳
* lightevalタスクID：`swallow|wmt20:en-ja`
* データセット：[lighteval/sacrebleu_manual/wmt20/en-ja.jsonl](https://huggingface.co/datasets/lighteval/sacrebleu_manual/blob/main/wmt20/en-ja.jsonl)
* 設問数：1,000文
* CoTプロンプト：なし
* 評価尺度：BLEU（corpus BLEU；全設問のマイクロ平均）  
  分かち書きは MeCab+IPADIC 互換の Janome を使用し，計算には sacreBLEU ([Post (2018)](https://aclanthology.org/W18-6319/)) を用います．モデル出力から翻訳文を抽出できなかった場合は空文字として扱います．  
* その他の評価尺度
    * Nagisa [taishi-i/nagisa](https://github.com/taishi-i/nagisa) で分かち書きしたBLEU ([JP LM Eval. Harness](https://github.com/tdcyamadaya/lm-evaluation-harness-jp-stable)準拠)：bleu_lmevalja
* 注意事項：プロンプトで指示するとおり `日本語: ` に続けて邦訳文を出力する必要があるため，指示追従性の低いモデルはスコアが極端に低くなる場合があります．  

### WMT20 日英翻訳
ニュース記事の翻訳を行うベンチマーク WMT20 [Barrault et al. (2020)](https://aclanthology.org/2020.wmt-1.1/) の日英翻訳サブセットです．

* タスク分類：機械翻訳
* lightevalタスクID：`swallow|wmt20:ja-en`
* データセット：[lighteval/sacrebleu_manual/wmt20/ja-en.jsonl](https://huggingface.co/datasets/lighteval/sacrebleu_manual/blob/main/wmt20/ja-en.jsonl)
* 設問数：993文
* CoTプロンプト：なし
* 評価尺度：BLEU（corpus BLEU；全設問のマイクロ平均）．計算には sacreBLEU ([Post (2018)](https://aclanthology.org/W18-6319/)) を用います．モデル出力から翻訳文を抽出できなかった場合は空文字として扱います．  
* 注意事項：プロンプトで指示するとおり `English: ` に続けて英訳文を出力する必要があるため，指示追従性の低いモデルはスコアが極端に低くなる場合があります．  

### MCLM MATH-100（日本語）
多言語の競技数学ベンチマークスイート MCLM [Son et al. (2025)](https://aclanthology.org/2025.acl-long.699/) のうち，MATH-500 [Lightman et al. (2024)](https://openreview.net/forum?id=v8L0pN6EOi) を出典とするサブセット MT-MATH100 から日本語の設問を抽出したものです．  

* タスク分類：数学
* lightevalタスクID：`swallow|math_100_japanese`
* データセット：[amphora/MCLM](https://huggingface.co/datasets/amphora/MCLM)
* ライセンス：MIT License
* 設問数：99問
* CoTプロンプト：あり
* 評価尺度：正解率．数式や数値による回答を正解と照合して正誤判定します．

### JMMLU
一般教養を問う4値選択式ベンチマーク MMLU [Hendrycks et al.](https://openreview.net/forum?id=d7KBjmI3GmQ) の邦訳版です．
STEM・社会科学・人文科学・その他の4カテゴリに属する56科目のうち，CC BY-NC-ND 4.0ライセンスの3科目を除く53科目を評価します．

* タスク分類：一般教養
* 出典：[尹ら (2024)](https://www.anlp.jp/proceedings/annual_meeting/2024/pdf_dir/A7-5.pdf)
* lightevalタスクID：`swallow|swallow_jmmlu`
* データセット：[nlp-waseda/JMMLU](https://huggingface.co/datasets/nlp-waseda/JMMLU)
* ライセンス：CC BY-SA 4.0
* 設問数：7,097問
* CoTプロンプト：あり
* 評価尺度：正解率．

### MMLU-ProX（日本語）
MMLU-Pro [Wang et al. (2024)](https://openreview.net/forum?id=y10DM6R2r3) の低品質な設問を削除したうえで邦訳した，一般教養を問うベンチマークです．
出題形式は MMLU-Pro と同じく多肢選択式で，最大で10件の選択肢が提示されます．

* タスク分類：一般教養
* 出典：[Xuan et al. (2025)](https://arxiv.org/abs/2503.10497)
* lightevalタスクID：`swallow|mmlu_prox_japanese`
* データセット：[tokyotech-llm/MMLU-ProX-Japanese](https://huggingface.co/datasets/tokyotech-llm/MMLU-ProX-Japanese)
    * オリジナル [li-lab/MMLU-ProX](https://huggingface.co/datasets/li-lab/MMLU-ProX) の ja subset かつ test split を複製して，科目別のサブセットを作成しました．
* 設問数：11,759問
* CoTプロンプト：あり
* 評価尺度：正解率

### JHumanEval
コード生成能力を評価するベンチマーク HumanEval [Chen et al. (2021)](https://arxiv.org/abs/2107.03374) の邦訳版です．

* タスク分類：コード生成
* 出典：[佐藤ら (2024)](https://www.anlp.jp/proceedings/annual_meeting/2024/pdf_dir/P10-9.pdf)
* lightevalタスクID：`swallow|swallow_jhumaneval`
* データセット：[kogi-jwu/jhumaneval](https://huggingface.co/datasets/kogi-jwu/jhumaneval)
* ライセンス：MIT License
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
* 事前準備：OpenAI API Key を 環境変数 `OPENAI_API_KEY` に設定してください．  
* 注意事項：コンテキスト超過エラーを防ぐため，1ターン目の応答文（深い推論過程を含まない最終出力）は最長8,192文字で切り詰めます．  

## 英語のベンチマーク

### HellaSwag
日常的な状況の文脈から次に起こる出来事を予測する常識推論能力を問う，4値選択式のベンチマークです。

* タスク分類：常識推論
* 出典：[Zellers et al. (2019)](https://aclanthology.org/P19-1472/)
* lightevalタスクID：`swallow|hellaswag`
* データセット：[Datasets](https://huggingface.co/docs/datasets/index) library `hellaswag`, [オリジナル](https://github.com/rowanz/hellaswag)
* ライセンス：MIT License
* 設問数：10,042件
* CoTプロンプト：あり
* 評価尺度：正解率

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
* 派生版：`swallow|lcb:codegeneration_{リリースID}` を指定することで評価するリリースを変更できます．  
  リリースIDの記法は LiveCodeBench公式リポジトリ [LiveCodeBench/LiveCodeBench](https://github.com/LiveCodeBench/LiveCodeBench) を参照してください．  
* 実装の出典：lighteval標準実装 `extended|lcb:codegeneration` のプロンプトおよびコードブロック抽出を改変しています．

### MMLU
STEM・社会科学・人文科学・その他の4カテゴリに属する57科目で構成される，高校から大学学部および専門職試験に相当する一般教養を問う4値選択式のベンチマークです．

* タスク分類：一般教養
* 出典：[Hendrycks et al. (2021)](https://openreview.net/forum?id=d7KBjmI3GmQ)
* lightevalタスクID：`swallow|mmlu_english`
* データセット：[lighteval/mmlu](https://huggingface.co/datasets/lighteval/mmlu)
* 設問数：14,042問
* CoTプロンプト：あり
* 評価尺度：正解率

### MMLU-Pro
一般教養を問うベンチマーク MMLU の難易度を高めた，多値選択式のベンチマークです．  
MMLU を発展させて，選択肢を最大10件に増加，推論を要求する設問の追加，および低品質な設問の削除が行われています．
ビジネス・法学・心理学・生物学・化学・歴史・保健/医療・経済学・数学・物理学・計算機科学・哲学・工学・その他の14カテゴリで構成されます．

* タスク分類：一般教養
* 出典：[Wang et al. (2024)](https://openreview.net/forum?id=y10DM6R2r3)
* lightevalタスクID：`swallow|mmlu_pro_english`
* データセット：[tokyotech-llm/MMLU-Pro](https://huggingface.co/datasets/tokyotech-llm/MMLU-Pro)
    * オリジナル [TIGER-Lab/MMLU-Pro](https://huggingface.co/datasets/TIGER-Lab/MMLU-Pro) の test split を複製して，科目ごとにサブセットを作成しました．
* ライセンス：MIT License
* 設問数：12,032問
* CoTプロンプト：あり
* 評価尺度：正解率．

### MMLU-ProX (英語)
MMLU-Pro [Wang et al. (2024)](https://openreview.net/forum?id=y10DM6R2r3) の低品質な設問を削除して29言語に翻訳した，一般教養を問う多言語ベンチマークの英語サブセットです．
出題形式は MMLU-Pro と同じく多肢選択式で，最大で10件の選択肢が提示されます．

* タスク分類：一般教養
* 出典：[Xuan et al. (2025)](https://arxiv.org/abs/2503.10497)
* lightevalタスクID：`swallow|mmlu_prox_english`
* データセット：[li-lab/MMLU-ProX](https://huggingface.co/datasets/li-lab/MMLU-ProX)
* データセット：[tokyotech-llm/MMLU-ProX-English](https://huggingface.co/datasets/tokyotech-llm/MMLU-ProX-English)
    * オリジナル [li-lab/MMLU-ProX](https://huggingface.co/datasets/li-lab/MMLU-ProX) の en subset かつ test split を複製して，科目ごとにサブセットを作成しました．
* 設問数：11,759問
* CoTプロンプト：あり
* 評価尺度：正解率

### GPQA（Diamond）
化学・物理学・生物学の博士課程レベルの設問を集めた4値選択式ベンチマーク GPQA のうち，高品質かつ非専門家の正答率が低い設問に限定した Diamond サブセットです．

* タスク分類：科学
* 出典：[Rein et al. (2024)](https://openreview.net/forum?id=Ti67584b98)
* lightevalタスクID：`swallow|gpqa:diamond`
* データセット：[Idavidrein/gpqa](https://huggingface.co/datasets/Idavidrein/gpqa)
* ライセンス：CC BY 4.0
* 設問数：198問
* CoTプロンプト：あり
* 評価尺度：正解率．
* 実装の出典：lighteval標準実装 `lighteval|gpqa:diamond` の出力トークン数制限を解除しています．

### MATH-500
数学能力を問う自由記述式のベンチマークです．
高校の競技数学レベルの問題で構成された MATH データセット [Hendrycks et al. (2021)](https://openreview.net/forum?id=7Bywt2mQsCe) の test スプリットから [Lightman et al. (2024)](https://openreview.net/forum?id=v8L0pN6EOi) がランダムに抽出した 500 問で構成されます．

* タスク分類：数学
* 出典：[Hendrycks et al. (2021)](https://openreview.net/forum?id=7Bywt2mQsCe), [Lightman et al. (2024)](https://openreview.net/forum?id=v8L0pN6EOi)
* lightevalタスクID：`swallow|math_500`
* データセット：[HuggingFaceH4/MATH-500](https://huggingface.co/datasets/HuggingFaceH4/MATH-500)
* 設問数：500問
* CoTプロンプト：あり
* 評価尺度：正解率．数式や数値による回答を正解と照合して正誤判定します．
* 実装の出典：lighteval標準実装 `lighteval|math_500` の出力トークン数制限を解除しています．

### AIME 2024–2025
AIME（American Invitational Mathematics Examination）の2024年と2025年の過去問から構成される，数学オリンピック予選相当の数学能力を問う整数回答式のベンチマークです．  
AIMEは主に米国高校生を対象とする試験で，代数・幾何・数論・確率・組合せ論から出題されます．

* タスク分類：数学
* 出典：[Art of Problem Solving Wiki](https://artofproblemsolving.com/wiki/)
* lightevalタスクID：`swallow|aime`
* データセット
  * 2024年：[HuggingFaceH4/aime_2024](https://huggingface.co/datasets/HuggingFaceH4/aime_2024)
  * 2025年：[yentinglin/aime_2025](https://huggingface.co/datasets/yentinglin/aime_2025)
* 設問数：60問
* CoTプロンプト：あり
* 評価尺度：正解率．
* 実装の出典：lighteval標準実装 `lighteval|aime{24,25}` の出力トークン数制限を解除しています．

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
コード生成能力を評価するベンチマーク HumanEval の設問はそのままに，誤判定を減らすために単体テストを増強したベンチマークです．  

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
* 事前準備：OpenAI API Key を 環境変数 `OPENAI_API_KEY` に設定してください．
* 注意事項：コンテキスト超過エラーを防ぐため，1ターン目の応答文（深い推論過程を含まない最終出力）は最長8,192文字で切り詰めます．  