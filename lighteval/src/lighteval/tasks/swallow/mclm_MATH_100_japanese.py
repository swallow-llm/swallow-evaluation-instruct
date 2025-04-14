
from lighteval.metrics.dynamic_metrics import (
    ExprExtractionConfig,
    IndicesExtractionConfig,
    LatexExtractionConfig,
    multilingual_extractive_match_metric,
)
from lighteval.tasks.lighteval_task import LightevalTaskConfig
from lighteval.tasks.requests import Doc
from lighteval.utils.language import Language

MATH_JAPANESE_QUERY_TEMPLATE = """
以下の数学の問題を、わかりやすく、論理的に解いてください。  
出力の最後の行は、次の形式にしてください。

回答: $\\boxed{{ANSWER}}$

`ANSWER` には、解答となる数式または数値が入ります。

ステップバイステップで考えてから回答してください。

{Question}
""".strip()

# 具体例
"""
以下の数学の問題を、わかりやすく、論理的に解いてください。  
出力の最後の行は、次の形式にしてください。

回答: $\boxed{{ANSWER}}$

`ANSWER` には、問題の答えに対する最終的な数式または数値が入ります。  

ステップバイステップで考えてから回答してください。

$f(x)=\frac{2x}{x^2-5x-14}$ のグラフには、垂直漸近線 $x=a$ と $x=b$、水平漸近線 $y=c$ があります。$a+b+c$ を求めなさい。
"""

def wrap_answer_with_latex_boxes(str_answer: str):
    TEMPLATE = "$\\boxed{{{ANSWER}}}$"
    
    if not str_answer.startswith("$\\boxed"):
        return TEMPLATE.format(ANSWER=str_answer)
    else:
        return str_answer

def math100_japanese_prompt_fn(line, task_name: str = None):
    return Doc(
        task_name=task_name,
        query=MATH_JAPANESE_QUERY_TEMPLATE.format(Question=line["ja"]),
        choices=[wrap_answer_with_latex_boxes(line["answer"])],
        gold_index=0,
    )

# Evaluation metric
# 回答スパン抽出：数式 (LatexExtractionConfig) と 数量表現 (ExprExtractionConfig) を併用
# ロケール：日本語．要検証
latex_gold_metric = multilingual_extractive_match_metric(
    language=Language.ENGLISH,
    fallback_mode="first_match",
    precision=5,
    gold_extraction_target=(LatexExtractionConfig(),),
    # Match boxed first before trying other regexes
    pred_extraction_target=(ExprExtractionConfig(), LatexExtractionConfig(boxed_match_priority=0)),
    aggregation_function=max,
)

math_100_japanese = LightevalTaskConfig(
    name="math_100_japanese",
    suite=["swallow"],
    prompt_function=math100_japanese_prompt_fn,
    hf_repo="amphora/MCLM",
    hf_subset="MT-MATH100",
    hf_avail_splits=["test"],
    evaluation_splits=["test"],
    few_shots_split=None,
    few_shots_select=None,
    metric=[latex_gold_metric],
    version=1,
)
