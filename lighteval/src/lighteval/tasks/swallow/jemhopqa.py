
import statistics

from lighteval.tasks.lighteval_task import LightevalTaskConfig
from lighteval.tasks.requests import Doc

from lighteval.metrics.metrics import Metric, MetricCategory, MetricUseCase
from lighteval.metrics.utils.metric_utils import SampleLevelMetric, SampleLevelMetricGrouping
from .metrics_sample_openqa_japanese import JapaneseOpenQAExactMatchSamplingFunc

JEMHOPQA_QUERY_TEMPLATE_COT = """
以下の問題に回答してください。二択問題の場合は 'YES' または 'NO' で答えてください。二択ではない問題は正しい文字列を答えてください。
出力の最後の行は、次の形式にしてください。

回答: $\\boxed{{ANSWER}}$

`ANSWER` には、問題の答えが入ります。

ステップバイステップで考えてから回答してください。

問題: {Question}
""".strip()

JEMHOPQA_QUERY_TEMPLATE = """
以下の問題に回答してください。二択問題の場合は 'YES' または 'NO' で答えてください。二択ではない問題は正しい文字列を答えてください。
出力の最後の行は、次の形式にしてください。

回答: $\\boxed{{ANSWER}}$

`ANSWER` には、問題の答えが入ります。

問題: {Question}
""".strip()

# CoT付きプロンプト
def jemhopqa_cot_prompt_fn(line, task_name: str = None):
    question = line["question"]
    choices = [line["answer"]]
    gold_index = 0
    query = JEMHOPQA_QUERY_TEMPLATE_COT.format(Question=question)
    
    return Doc(
        task_name=task_name,
        query=query,
        choices=choices,
        gold_index=gold_index,
        instruction=None,
    )

# [デフォルト] CoTなしプロンプト
def jemhopqa_prompt_fn(line, task_name: str = None):
    question = line["question"]
    choices = [line["answer"]]
    gold_index = 0
    query = JEMHOPQA_QUERY_TEMPLATE.format(Question=question)
    
    return Doc(
        task_name=task_name,
        query=query,
        choices=choices,
        gold_index=gold_index,
        instruction=None,
    )

# 自由記述式問題から回答スパン抽出・正規化・正準化 および accuracy や 文字F1 を計測する Metric．
jemhopqa_extractive_match_metric = SampleLevelMetricGrouping(
    metric_name=JapaneseOpenQAExactMatchSamplingFunc.METRIC_NAMES(),
    higher_is_better=JapaneseOpenQAExactMatchSamplingFunc.METRICS_HIGHER_IS_BETTER(),
    category=MetricCategory.GENERATIVE,
    use_case=MetricUseCase.ACCURACY,
    # JapaneseOpenQAExactMatchSamplingFuncはデフォルト設定を使う
    sample_level_fn=JapaneseOpenQAExactMatchSamplingFunc(
        cfg_exact_match_gold_extractor=None,
        cfg_exact_match_pred_extractor=None,
        cfg_quasi_exact_match_gold_extractor=None,
        cfg_quasi_exact_match_pred_extractor=None,
        instance_level_aggregation_function=max,
    ).sample_level_fn,
    corpus_level_fn={metric_name:statistics.mean for metric_name in JapaneseOpenQAExactMatchSamplingFunc.METRIC_NAMES()}
)

jemhopqa = LightevalTaskConfig(
    name="jemhopqa",
    suite=["swallow"],
    prompt_function=jemhopqa_prompt_fn,
    hf_repo="tokyotech-llm/JEMHopQA",
    hf_subset="v1.2",
    hf_avail_splits=["train", "validation"],
    evaluation_splits=["validation"],
    metric=[jemhopqa_extractive_match_metric],
    few_shots_split=None,
    few_shots_select=None,
    stop_sequence=[],
    trust_dataset=True,
    version=0,    
)

jemhopqa_cot = LightevalTaskConfig(
    name="jemhopqa_cot",
    suite=["swallow"],
    prompt_function=jemhopqa_cot_prompt_fn,
    hf_repo="tokyotech-llm/JEMHopQA",
    hf_subset="v1.2",
    hf_avail_splits=["train", "validation"],
    evaluation_splits=["validation"],
    metric=[jemhopqa_extractive_match_metric],
    few_shots_split=None,
    few_shots_select=None,
    stop_sequence=[],
    trust_dataset=True,
    version=0,    
)