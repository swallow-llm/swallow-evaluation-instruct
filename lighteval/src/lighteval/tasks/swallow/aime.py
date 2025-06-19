from lighteval.tasks.lighteval_task import LightevalTaskConfig
import lighteval.tasks.default_prompts as prompt
from lighteval.metrics.metrics import Metrics

aime_24_swallow = LightevalTaskConfig(
    name="aime:24",
    suite=["swallow"],
    prompt_function=prompt.aime_prompt_fn,
    hf_repo="HuggingFaceH4/aime_2024",
    hf_subset="default",
    hf_avail_splits=["train"],
    evaluation_splits=["train"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=None,
    metric=[Metrics.expr_gold_metric],
    version=1,
)

aime_25_swallow = LightevalTaskConfig(
    name="aime:25",
    suite=["swallow"],
    prompt_function=prompt.aime_prompt_fn,
    hf_repo="yentinglin/aime_2025",
    hf_subset="default",
    hf_avail_splits=["train"],
    evaluation_splits=["train"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=None,
    metric=[Metrics.expr_gold_metric],
    version=1,
)
