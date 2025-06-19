from lighteval.tasks.lighteval_task import LightevalTaskConfig
import lighteval.tasks.default_prompts as prompt
from lighteval.metrics.metrics import Metrics

math_500_swallow = LightevalTaskConfig(
    name="math_500",
    suite=["swallow"],
    prompt_function=prompt.math_500,
    hf_repo="HuggingFaceH4/MATH-500",
    hf_subset="default",
    hf_avail_splits=["test"],
    evaluation_splits=["test"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=None,  # swallow用に変更
    metric=[Metrics.latex_gold_metric],
    version=1,
)
