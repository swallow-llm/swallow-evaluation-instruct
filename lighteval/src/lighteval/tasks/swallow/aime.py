from lighteval.tasks.lighteval_task import LightevalTaskConfig
import lighteval.tasks.default_prompts as prompt
from lighteval.metrics.metrics import Metrics
from lighteval.metrics.sample_metric_utils import create_passk_metrics, create_majk_metrics, powers_of_two_up_to_n
from copy import deepcopy


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

# Pass@K and Maj@K variant
lst_aime_swallow_passk_majk = []
for num_samples in [16, 32, 64, 128, 256]:
    lst_k = powers_of_two_up_to_n(num_samples)
    # Metricsクラスに属するSampleLevelMetricを指定する場合は .value をつける
    _lst_pass_at_k_metrics = create_passk_metrics(base_metric=Metrics.expr_gold_metric.value, k_values=lst_k, num_samples=num_samples)
    _lst_maj_at_k_metrics = create_majk_metrics(base_metric=Metrics.expr_gold_metric.value, k_values=lst_k, num_samples=num_samples)

    task_config = deepcopy(aime_24_swallow)
    task_config.name = f"aime_N{num_samples}:24"
    task_config.metric = _lst_pass_at_k_metrics + _lst_maj_at_k_metrics
    lst_aime_swallow_passk_majk.append(task_config)

    task_config = deepcopy(aime_25_swallow)
    task_config.name = f"aime_N{num_samples}:25"
    task_config.metric = _lst_pass_at_k_metrics + _lst_maj_at_k_metrics
    lst_aime_swallow_passk_majk.append(task_config)
