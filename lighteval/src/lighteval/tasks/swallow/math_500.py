from lighteval.tasks.lighteval_task import LightevalTaskConfig
import lighteval.tasks.default_prompts as prompt
from lighteval.metrics.metrics import Metrics
from lighteval.metrics.sample_metric_utils import create_passk_metrics, create_majk_metrics, powers_of_two_up_to_n
from copy import deepcopy

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

# Pass@K and Maj@K variant
lst_math_500_swallow_passk_majk = []
for num_samples in [16, 32, 64, 128, 256]:
    lst_k = powers_of_two_up_to_n(num_samples)
    # Metricsクラスに属するSampleLevelMetricを指定する場合は .value をつける
    _lst_pass_at_k_metrics = create_passk_metrics(base_metric=Metrics.latex_gold_metric.value, k_values=lst_k, num_samples=num_samples)
    _lst_maj_at_k_metrics = create_majk_metrics(base_metric=Metrics.latex_gold_metric.value, k_values=lst_k, num_samples=num_samples)
    
    task_config = deepcopy(math_500_swallow)
    task_config.name = f"math_500_N{num_samples}"
    task_config.metric = _lst_pass_at_k_metrics + _lst_maj_at_k_metrics
    lst_math_500_swallow_passk_majk.append(task_config)