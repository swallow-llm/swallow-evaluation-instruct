from lighteval.tasks.lighteval_task import LightevalTaskConfig
import lighteval.tasks.default_prompts as prompt
from lighteval.metrics.metrics import Metrics
from lighteval.tasks.swallow.utils import remove_instruction_decorator
from lighteval.metrics.sample_metric_utils import create_passk_metrics, create_majk_metrics, powers_of_two_up_to_n
from copy import deepcopy

# instruciton=None にしたバージョン
@remove_instruction_decorator
def gpqa_instruct_without_instruction(line, task_name: str = None):
    return prompt.gpqa_instruct(line, task_name)

gpqa_diamond_instruct_swallow = LightevalTaskConfig(
    name="gpqa:diamond",
    suite=["swallow"],
    prompt_function=gpqa_instruct_without_instruction,
    hf_repo="Idavidrein/gpqa",
    hf_subset="gpqa_diamond",
    hf_avail_splits=["train"],
    evaluation_splits=["train"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=None,  # swallow用に変更
    metric=[Metrics.gpqa_instruct_metric],
    stop_sequence=[],  # no stop sequence, will use eos token
    trust_dataset=True,
    version=0,
)

# Pass@K and Maj@K variant
lst_gpqa_diamond_instruct_swallow_passk_majk = []
for num_samples in [16, 32, 64, 128, 256]:
    lst_k = powers_of_two_up_to_n(num_samples)
    # Metricsクラスに属するSampleLevelMetricを指定する場合は .value をつける
    _lst_pass_at_k_metrics = create_passk_metrics(base_metric=Metrics.gpqa_instruct_metric.value, k_values=lst_k, num_samples=num_samples)
    _lst_maj_at_k_metrics = create_majk_metrics(base_metric=Metrics.gpqa_instruct_metric.value, k_values=lst_k, num_samples=num_samples)

    task_config = deepcopy(gpqa_diamond_instruct_swallow)
    task_config.name = f"gpqa_N{num_samples}:diamond"
    task_config.metric = _lst_pass_at_k_metrics + _lst_maj_at_k_metrics
    lst_gpqa_diamond_instruct_swallow_passk_majk.append(task_config)