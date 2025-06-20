from lighteval.tasks.lighteval_task import LightevalTaskConfig
import lighteval.tasks.default_prompts as prompt
from lighteval.metrics.metrics import Metrics

gpqa_diamond_instruct_swallow = LightevalTaskConfig(
    name="gpqa:diamond",
    suite=["swallow"],
    prompt_function=prompt.gpqa_instruct,
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
