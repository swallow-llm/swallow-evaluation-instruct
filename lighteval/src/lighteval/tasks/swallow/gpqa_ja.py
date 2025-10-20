import random

from lighteval.tasks.requests import Doc
from lighteval.tasks.default_prompts import LETTER_INDICES
from lighteval.metrics.dynamic_metrics import (
    IndicesExtractionConfig,
    multilingual_extractive_match_metric,
)
from lighteval.utils.language import Language
from lighteval.tasks.lighteval_task import LightevalTaskConfig
from lighteval.metrics.sample_metric_utils import create_passk_metrics, create_majk_metrics, powers_of_two_up_to_n
from copy import deepcopy


# Prompt function
def swallow_gpqa_ja_prompt_fn(line, task_name: str = None):
    GPQA_JA_QUERY_TEMPLATE = """
次の選択問題に答えてください。出力の最後の行には「回答: $選択肢」（鉤括弧は書かない）という形でA、B、C、Dから選んだ選択肢を答えてください。ステップバイステップで考えてから回答してください。

{Question}

A) {A}
B) {B}
C) {C}
D) {D}
""".strip()
    gold_index = random.randint(0, 3)
    choices = [line["Incorrect Answer 1"], line["Incorrect Answer 2"], line["Incorrect Answer 3"]]
    choices.insert(gold_index, line["Correct Answer"])

    query = GPQA_JA_QUERY_TEMPLATE.format(
        A=choices[0], B=choices[1], C=choices[2], D=choices[3], Question=line["Question"]
    )

    return Doc(
        task_name=task_name,
        query=query,
        choices=LETTER_INDICES[: len(choices)],
        gold_index=gold_index,
        instruction=None
    )


gpqa_ja_instruct_metric = multilingual_extractive_match_metric(
    language=Language.JAPANESE,
    gold_extraction_target=[IndicesExtractionConfig(prefix_for_extraction="NativeLetters")],
    pred_extraction_target=[IndicesExtractionConfig(prefix_for_extraction="NativeLetters")],
    precision=6,
)

gpqa_ja_instruct_lighteval = LightevalTaskConfig(
    name="swallow_gpqa_ja",
    suite=["swallow"],
    prompt_function=swallow_gpqa_ja_prompt_fn,
    hf_repo="LLaMAX/BenchMAX_Science",
    hf_subset="ja",
    hf_avail_splits=["train"],
    evaluation_splits=["train"],
    few_shots_split=None,
    few_shots_select=None,
    metric=[gpqa_ja_instruct_metric],
    stop_sequence=[],  # no stop sequence, will use eos token
    trust_dataset=True,
    version=0,
)

# Pass@K and Maj@K variant
lst_gpqa_ja_instruct_passk_majk = []
for num_samples in [16, 32, 64, 128, 256]:
    lst_k = powers_of_two_up_to_n(num_samples)
    _lst_pass_at_k_metrics = create_passk_metrics(base_metric=gpqa_ja_instruct_metric, k_values=lst_k, num_samples=num_samples)
    _lst_maj_at_k_metrics = create_majk_metrics(base_metric=gpqa_ja_instruct_metric, k_values=lst_k, num_samples=num_samples)

    task_config = deepcopy(gpqa_ja_instruct_lighteval)
    task_config.name = f"swallow_gpqa_ja_N{num_samples}"
    task_config.metric = _lst_pass_at_k_metrics + _lst_maj_at_k_metrics
    lst_gpqa_ja_instruct_passk_majk.append(task_config)