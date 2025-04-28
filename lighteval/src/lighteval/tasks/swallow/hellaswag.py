from lighteval.tasks.lighteval_task import LightevalTaskConfig
from lighteval.tasks.requests import Doc
from lighteval.utils.language import Language

from lighteval.metrics.dynamic_metrics import IndicesExtractionConfig, multilingual_extractive_match_metric


# Prompt template from simple-evals: https://github.com/openai/simple-evals/blob/83ed7640a7d9cd26849bcb3340125002ef14abbe/common.py#L14
MULTI_CHOICE_QUERY_TEMPLATE = """
Answer the following multiple choice question. The last line of your response should be of the following format: 'Answer: $LETTER' (without quotes) where LETTER is one of ABCD. Think step by step before answering.

{Question}

A) {A}
B) {B}
C) {C}
D) {D}
""".strip()

# Extractive multiple choice accuracy metric
multi_choice_metric = multilingual_extractive_match_metric(
    language=Language.ENGLISH,
    gold_extraction_target=[IndicesExtractionConfig(prefix_for_extraction="NativeLetters")],
    pred_extraction_target=[IndicesExtractionConfig(prefix_for_extraction="NativeLetters")],
    precision=5,
)

def hellaswag_prompt_fn(line, task_name: str = None):
    question = f"{line['activity_label']}: {line['ctx_a']} {line['ctx_b'].capitalize()}"
    choices = line["endings"]
    assert len(choices) == 4, "length != 4."
    gold_index = int(line["label"])
    query = MULTI_CHOICE_QUERY_TEMPLATE.format(
        A=choices[0], B=choices[1], C=choices[2], D=choices[3], Question=question
    )
    return Doc(
        task_name=task_name,
        query=query,
        choices=["A", "B", "C", "D"],
        gold_index=gold_index,
        instruction=query,
    )
    
hellaswag_generative = LightevalTaskConfig(
    name="hellaswag",
    suite=["swallow"],
    prompt_function=hellaswag_prompt_fn,
    hf_repo="hellaswag",
    hf_subset="default",
    hf_avail_splits=["train", "test", "validation"],
    evaluation_splits=["validation"],
    few_shots_split=None,
    few_shots_select=None,
    metric=[multi_choice_metric],
    stop_sequence=[],
    trust_dataset=True,
    version=0,
)