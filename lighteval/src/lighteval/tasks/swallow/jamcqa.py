import random

from lighteval.tasks.requests import Doc
from lighteval.tasks.default_prompts import LETTER_INDICES
from lighteval.metrics.dynamic_metrics import (
    IndicesExtractionConfig,
    multilingual_extractive_match_metric,
)
from lighteval.utils.language import Language
from lighteval.tasks.lighteval_task import LightevalTaskConfig


# Prompt function
def swallow_jamcqa_prompt_fn(line, task_name: str = None):
    JamC_QA_QUERY_TEMPLATE = """
次の選択問題に答えてください。出力の最後の行には「回答: $選択肢」（鉤括弧は書かない）という形でA、B、C、Dから選んだ選択肢を答えてください。

{Question}

A) {A}
B) {B}
C) {C}
D) {D}
""".strip()
    correct_idx = int(line["answer_index"])
    all_choices = [
        line["choice0"],
        line["choice1"],
        line["choice2"],
        line["choice3"],
    ]
    
    correct_answer = all_choices[correct_idx]
    incorrect_answers = [c for i, c in enumerate(all_choices) if i != correct_idx]

    line["Correct Answer"] = correct_answer
    line["Incorrect Answer 1"] = incorrect_answers[0]
    line["Incorrect Answer 2"] = incorrect_answers[1]
    line["Incorrect Answer 3"] = incorrect_answers[2]

    gold_index = random.randint(0, 3)
    choices = [line["Incorrect Answer 1"], line["Incorrect Answer 2"], line["Incorrect Answer 3"]]
    choices.insert(gold_index, line["Correct Answer"])

    query = JamC_QA_QUERY_TEMPLATE.format(
        A=choices[0], B=choices[1], C=choices[2], D=choices[3], Question=line["question"]
    )

    return Doc(
        task_name=task_name,
        query=query,
        choices=LETTER_INDICES[: len(choices)],
        gold_index=gold_index,
        instruction=None
    )


def swallow_jamcqa_cot_prompt_fn(line, task_name: str = None):
    JamC_QA_QUERY_TEMPLATE = """
次の選択問題に答えてください。出力の最後の行には「回答: $選択肢」（鉤括弧は書かない）という形でA、B、C、Dから選んだ選択肢を答えてください。ステップバイステップで考えてから回答してください。

{Question}

A) {A}
B) {B}
C) {C}
D) {D}
""".strip()

    correct_idx = int(line["answer_index"])
    all_choices = [
        line["choice0"],
        line["choice1"],
        line["choice2"],
        line["choice3"],
    ]
    
    correct_answer = all_choices[correct_idx]
    incorrect_answers = [c for i, c in enumerate(all_choices) if i != correct_idx]

    line["Correct Answer"] = correct_answer
    line["Incorrect Answer 1"] = incorrect_answers[0]
    line["Incorrect Answer 2"] = incorrect_answers[1]
    line["Incorrect Answer 3"] = incorrect_answers[2]
    gold_index = random.randint(0, 3)
    choices = [line["Incorrect Answer 1"], line["Incorrect Answer 2"], line["Incorrect Answer 3"]]
    choices.insert(gold_index, line["Correct Answer"])

    query = JamC_QA_QUERY_TEMPLATE.format(
        A=choices[0], B=choices[1], C=choices[2], D=choices[3], Question=line["question"]
    )

    return Doc(
        task_name=task_name,
        query=query,
        choices=LETTER_INDICES[: len(choices)],
        gold_index=gold_index,
        instruction=None
    )


jamcqa_instruct_metric = multilingual_extractive_match_metric(
    language=Language.JAPANESE,
    gold_extraction_target=[IndicesExtractionConfig(prefix_for_extraction="NativeLetters")],
    pred_extraction_target=[IndicesExtractionConfig(prefix_for_extraction="NativeLetters")],
    precision=6,
)

jamcqa_task = LightevalTaskConfig(
    name="jamcqa",
    suite=["swallow"],
    prompt_function=swallow_jamcqa_prompt_fn,
    hf_repo="sbintuitions/JamC-QA",
    hf_subset=None,
    hf_avail_splits=["test"],
    evaluation_splits=["test"],
    few_shots_split=None,
    few_shots_select=None,
    metric=[jamcqa_instruct_metric],
    stop_sequence=[],  # no stop sequence, will use eos token
    trust_dataset=True,
    version=0,
)

jamcqa_cot_task = LightevalTaskConfig(
    name="jamcqa_cot",
    suite=["swallow"],
    prompt_function=swallow_jamcqa_cot_prompt_fn,
    hf_repo="sbintuitions/JamC-QA",
    hf_subset=None,
    hf_avail_splits=["test"],
    evaluation_splits=["test"],
    few_shots_split=None,
    few_shots_select=None,
    metric=[jamcqa_instruct_metric],
    stop_sequence=[],  # no stop sequence, will use eos token
    trust_dataset=True,
    version=0,
)




