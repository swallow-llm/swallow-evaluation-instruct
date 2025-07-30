from lighteval.metrics.dynamic_metrics import (
    IndicesExtractionConfig,
    multilingual_extractive_match_metric,
)
from lighteval.tasks.lighteval_task import LightevalTaskConfig
from lighteval.tasks.requests import Doc
from lighteval.utils.language import Language


# SWALLOW JMMLU SUBSETS consist of all JMMLU subsets
# excluding ones with CC-BY-NC-ND license: japanese_civics, japanese_idiom, and japanese_geography.
SWALLOW_JMMLU_SUBSETS = [
    "japanese_history",
    "miscellaneous",
    "security_studies",
    "virology",
    "nutrition",
    "human_sexuality",
    "college_mathematics",
    "econometrics",
    "computer_security",
    "clinical_knowledge",
    "machine_learning",
    "high_school_chemistry",
    "human_aging",
    "logical_fallacies",
    "sociology",
    "high_school_european_history",
    "high_school_statistics",
    "high_school_physics",
    "high_school_microeconomics",
    "college_physics",
    "anatomy",
    "high_school_psychology",
    "business_ethics",
    "professional_psychology",
    "college_medicine",
    "elementary_mathematics",
    "moral_disputes",
    "marketing",
    "high_school_macroeconomics",
    "world_religions",
    "conceptual_physics",
    "professional_medicine",
    "prehistory",
    "high_school_mathematics",
    "international_law",
    "philosophy",
    "management",
    "high_school_computer_science",
    "medical_genetics",
    "college_computer_science",
    "public_relations",
    "professional_accounting",
    "abstract_algebra",
    "global_facts",
    "college_biology",
    "high_school_geography",
    "world_history",
    "high_school_biology",
    "college_chemistry",
    "electrical_engineering",
    "astronomy",
    "jurisprudence",
    "formal_logic",
]


# Query template
SWALLOW_JMMLU_QUERY_TEMPLATE = """
次の選択問題に答えてください。出力の最後の行には「回答: $選択肢」（鉤括弧は書かない）という形でA、B、C、Dから選んだ選択肢を答えてください。ステップバイステップで考えてから回答してください。

{Question}

A) {A}
B) {B}
C) {C}
D) {D}
""".strip()


# Prompt function
def swallow_jmmlu_prompt_fn(line, task_name: str = None):
    answer_letter = line["answer"]
    gold_index = "A,B,C,D".split(",").index(answer_letter)
    choices = [line["A"], line["B"], line["C"], line["D"]]
    query = SWALLOW_JMMLU_QUERY_TEMPLATE.format(
        A=choices[0], B=choices[1], C=choices[2], D=choices[3], Question=line["question"]
    )
    return Doc(
        task_name=task_name,
        query=query,
        choices=["A", "B", "C", "D"],
        gold_index=gold_index,
        instruction=None,
    )


# Metric
multi_choice_metric = multilingual_extractive_match_metric(
    language=Language.JAPANESE,
    gold_extraction_target=[IndicesExtractionConfig(prefix_for_extraction="NativeLetters")],
    pred_extraction_target=[IndicesExtractionConfig(prefix_for_extraction="NativeLetters")],
    precision=5,
)


# Task table
JMMLU_SUBSET_TASKS = [
    LightevalTaskConfig(
        name=f"swallow_jmmlu:{subset}",
        prompt_function=swallow_jmmlu_prompt_fn,
        suite=["swallow"],
        hf_repo="nlp-waseda/JMMLU",
        hf_subset=subset,
        evaluation_splits=["test"],
        hf_avail_splits=["test"],
        trust_dataset=True,
        stop_sequence=[],
        metric=[multi_choice_metric],
        version=0,
    )
    for subset in SWALLOW_JMMLU_SUBSETS
]
