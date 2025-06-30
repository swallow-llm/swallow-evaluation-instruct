from lighteval.metrics.dynamic_metrics import (
    IndicesExtractionConfig,
    multilingual_extractive_match_metric,
)
from lighteval.tasks.lighteval_task import LightevalTaskConfig
from lighteval.tasks.requests import Doc
from lighteval.tasks.default_prompts import LETTER_INDICES
from lighteval.utils.language import Language


# SWALLOW MMLU ENGLISH SUBSETS
CATEGORIES = {
    "STEM": {
        "physics": ["astronomy", "college_physics", "conceptual_physics", "high_school_physics"],
        "chemistry": ["college_chemistry", "high_school_chemistry"],
        "biology": ["college_biology", "high_school_biology"],
        "computer_science": [
            "college_computer_science",
            "computer_security",
            "high_school_computer_science",
            "machine_learning",
        ],
        "math": [
            "abstract_algebra",
            "college_mathematics",
            "elementary_mathematics",
            "high_school_mathematics",
            "high_school_statistics",
        ],
        "engineering": ["electrical_engineering"],
    },
    "humanities": {
        "history": [
            "high_school_european_history",
            "high_school_us_history",
            "high_school_world_history",
            "prehistory",
        ],
        "philosophy": [
            "formal_logic",
            "logical_fallacies",
            "moral_disputes",
            "moral_scenarios",
            "philosophy",
            "world_religions",
        ],
        "law": ["international_law", "jurisprudence", "professional_law"],
    },
    "social_sciences": {
        "politics": [
            "high_school_government_and_politics",
            "public_relations",
            "security_studies",
            "us_foreign_policy",
        ],
        "culture": ["human_sexuality", "sociology"],
        "economics": ["econometrics", "high_school_macroeconomics", "high_school_microeconomics"],
        "geography": ["high_school_geography"],
        "psychology": ["high_school_psychology", "professional_psychology"],
    },
    "other": {
        "other": ["global_facts", "miscellaneous", "professional_accounting"],
        "business": ["business_ethics", "management", "marketing"],
        "health": [
            "anatomy",
            "clinical_knowledge",
            "college_medicine",
            "human_aging",
            "medical_genetics",
            "nutrition",
            "professional_medicine",
            "virology",
        ],
    },
}

# Query template
SWALLOW_MMLU_ENGLISH_QUERY_TEMPLATE = """
Please answer the following multiple-choice question.
At the end of your response, write your answer in the format: Answer: $choice (without quotation marks), selecting one option from {choices}. Think step by step before giving your final answer.

{Question}

{choice_contents}
""".strip()


# Prompt function
def swallow_mmlu_english_prompt_fn(line, task_name: str = None):
    gold_index = line["answer"]
    choice_contents = line["choices"]
    choices = LETTER_INDICES[: len(choice_contents)]
    query = SWALLOW_MMLU_ENGLISH_QUERY_TEMPLATE.format(
        choices=", ".join(choices),
        Question=line["question"],
        choice_contents="\n".join([f"{letter}) {content}" for letter, content in zip(choices, choice_contents)]),
    )
    return Doc(
        task_name=task_name,
        query=query,
        choices=choices,
        gold_index=gold_index,
        instruction=query,
    )


# Metric
multi_choice_metric = multilingual_extractive_match_metric(
    language=Language.ENGLISH,
    gold_extraction_target=[IndicesExtractionConfig(prefix_for_extraction="NativeLetters")],
    pred_extraction_target=[IndicesExtractionConfig(prefix_for_extraction="NativeLetters")],
    precision=5,
)


# Task table
MMLU_ENGLISH_SUBSETS = []
for major_category in CATEGORIES:
    for middle_category in CATEGORIES[major_category]:
        MMLU_ENGLISH_SUBSETS += CATEGORIES[major_category][middle_category]

mmlu_english_tasks = [
    LightevalTaskConfig(
        name=f"mmlu_english:{subset}",
        prompt_function=swallow_mmlu_english_prompt_fn,
        suite=["swallow"],
        hf_repo="lighteval/mmlu",
        hf_subset=subset,
        evaluation_splits=["test"],
        hf_avail_splits=["test"],
        trust_dataset=True,
        stop_sequence=[],
        metric=[multi_choice_metric],
        version=0,
    )
    for subset in MMLU_ENGLISH_SUBSETS
]
