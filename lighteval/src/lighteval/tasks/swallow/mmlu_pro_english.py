from lighteval.metrics.dynamic_metrics import (
    IndicesExtractionConfig,
    multilingual_extractive_match_metric,
)
from lighteval.tasks.lighteval_task import LightevalTaskConfig
from lighteval.tasks.requests import Doc
from lighteval.tasks.default_prompts import LETTER_INDICES
from lighteval.utils.language import Language


# SWALLOW MMLU-Pro ENGLISH SUBSETS
CATEGORY2SUBSETS = {
    "business": [
        "ori_mmlu_business_ethics",
        "ori_mmlu_marketing",
        "ori_mmlu_management",
        "stemez_Business",
        "theoremQA_Finance"
    ],
    "law": [
        "ori_mmlu_international_law",
        "ori_mmlu_professional_law",
        "ori_mmlu_jurisprudence"
    ],
    "psychology": [
        "ori_mmlu_professional_psychology",
        "ori_mmlu_high_school_psychology",
        "stemez_Psychology"
    ],
    "biology": [
        "ori_mmlu_high_school_biology",
        "ori_mmlu_college_biology",
        "stemez_Biology",
        "stemez_Genetics"
    ],
    "chemistry": [
        "scibench_matter",
        "ori_mmlu_high_school_chemistry",
        "scibench_quan",
        "stemez_OrganicChemistry",
        "stemez_PhysicalChemistry",
        "scibench_chemmc",
        "stemez_Chemistry",
        "scibench_atkins",
        "ori_mmlu_college_chemistry"
    ],
    "history": [
        "ori_mmlu_prehistory",
        "ori_mmlu_high_school_us_history",
        "ori_mmlu_high_school_european_history",
        "ori_mmlu_high_school_world_history"
    ],
    "other": [
        "ori_mmlu_security_studies",
        "ori_mmlu_high_school_government_and_politics",
        "ori_mmlu_human_sexuality",
        "ori_mmlu_high_school_geography",
        "ori_mmlu_us_foreign_policy",
        "ori_mmlu_sociology",
        "ori_mmlu_miscellaneous",
        "ori_mmlu_public_relations",
        "ori_mmlu_professional_accounting",
        "ori_mmlu_global_facts"
    ],
    "health": [
        "ori_mmlu_virology",
        "ori_mmlu_college_medicine",
        "ori_mmlu_clinical_knowledge",
        "ori_mmlu_human_aging",
        "ori_mmlu_anatomy",
        "ori_mmlu_nutrition",
        "ori_mmlu_medical_genetics",
        "ori_mmlu_professional_medicine"
    ],
    "economics": [
        "ori_mmlu_econometrics",
        "ori_mmlu_high_school_macroeconomics",
        "stemez_Economics",
        "ori_mmlu_high_school_microeconomics"
    ],
    "math": [
        "scibench_diff",
        "scibench_calculus",
        "ori_mmlu_high_school_mathematics",
        "ori_mmlu_high_school_statistics",
        "ori_mmlu_college_mathematics",
        "ori_mmlu_elementary_mathematics",
        "scibench_stat",
        "ori_mmlu_abstract_algebra",
        "theoremQA_Math"
    ],
    "physics": [
        "theoremQA_Physics",
        "stemez_Optics",
        "stemez_Mechanics",
        "scibench_class",
        "ori_mmlu_astronomy",
        "stemez_Physics",
        "ori_mmlu_high_school_physics",
        "ori_mmlu_college_physics",
        "ori_mmlu_conceptual_physics",
        "scibench_fund",
        "scibench_thermo"
    ],
    "computer science": [
        "theoremQA_EECS",
        "ori_mmlu_college_computer_science",
        "ori_mmlu_high_school_computer_science",
        "ori_mmlu_computer_security",
        "stemez_ComputerScience",
        "ori_mmlu_machine_learning"
    ],
    "philosophy": [
        "ori_mmlu_formal_logic",
        "ori_mmlu_moral_disputes",
        "ori_mmlu_world_religions",
        "ori_mmlu_logical_fallacies",
        "ori_mmlu_philosophy"
    ],
    "engineering": [
        "stemez_Thermodynamics",
        "stemez_Electromagnetics",
        "stemez_FluidMechanics",
        "stemez_MachineDesign",
        "stemez_ElectronicCommunications",
        "stemez_HeatTransfer",
        "ori_mmlu_electrical_engineering",
        "stemez_ElectricalMachines",
        "stemez_TransportPhenomena",
        "stemez_ElectricCircuits"
    ]
}

# Query template
SWALLOW_MMLU_PRO_ENGLISH_QUERY_TEMPLATE = """
Please answer the following multiple-choice question.
At the end of your response, write your answer in the format: Answer: $choice (without quotation marks), selecting one option from {choices}. Think step by step before giving your final answer.

{Question}

{choice_contents}
""".strip()


# Prompt function
def swallow_mmlu_pro_english_prompt_fn(line, task_name: str = None):
    gold_index = line["answer_index"]
    choice_contents = line["options"]
    choices = LETTER_INDICES[:len(choice_contents)]
    query = SWALLOW_MMLU_PRO_ENGLISH_QUERY_TEMPLATE.format(
        choices=", ".join(choices),
        Question=line["question"],
        choice_contents="\n".join(
            [f"{letter}) {content}" for letter, content in zip(choices, choice_contents)]
        )
    )
    return Doc(
        task_name=task_name,
        query=query,
        choices=choices,
        gold_index=gold_index,
        instruction=None,
    )


# Metric
multi_choice_metric = multilingual_extractive_match_metric(
    language=Language.ENGLISH,
    gold_extraction_target=[IndicesExtractionConfig(prefix_for_extraction="NativeLetters")],
    pred_extraction_target=[IndicesExtractionConfig(prefix_for_extraction="NativeLetters")],
    precision=5,
)


# Task table
MMLU_PRO_ENGLISH_SUBSETS = []
for _, subsets in CATEGORY2SUBSETS.items():
    MMLU_PRO_ENGLISH_SUBSETS += subsets

mmlu_pro_english_tasks = [
    LightevalTaskConfig(
        name=f"mmlu_pro_english:{subset}",
        prompt_function=swallow_mmlu_pro_english_prompt_fn,
        suite=["swallow"],
        hf_repo="tokyotech-llm/MMLU-Pro-English",
        hf_subset=subset,
        evaluation_splits=["test"],
        hf_avail_splits=["test"],
        trust_dataset=True,
        stop_sequence=[],
        metric=[multi_choice_metric],
        version=0,
    )
    for subset in MMLU_PRO_ENGLISH_SUBSETS
]
