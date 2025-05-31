from lighteval.metrics.dynamic_metrics import (
    IndicesExtractionConfig,
    multilingual_extractive_match_metric,
)
from lighteval.tasks.lighteval_task import LightevalTaskConfig
from lighteval.tasks.requests import Doc
from lighteval.tasks.default_prompts import LETTER_INDICES
from lighteval.utils.language import Language


# SWALLOW MMLU-ProX Japanese SUBSETS
CATEGORY2SUBSETS = {
    "business": [
        "ori_mmlu_marketing",
        "stemez_Business",
        "ori_mmlu_management",
        "theoremQA_Finance",
        "ori_mmlu_business_ethics"
    ],
    "law": [
        "ori_mmlu_jurisprudence",
        "ori_mmlu_international_law",
        "ori_mmlu_professional_law"
    ],
    "psychology": [
        "stemez_Psychology",
        "ori_mmlu_professional_psychology",
        "ori_mmlu_high_school_psychology"
    ],
    "biology": [
        "ori_mmlu_high_school_biology",
        "ori_mmlu_college_biology",
        "stemez_Genetics",
        "stemez_Biology"
    ],
    "chemistry": [
        "stemez_OrganicChemistry",
        "stemez_PhysicalChemistry",
        "scibench_atkins",
        "scibench_chemmc",
        "scibench_matter",
        "stemez_Chemistry",
        "scibench_quan",
        "ori_mmlu_college_chemistry",
        "ori_mmlu_high_school_chemistry"
    ],
    "history": [
        "ori_mmlu_prehistory",
        "ori_mmlu_high_school_us_history",
        "ori_mmlu_high_school_european_history",
        "ori_mmlu_high_school_world_history"
    ],
    "other": [
        "ori_mmlu_miscellaneous",
        "ori_mmlu_global_facts",
        "ori_mmlu_professional_accounting",
        "ori_mmlu_high_school_government_and_politics",
        "ori_mmlu_human_sexuality",
        "ori_mmlu_high_school_geography",
        "ori_mmlu_public_relations",
        "ori_mmlu_sociology",
        "ori_mmlu_security_studies",
        "ori_mmlu_us_foreign_policy"
    ],
    "health": [
        "ori_mmlu_human_aging",
        "ori_mmlu_professional_medicine",
        "ori_mmlu_virology",
        "ori_mmlu_anatomy",
        "ori_mmlu_college_medicine",
        "ori_mmlu_medical_genetics",
        "ori_mmlu_clinical_knowledge",
        "ori_mmlu_nutrition"
    ],
    "economics": [
        "ori_mmlu_high_school_microeconomics",
        "stemez_Economics",
        "ori_mmlu_high_school_macroeconomics",
        "ori_mmlu_econometrics"
    ],
    "math": [
        "ori_mmlu_elementary_mathematics",
        "ori_mmlu_high_school_mathematics",
        "theoremQA_Math",
        "ori_mmlu_abstract_algebra",
        "ori_mmlu_high_school_statistics",
        "scibench_stat",
        "scibench_diff",
        "scibench_calculus",
        "ori_mmlu_college_mathematics"
    ],
    "physics": [
        "theoremQA_Physics",
        "stemez_Mechanics",
        "stemez_Physics",
        "ori_mmlu_high_school_physics",
        "ori_mmlu_astronomy",
        "stemez_Optics",
        "ori_mmlu_college_physics",
        "scibench_class",
        "ori_mmlu_conceptual_physics",
        "scibench_thermo",
        "scibench_fund"
    ],
    "computer science": [
        "stemez_ComputerScience",
        "ori_mmlu_high_school_computer_science",
        "ori_mmlu_computer_security",
        "theoremQA_EECS",
        "ori_mmlu_machine_learning",
        "ori_mmlu_college_computer_science"
    ],
    "philosophy": [
        "ori_mmlu_world_religions",
        "ori_mmlu_formal_logic",
        "ori_mmlu_logical_fallacies",
        "ori_mmlu_philosophy",
        "ori_mmlu_moral_disputes"
    ],
    "engineering": [
        "ori_mmlu_electrical_engineering",
        "stemez_Thermodynamics",
        "stemez_HeatTransfer",
        "stemez_FluidMechanics",
        "stemez_Electromagnetics",
        "stemez_ElectricalMachines",
        "stemez_ElectricCircuits",
        "stemez_MachineDesign",
        "stemez_TransportPhenomena",
        "stemez_ElectronicCommunications"
    ]
}

# Query template
SWALLOW_MMLU_PROX_JAPANESE_QUERY_TEMPLATE = """
次の選択問題に答えてください。出力の最後の行には「回答: $選択肢」（鉤括弧は書かない）という形で{choices}から1つ選んで答えてください。ステップバイステップで考えてから回答してください。

{Question}

{choice_contents}
""".strip()


# Prompt function
def swallow_mmlu_prox_japanese_prompt_fn(line, task_name: str = None):
    gold_index = line["answer_index"]
    choice_contents = []
    for i in range(10):
        if line[f"option_{i}"] is not None:
            choice_contents.append(line[f"option_{i}"])
    choices = LETTER_INDICES[:len(choice_contents)]
    query = SWALLOW_MMLU_PROX_JAPANESE_QUERY_TEMPLATE.format(
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
        instruction=query,
    )


# Metric
multi_choice_metric = multilingual_extractive_match_metric(
    language=Language.JAPANESE,
    gold_extraction_target=[IndicesExtractionConfig(prefix_for_extraction="NativeLetters")],
    pred_extraction_target=[IndicesExtractionConfig(prefix_for_extraction="NativeLetters")],
    precision=5,
)


# Task table
MMLU_PROX_JAPANESE_SUBSETS = []
for _, subsets in CATEGORY2SUBSETS.items():
    MMLU_PROX_JAPANESE_SUBSETS += subsets

mmlu_prox_japanese_tasks = [
    LightevalTaskConfig(
        name=f"mmlu_prox_japanese:{subset}",
        prompt_function=swallow_mmlu_prox_japanese_prompt_fn,
        suite=["swallow"],
        hf_repo="tokyotech-llm/MMLU-ProX-Japanese",
        hf_subset=subset,
        evaluation_splits=["test"],
        hf_avail_splits=["test"],
        trust_dataset=True,
        stop_sequence=[],
        metric=[multi_choice_metric],
        version=0,
    )
    for subset in MMLU_PROX_JAPANESE_SUBSETS
]
