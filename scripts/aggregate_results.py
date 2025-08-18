import argparse
import os
import json
import glob
from datetime import datetime
from functools import wraps
from typing import Any

## lighteval のメトリクスから欲しいメトリクスへ整形する関数 #########################################################

def resolve_multi_task_key(func):
    """
    target["task_key"] が "key1,key2,..." のように複数書かれている場合に
    ・latest_results で見つかったキーが 0 個なら -1 を返す
    ・1 個ならそのキーだけを target にセットして元関数へ委譲
    ・2 個以上見つかったら ValueError を送出
    単一キーの場合は何もしないでそのまま元関数へ委譲する
    """
    @wraps(func)
    def wrapper(latest_results: dict, target: dict, *args, **kwargs):
        raw_key = target.get("task_key", "")
        # ① task_key が空／単一ならそのまま
        if (not raw_key) or ("," not in raw_key):
            return func(latest_results, target, *args, **kwargs)

        # ② 複数キーを個別に試す
        keys = [k.strip() for k in raw_key.split(",") if k.strip()]
        found_keys = []
        result_for_found = None

        for k in keys:
            ## keys から一つ選んで task_key としてセットし，元関数を走らせる 
            ## 見つからないと -1 が返る仕様を活用して，見つかったキーとその計算結果を保存しておく
            tmp_target = dict(target)
            tmp_target["task_key"] = k
            r = func(latest_results, tmp_target, *args, **kwargs)
            if r != -1:
                found_keys.append(k)
                result_for_found = r

        # ③ 見つかった結果に応じた処理
        if len(found_keys) == 0:
            ## キーが一つも見つからなかった場合は -1 を返す
            return -1

        elif len(found_keys) == 1:
            ## キーが一つだけ見つかった場合はその結果を採用
            return result_for_found

        else:
            ## キーが複数見つかった場合は ValueError を送出
            raise ValueError(
                f"Multiple task_keys {found_keys} are present in latest_results "
                f"for requested '{raw_key}'. Please disambiguate."
            )

    return wrapper


@resolve_multi_task_key
def pick(latest_results: dict, target: dict, metric_key: str) -> float:
    """
    対象タスクの最新エントリから、指定された metric の値をそのまま返す．
    対象エントリが存在しなければ -1 を返す．
    """
    task_key = target.get("task_key")
    for entry in latest_results.values():
        if entry["task_key"] == task_key:
            return entry["metrics"].get(metric_key, -1)
    print(f"Warning: {task_key} does not have {metric_key}.")
    return -1


@resolve_multi_task_key
def micro_average(latest_results: dict, target: dict, metric_key: str, white_list: list[str]=[]) -> float:
    """
    対象タスクの全サブセットについて、指定された metric のサンプル数重み付き平均を計算する．
    もし対象となるエントリが無ければ -1 を返す．
    なお white_list に特定のサブセット群を渡すことで，サブセットの中でも計算の対象をフィルタリングすることができる．
    """
    task_key = target.get("task_key")
    display_name = target.get("display_name")
    total_sample = 0
    weighted_sum = 0.0
    for entry in latest_results.values():
        # task_key の例: "swallow|swallow_jmmlu:public_relations|0" や "swallow|swallow_jmmlu:abstract_algebra|0"
        parts = entry["task_key"].split("|")
        if len(parts) < 3:
            continue
        # parts[1] を ":" で分割して先頭部分を抽出し，base_key を作る．
        if ":" in parts[1]:
            base_second, subset_name = parts[1].split(":", 1)
            base_key = f"{parts[0]}|{base_second}|{parts[2]}"
        else:
            base_second = parts[1]
            subset_name = ""
            base_key = f"{parts[0]}|{base_second}|{parts[2]}"

        if (base_key == task_key) and ((len(white_list)==0) or (subset_name in white_list)):
            sample_num = entry.get("sample_num", 0)
            metric_value = entry["metrics"].get(metric_key)
            if metric_value is None:
                print(f"Warning: {task_key} does not have {metric_key}.")
                continue
            total_sample += sample_num
            weighted_sum += sample_num * metric_value
    if total_sample > 0:
        return weighted_sum / total_sample
    else:
        return -1


@resolve_multi_task_key
def average_in_one_task(latest_results: dict, target: dict, metric_key_list: list[str]) -> float:
    """
    対象タスクに対して、指定されたmetric間の平均を計算する
    """
    task_key = target.get("task_key")
    result = 0
    count = 0
    for entry in latest_results.values():
        if entry["task_key"] == task_key:
            count += 1
            for metric_key in metric_key_list:
                assert metric_key in entry["metrics"], f"{metric_key} is not in {entry['metrics']}"
                result += entry["metrics"][metric_key]
    if count == 0:
        return -1

    return result / len(metric_key_list)


## まとめて計算するサブセットの定義 #####################################################################################

# JMMLU Subjects
JMMLU_STEM = [
    'abstract_algebra', 'anatomy', 'astronomy', 'college_biology', 'college_chemistry',
    'college_computer_science', 'college_mathematics', 'college_physics',
    'computer_security', 'conceptual_physics', 'electrical_engineering',
    'elementary_mathematics', 'high_school_biology', 'high_school_chemistry',
    'high_school_computer_science', 'high_school_mathematics', 'high_school_physics',
    'high_school_statistics', 'machine_learning'
]

JMMLU_OTHERS = [
    'business_ethics', 'clinical_knowledge', 'college_medicine', 'global_facts',
    'human_aging', 'management', 'marketing', 'medical_genetics', 'miscellaneous',
    'nutrition', 'professional_accounting', 'professional_medicine', 'virology'
]

JMMLU_SOCIAL_SCIENCES = [
    'econometrics', 'high_school_geography', 'high_school_government_and_politics',
    'high_school_macroeconomics', 'high_school_microeconomics',
    'high_school_psychology', 'human_sexuality', 'professional_psychology',
    'public_relations', 'security_studies', 'sociology', 'us_foreign_policy'
]

JMMLU_HUMANITIES = [
    'formal_logic', 'high_school_european_history', 'high_school_us_history',
    'high_school_world_history', 'international_law', 'jurisprudence',
    'logical_fallacies', 'moral_disputes', 'moral_scenarios', 'philosophy',
    'prehistory', 'professional_law', 'world_religions'
]

# MMLU Prox Japanese Subjects
MMLU_PROX_JAPANESE_BUSINESS = [
    "theoremQA_Finance",
    "ori_mmlu_marketing",
    "stemez_Business",
    "ori_mmlu_management",
    "ori_mmlu_business_ethics"
]

MMLU_PROX_JAPANESE_LAW = [
    "ori_mmlu_professional_law",
    "ori_mmlu_international_law",
    "ori_mmlu_jurisprudence"
]

MMLU_PROX_JAPANESE_PSYCHOLOGY = [
    "stemez_Psychology",
    "ori_mmlu_high_school_psychology",
    "ori_mmlu_professional_psychology"
]

MMLU_PROX_JAPANESE_BIOLOGY = [
    "ori_mmlu_high_school_biology",
    "stemez_Genetics",
    "ori_mmlu_college_biology",
    "stemez_Biology"
]

MMLU_PROX_JAPANESE_CHEMISTRY = [
    "scibench_matter",
    "scibench_atkins",
    "scibench_chemmc",
    "stemez_PhysicalChemistry",
    "ori_mmlu_college_chemistry",
    "ori_mmlu_high_school_chemistry",
    "stemez_OrganicChemistry",
    "stemez_Chemistry",
    "scibench_quan"
]

MMLU_PROX_JAPANESE_HISTORY = [
    "ori_mmlu_high_school_us_history",
    "ori_mmlu_high_school_world_history",
    "ori_mmlu_high_school_european_history",
    "ori_mmlu_prehistory"
]

MMLU_PROX_JAPANESE_OTHER = [
    "ori_mmlu_high_school_government_and_politics",
    "ori_mmlu_professional_accounting",
    "ori_mmlu_security_studies",
    "ori_mmlu_sociology",
    "ori_mmlu_miscellaneous",
    "ori_mmlu_human_sexuality",
    "ori_mmlu_high_school_geography",
    "ori_mmlu_global_facts",
    "ori_mmlu_public_relations",
    "ori_mmlu_us_foreign_policy"
]

MMLU_PROX_JAPANESE_HEALTH = [
    "ori_mmlu_nutrition",
    "ori_mmlu_clinical_knowledge",
    "ori_mmlu_virology",
    "ori_mmlu_medical_genetics",
    "ori_mmlu_professional_medicine",
    "ori_mmlu_anatomy",
    "ori_mmlu_human_aging",
    "ori_mmlu_college_medicine"
]

MMLU_PROX_JAPANESE_ECONOMICS = [
    "ori_mmlu_high_school_macroeconomics",
    "stemez_Economics",
    "ori_mmlu_econometrics",
    "ori_mmlu_high_school_microeconomics"
]

MMLU_PROX_JAPANESE_MATH = [
    "theoremQA_Math",
    "scibench_calculus",
    "ori_mmlu_high_school_mathematics",
    "scibench_stat",
    "ori_mmlu_college_mathematics",
    "ori_mmlu_high_school_statistics",
    "ori_mmlu_elementary_mathematics",
    "scibench_diff",
    "ori_mmlu_abstract_algebra"
]

MMLU_PROX_JAPANESE_PHYSICS = [
    "scibench_class",
    "stemez_Optics",
    "theoremQA_Physics",
    "ori_mmlu_conceptual_physics",
    "stemez_Physics",
    "ori_mmlu_high_school_physics",
    "ori_mmlu_college_physics",
    "scibench_thermo",
    "scibench_fund",
    "ori_mmlu_astronomy",
    "stemez_Mechanics"
]

MMLU_PROX_JAPANESE_COMPUTER_SCIENCE = [
    "ori_mmlu_high_school_computer_science",
    "ori_mmlu_college_computer_science",
    "stemez_ComputerScience",
    "ori_mmlu_machine_learning",
    "ori_mmlu_computer_security",
    "theoremQA_EECS"
]

MMLU_PROX_JAPANESE_PHILOSOPHY = [
    "ori_mmlu_logical_fallacies",
    "ori_mmlu_world_religions",
    "ori_mmlu_formal_logic",
    "ori_mmlu_moral_disputes",
    "ori_mmlu_philosophy"
]

MMLU_PROX_JAPANESE_ENGINEERING = [
    "stemez_Electromagnetics",
    "stemez_FluidMechanics",
    "stemez_HeatTransfer",
    "stemez_Thermodynamics",
    "stemez_TransportPhenomena",
    "ori_mmlu_electrical_engineering",
    "stemez_ElectricalMachines",
    "stemez_ElectronicCommunications",
    "stemez_ElectricCircuits",
    "stemez_MachineDesign"
]

# MMLU Prox English Subjects
MMLU_PROX_ENGLISH_BUSINESS = [
    "theoremQA_Finance",
    "ori_mmlu_marketing",
    "stemez_Business",
    "ori_mmlu_management",
    "ori_mmlu_business_ethics"
]

MMLU_PROX_ENGLISH_LAW = [
    "ori_mmlu_professional_law",
    "ori_mmlu_international_law",
    "ori_mmlu_jurisprudence"
]

MMLU_PROX_ENGLISH_PSYCHOLOGY = [
    "stemez_Psychology",
    "ori_mmlu_high_school_psychology",
    "ori_mmlu_professional_psychology"
]

MMLU_PROX_ENGLISH_BIOLOGY = [
    "ori_mmlu_high_school_biology",
    "stemez_Genetics",
    "ori_mmlu_college_biology",
    "stemez_Biology"
]

MMLU_PROX_ENGLISH_CHEMISTRY = [
    "scibench_matter",
    "scibench_atkins",
    "scibench_chemmc",
    "stemez_PhysicalChemistry",
    "ori_mmlu_college_chemistry",
    "ori_mmlu_high_school_chemistry",
    "stemez_OrganicChemistry",
    "stemez_Chemistry",
    "scibench_quan"
]

MMLU_PROX_ENGLISH_HISTORY = [
    "ori_mmlu_high_school_us_history",
    "ori_mmlu_high_school_world_history",
    "ori_mmlu_high_school_european_history",
    "ori_mmlu_prehistory"
]

MMLU_PROX_ENGLISH_OTHER = [
    "ori_mmlu_high_school_government_and_politics",
    "ori_mmlu_professional_accounting",
    "ori_mmlu_security_studies",
    "ori_mmlu_sociology",
    "ori_mmlu_miscellaneous",
    "ori_mmlu_human_sexuality",
    "ori_mmlu_high_school_geography",
    "ori_mmlu_global_facts",
    "ori_mmlu_public_relations",
    "ori_mmlu_us_foreign_policy"
]

MMLU_PROX_ENGLISH_HEALTH = [
    "ori_mmlu_nutrition",
    "ori_mmlu_clinical_knowledge",
    "ori_mmlu_virology",
    "ori_mmlu_medical_genetics",
    "ori_mmlu_professional_medicine",
    "ori_mmlu_anatomy",
    "ori_mmlu_human_aging",
    "ori_mmlu_college_medicine"
]

MMLU_PROX_ENGLISH_ECONOMICS = [
    "ori_mmlu_high_school_macroeconomics",
    "stemez_Economics",
    "ori_mmlu_econometrics",
    "ori_mmlu_high_school_microeconomics"
]

MMLU_PROX_ENGLISH_MATH = [
    "theoremQA_Math",
    "scibench_calculus",
    "ori_mmlu_high_school_mathematics",
    "scibench_stat",
    "ori_mmlu_college_mathematics",
    "ori_mmlu_high_school_statistics",
    "ori_mmlu_elementary_mathematics",
    "scibench_diff",
    "ori_mmlu_abstract_algebra"
]

MMLU_PROX_ENGLISH_PHYSICS = [
    "scibench_class",
    "stemez_Optics",
    "theoremQA_Physics",
    "ori_mmlu_conceptual_physics",
    "stemez_Physics",
    "ori_mmlu_high_school_physics",
    "ori_mmlu_college_physics",
    "scibench_thermo",
    "scibench_fund",
    "ori_mmlu_astronomy",
    "stemez_Mechanics"
]

MMLU_PROX_ENGLISH_COMPUTER_SCIENCE = [
    "ori_mmlu_high_school_computer_science",
    "ori_mmlu_college_computer_science",
    "stemez_ComputerScience",
    "ori_mmlu_machine_learning",
    "ori_mmlu_computer_security",
    "theoremQA_EECS"
]

MMLU_PROX_ENGLISH_PHILOSOPHY = [
    "ori_mmlu_logical_fallacies",
    "ori_mmlu_world_religions",
    "ori_mmlu_formal_logic",
    "ori_mmlu_moral_disputes",
    "ori_mmlu_philosophy"
]

MMLU_PROX_ENGLISH_ENGINEERING = [
    "stemez_Electromagnetics",
    "stemez_FluidMechanics",
    "stemez_HeatTransfer",
    "stemez_Thermodynamics",
    "stemez_TransportPhenomena",
    "ori_mmlu_electrical_engineering",
    "stemez_ElectricalMachines",
    "stemez_ElectronicCommunications",
    "stemez_ElectricCircuits",
    "stemez_MachineDesign"
]

# MMLU Pro English Subjects
MMLU_PRO_ENGLISH_BUSINESS = [
    "ori_mmlu_business_ethics",
    "ori_mmlu_marketing",
    "ori_mmlu_management",
    "stemez_Business",
    "theoremQA_Finance"
]

MMLU_PRO_ENGLISH_LAW = [
    "ori_mmlu_international_law",
    "ori_mmlu_professional_law",
    "ori_mmlu_jurisprudence"
]

MMLU_PRO_ENGLISH_PSYCHOLOGY = [
    "ori_mmlu_professional_psychology",
    "ori_mmlu_high_school_psychology",
    "stemez_Psychology"
]

MMLU_PRO_ENGLISH_BIOLOGY = [
    "ori_mmlu_high_school_biology",
    "ori_mmlu_college_biology",
    "stemez_Biology",
    "stemez_Genetics"
]

MMLU_PRO_ENGLISH_CHEMISTRY = [
    "scibench_matter",
    "ori_mmlu_high_school_chemistry",
    "scibench_quan",
    "stemez_OrganicChemistry",
    "stemez_PhysicalChemistry",
    "scibench_chemmc",
    "stemez_Chemistry",
    "scibench_atkins",
    "ori_mmlu_college_chemistry"
]

MMLU_PRO_ENGLISH_HISTORY = [
    "ori_mmlu_prehistory",
    "ori_mmlu_high_school_us_history",
    "ori_mmlu_high_school_european_history",
    "ori_mmlu_high_school_world_history"
]

MMLU_PRO_ENGLISH_OTHER = [
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
]

MMLU_PRO_ENGLISH_HEALTH = [
    "ori_mmlu_virology",
    "ori_mmlu_college_medicine",
    "ori_mmlu_clinical_knowledge",
    "ori_mmlu_human_aging",
    "ori_mmlu_anatomy",
    "ori_mmlu_nutrition",
    "ori_mmlu_medical_genetics",
    "ori_mmlu_professional_medicine"
]

MMLU_PRO_ENGLISH_ECONOMICS = [
    "ori_mmlu_econometrics",
    "ori_mmlu_high_school_macroeconomics",
    "stemez_Economics",
    "ori_mmlu_high_school_microeconomics"
]

MMLU_PRO_ENGLISH_MATH = [
    "scibench_diff",
    "scibench_calculus",
    "ori_mmlu_high_school_mathematics",
    "ori_mmlu_high_school_statistics",
    "ori_mmlu_college_mathematics",
    "ori_mmlu_elementary_mathematics",
    "scibench_stat",
    "ori_mmlu_abstract_algebra",
    "theoremQA_Math"
]

MMLU_PRO_ENGLISH_PHYSICS = [
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
]

MMLU_PRO_ENGLISH_COMPUTER_SCIENCE = [
    "theoremQA_EECS",
    "ori_mmlu_college_computer_science",
    "ori_mmlu_high_school_computer_science",
    "ori_mmlu_computer_security",
    "stemez_ComputerScience",
    "ori_mmlu_machine_learning"
]

MMLU_PRO_ENGLISH_PHILOSOPHY = [
    "ori_mmlu_formal_logic",
    "ori_mmlu_moral_disputes",
    "ori_mmlu_world_religions",
    "ori_mmlu_logical_fallacies",
    "ori_mmlu_philosophy"
]

MMLU_PRO_ENGLISH_ENGINEERING = [
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

# MMLU English Subjects
MMLU_ENGLISH_STEM = [
    "astronomy", "anatomy", "college_physics", "conceptual_physics", "high_school_physics",
    "college_chemistry", "high_school_chemistry",
    "college_biology", "high_school_biology",
    "college_computer_science", "computer_security", "high_school_computer_science", "machine_learning",
    "abstract_algebra", "college_mathematics", "elementary_mathematics", "high_school_mathematics", "high_school_statistics",
    "electrical_engineering"
]

MMLU_ENGLISH_HUMANITIES = [
    "high_school_european_history", "high_school_us_history", "high_school_world_history", "prehistory",
    "formal_logic", "logical_fallacies", "moral_disputes", "moral_scenarios", "philosophy", "world_religions",
    "international_law", "jurisprudence", "professional_law"
]

MMLU_ENGLISH_SOCIAL_SCIENCES = [
    "high_school_government_and_politics", "public_relations", "security_studies", "us_foreign_policy",
    "human_sexuality", "sociology",
    "econometrics", "high_school_macroeconomics", "high_school_microeconomics",
    "high_school_geography",
    "high_school_psychology", "professional_psychology"
]

MMLU_ENGLISH_OTHER = [
    "global_facts", "miscellaneous", "professional_accounting",
    "business_ethics", "management", "marketing",
    "clinical_knowledge", "college_medicine", "human_aging", "medical_genetics", "nutrition", "professional_medicine", "virology"
]

M_IFEVAL = [
    "prompt_level_strict_acc",
    "inst_level_strict_acc",
    "prompt_level_loose_acc",
    "inst_level_loose_acc",
]

## 計算するメトリクスの定義 ##############################################################################################

AGGREGATE_CONF = [
    # JMMLU
    {
        'display_name': 'jmmlu_social_sciences', 
        'func': micro_average, 
        'func_args': {
            'metric_key': 'extractive_match', 
            'white_list': JMMLU_SOCIAL_SCIENCES
        }, 
        'target': {
            'task_key': 'swallow|swallow_jmmlu|0'
        },
    },
    {
        'display_name': 'jmmlu_humanities', 
        'func': micro_average, 
        'func_args': {
            'metric_key': 'extractive_match', 
            'white_list': JMMLU_HUMANITIES
        }, 
        'target': {
            'task_key': 'swallow|swallow_jmmlu|0'
        },
    },
    {
        'display_name': 'jmmlu_stem', 
        'func': micro_average, 
        'func_args': {
            'metric_key': 'extractive_match', 
            'white_list': JMMLU_STEM
        }, 
        'target': {
            'task_key': 'swallow|swallow_jmmlu|0'
        },
    },
    {
        'display_name': 'jmmlu_other', 
        'func': micro_average, 
        'func_args': {
            'metric_key': 'extractive_match', 
            'white_list': JMMLU_OTHERS
        }, 
        'target': {
            'task_key': 'swallow|swallow_jmmlu|0'
        },
    },
    {
        'display_name': 'jmmlu', 
        'func': micro_average, 
        'func_args': {
            'metric_key': 'extractive_match'
        }, 
        'target': {
            'task_key': 'swallow|swallow_jmmlu|0'
        },
    },

    # JHumanEval
    {
        'display_name': 'jhumaneval_pass@1', 
        'func': pick, 
        'func_args': {
            'metric_key': 'humaneval_pass@1:10'
        }, 
        'target': {
            'task_key': 'swallow|swallow_jhumaneval|0'
        },
    },
    {
        'display_name': 'jhumaneval_pass@10', 
        'func': pick, 
        'func_args': {
            'metric_key': 'humaneval_pass@10:10'
        }, 
        'target': {
            'task_key': 'swallow|swallow_jhumaneval|0'
        },
    },

    # MMLU Prox Japanese
    {
        "display_name": "mmlu_prox_japanese_business",
        "func": micro_average,
        "func_args": {
            "metric_key": "extractive_match",
            "white_list": MMLU_PROX_JAPANESE_BUSINESS,
        },
        "target": {
            "task_key": "swallow|mmlu_prox_japanese|0",
        },
    },
    {
        "display_name": "mmlu_prox_japanese_law",
        "func": micro_average,
        "func_args": {
            "metric_key": "extractive_match",
            "white_list": MMLU_PROX_JAPANESE_LAW,
        },
        "target": {
            "task_key": "swallow|mmlu_prox_japanese|0",
        },
    },
    {
        "display_name": "mmlu_prox_japanese_psychology",
        "func": micro_average,
        "func_args": {
            "metric_key": "extractive_match",
            "white_list": MMLU_PROX_JAPANESE_PSYCHOLOGY,
        },
        "target": {
            "task_key": "swallow|mmlu_prox_japanese|0",
        },
    },
    {
        "display_name": "mmlu_prox_japanese_biology",
        "func": micro_average,
        "func_args": {
            "metric_key": "extractive_match",
            "white_list": MMLU_PROX_JAPANESE_BIOLOGY,
        },
        "target": {
            "task_key": "swallow|mmlu_prox_japanese|0",
        },
    },
    {
        "display_name": "mmlu_prox_japanese_chemistry",
        "func": micro_average,
        "func_args": {
            "metric_key": "extractive_match",
            "white_list": MMLU_PROX_JAPANESE_CHEMISTRY,
        },
        "target": {
            "task_key": "swallow|mmlu_prox_japanese|0",
        },
    },
    {
        "display_name": "mmlu_prox_japanese_history",
        "func": micro_average,
        "func_args": {
            "metric_key": "extractive_match",
            "white_list": MMLU_PROX_JAPANESE_HISTORY,
        },
        "target": {
            "task_key": "swallow|mmlu_prox_japanese|0",
        },
    },
    {
        "display_name": "mmlu_prox_japanese_other",
        "func": micro_average,
        "func_args": {
            "metric_key": "extractive_match",
            "white_list": MMLU_PROX_JAPANESE_OTHER,
        },
        "target": {
            "task_key": "swallow|mmlu_prox_japanese|0",
        },
    },
    {
        "display_name": "mmlu_prox_japanese_health",
        "func": micro_average,
        "func_args": {
            "metric_key": "extractive_match",
            "white_list": MMLU_PROX_JAPANESE_HEALTH,
        },
        "target": {
            "task_key": "swallow|mmlu_prox_japanese|0",
        },
    },
    {
        "display_name": "mmlu_prox_japanese_economics",
        "func": micro_average,
        "func_args": {
            "metric_key": "extractive_match",
            "white_list": MMLU_PROX_JAPANESE_ECONOMICS,
        },
        "target": {
            "task_key": "swallow|mmlu_prox_japanese|0",
        },
    },
    {
        "display_name": "mmlu_prox_japanese_math",
        "func": micro_average,
        "func_args": {
            "metric_key": "extractive_match",
            "white_list": MMLU_PROX_JAPANESE_MATH,
        },
        "target": {
            "task_key": "swallow|mmlu_prox_japanese|0",
        },
    },
    {
        "display_name": "mmlu_prox_japanese_physics",
        "func": micro_average,
        "func_args": {
            "metric_key": "extractive_match",
            "white_list": MMLU_PROX_JAPANESE_PHYSICS,
        },
        "target": {
            "task_key": "swallow|mmlu_prox_japanese|0",
        },
    },
    {
        "display_name": "mmlu_prox_japanese_computer_science",
        "func": micro_average,
        "func_args": {
            "metric_key": "extractive_match",
            "white_list": MMLU_PROX_JAPANESE_COMPUTER_SCIENCE,
        },
        "target": {
            "task_key": "swallow|mmlu_prox_japanese|0",
        },
    },
    {
        "display_name": "mmlu_prox_japanese_philosophy",
        "func": micro_average,
        "func_args": {
            "metric_key": "extractive_match",
            "white_list": MMLU_PROX_JAPANESE_PHILOSOPHY,
        },
        "target": {
            "task_key": "swallow|mmlu_prox_japanese|0",
        },
    },
    {
        "display_name": "mmlu_prox_japanese_engineering",
        "func": micro_average,
        "func_args": {
            "metric_key": "extractive_match",
            "white_list": MMLU_PROX_JAPANESE_ENGINEERING,
        },
        "target": {
            "task_key": "swallow|mmlu_prox_japanese|0",
        },
    },
    {
        "display_name": "mmlu_prox_japanese",
        "func": micro_average,
        "func_args": {
            "metric_key": "extractive_match"
        },
        "target": {
            "task_key": "swallow|mmlu_prox_japanese|0",
        },
    },

    # MMLU Prox English
    {
        "display_name": "mmlu_prox_english_business",
        "func": micro_average,
        "func_args": {
            "metric_key": "extractive_match",
            "white_list": MMLU_PROX_ENGLISH_BUSINESS,
        },
        "target": {
            "task_key": "swallow|mmlu_prox_english|0",
        },
    },
    {
        "display_name": "mmlu_prox_english_law",
        "func": micro_average,
        "func_args": {
            "metric_key": "extractive_match",
            "white_list": MMLU_PROX_ENGLISH_LAW,
        },
        "target": {
            "task_key": "swallow|mmlu_prox_english|0",
        },
    },
    {
        "display_name": "mmlu_prox_english_psychology",
        "func": micro_average,
        "func_args": {
            "metric_key": "extractive_match",
            "white_list": MMLU_PROX_ENGLISH_PSYCHOLOGY,
        },
        "target": {
            "task_key": "swallow|mmlu_prox_english|0",
        },
    },
    {
        "display_name": "mmlu_prox_english_biology",
        "func": micro_average,
        "func_args": {
            "metric_key": "extractive_match",
            "white_list": MMLU_PROX_ENGLISH_BIOLOGY,
        },
        "target": {
            "task_key": "swallow|mmlu_prox_english|0",
        },
    },
    {
        "display_name": "mmlu_prox_english_chemistry",
        "func": micro_average,
        "func_args": {
            "metric_key": "extractive_match",
            "white_list": MMLU_PROX_ENGLISH_CHEMISTRY,
        },
        "target": {
            "task_key": "swallow|mmlu_prox_english|0",
        },
    },
    {
        "display_name": "mmlu_prox_english_history",
        "func": micro_average,
        "func_args": {
            "metric_key": "extractive_match",
            "white_list": MMLU_PROX_ENGLISH_HISTORY,
        },
        "target": {
            "task_key": "swallow|mmlu_prox_english|0",
        },
    },
    {
        "display_name": "mmlu_prox_english_other",
        "func": micro_average,
        "func_args": {
            "metric_key": "extractive_match",
            "white_list": MMLU_PROX_ENGLISH_OTHER,
        },
        "target": {
            "task_key": "swallow|mmlu_prox_english|0",
        },
    },
    {
        "display_name": "mmlu_prox_english_health",
        "func": micro_average,
        "func_args": {
            "metric_key": "extractive_match",
            "white_list": MMLU_PROX_ENGLISH_HEALTH,
        },
        "target": {
            "task_key": "swallow|mmlu_prox_english|0",
        },
    },
    {
        "display_name": "mmlu_prox_english_economics",
        "func": micro_average,
        "func_args": {
            "metric_key": "extractive_match",
            "white_list": MMLU_PROX_ENGLISH_ECONOMICS,
        },
        "target": {
            "task_key": "swallow|mmlu_prox_english|0",
        },
    },
    {
        "display_name": "mmlu_prox_english_math",
        "func": micro_average,
        "func_args": {
            "metric_key": "extractive_match",
            "white_list": MMLU_PROX_ENGLISH_MATH,
        },
        "target": {
            "task_key": "swallow|mmlu_prox_english|0",
        },
    },
    {
        "display_name": "mmlu_prox_english_physics",
        "func": micro_average,
        "func_args": {
            "metric_key": "extractive_match",
            "white_list": MMLU_PROX_ENGLISH_PHYSICS,
        },
        "target": {
            "task_key": "swallow|mmlu_prox_english|0",
        },
    },
    {
        "display_name": "mmlu_prox_english_computer_science",
        "func": micro_average,
        "func_args": {
            "metric_key": "extractive_match",
            "white_list": MMLU_PROX_ENGLISH_COMPUTER_SCIENCE,
        },
        "target": {
            "task_key": "swallow|mmlu_prox_english|0",
        },
    },
    {
        "display_name": "mmlu_prox_english_philosophy",
        "func": micro_average,
        "func_args": {
            "metric_key": "extractive_match",
            "white_list": MMLU_PROX_ENGLISH_PHILOSOPHY,
        },
        "target": {
            "task_key": "swallow|mmlu_prox_english|0",
        },
    },
    {
        "display_name": "mmlu_prox_english_engineering",
        "func": micro_average,
        "func_args": {
            "metric_key": "extractive_match",
            "white_list": MMLU_PROX_ENGLISH_ENGINEERING,
        },
        "target": {
            "task_key": "swallow|mmlu_prox_english|0",
        },
    },
    {
        "display_name": "mmlu_prox_english",
        "func": micro_average,
        "func_args": {
            "metric_key": "extractive_match"
        },
        "target": {
            "task_key": "swallow|mmlu_prox_english|0",
        },
    },

    # MMLU Pro English
    {
        "display_name": "mmlu_pro_english_business",
        "func": micro_average,
        "func_args": {
            "metric_key": "extractive_match",
            "white_list": MMLU_PRO_ENGLISH_BUSINESS,
        },
        "target": {
            "task_key": "swallow|mmlu_pro_english|0",
        },
    },
    {
        "display_name": "mmlu_pro_english_law",
        "func": micro_average,
        "func_args": {
            "metric_key": "extractive_match",
            "white_list": MMLU_PRO_ENGLISH_LAW,
        },
        "target": {
            "task_key": "swallow|mmlu_pro_english|0",
        },
    },
    {
        "display_name": "mmlu_pro_english_psychology",
        "func": micro_average,
        "func_args": {
            "metric_key": "extractive_match",
            "white_list": MMLU_PRO_ENGLISH_PSYCHOLOGY,
        },
        "target": {
            "task_key": "swallow|mmlu_pro_english|0",
        },
    },
    {
        "display_name": "mmlu_pro_english_biology",
        "func": micro_average,
        "func_args": {
            "metric_key": "extractive_match",
            "white_list": MMLU_PRO_ENGLISH_BIOLOGY,
        },
        "target": {
            "task_key": "swallow|mmlu_pro_english|0",
        },
    },
    {
        "display_name": "mmlu_pro_english_chemistry",
        "func": micro_average,
        "func_args": {
            "metric_key": "extractive_match",
            "white_list": MMLU_PRO_ENGLISH_CHEMISTRY,
        },
        "target": {
            "task_key": "swallow|mmlu_pro_english|0",
        },
    },
    {
        "display_name": "mmlu_pro_english_history",
        "func": micro_average,
        "func_args": {
            "metric_key": "extractive_match",
            "white_list": MMLU_PRO_ENGLISH_HISTORY,
        },
        "target": {
            "task_key": "swallow|mmlu_pro_english|0",
        },
    },
    {
        "display_name": "mmlu_pro_english_other",
        "func": micro_average,
        "func_args": {
            "metric_key": "extractive_match",
            "white_list": MMLU_PRO_ENGLISH_OTHER,
        },
        "target": {
            "task_key": "swallow|mmlu_pro_english|0",
        },
    },
    {
        "display_name": "mmlu_pro_english_health",
        "func": micro_average,
        "func_args": {
            "metric_key": "extractive_match",
            "white_list": MMLU_PRO_ENGLISH_HEALTH,
        },
        "target": {
            "task_key": "swallow|mmlu_pro_english|0",
        },
    },
    {
        "display_name": "mmlu_pro_english_economics",
        "func": micro_average,
        "func_args": {
            "metric_key": "extractive_match",
            "white_list": MMLU_PRO_ENGLISH_ECONOMICS,
        },
        "target": {
            "task_key": "swallow|mmlu_pro_english|0",
        },
    },
    {
        "display_name": "mmlu_pro_english_math",
        "func": micro_average,
        "func_args": {
            "metric_key": "extractive_match",
            "white_list": MMLU_PRO_ENGLISH_MATH,
        },
        "target": {
            "task_key": "swallow|mmlu_pro_english|0",
        },
    },
    {
        "display_name": "mmlu_pro_english_physics",
        "func": micro_average,
        "func_args": {
            "metric_key": "extractive_match",
            "white_list": MMLU_PRO_ENGLISH_PHYSICS,
        },
        "target": {
            "task_key": "swallow|mmlu_pro_english|0",
        },
    },
    {
        "display_name": "mmlu_pro_english_computer_science",
        "func": micro_average,
        "func_args": {
            "metric_key": "extractive_match",
            "white_list": MMLU_PRO_ENGLISH_COMPUTER_SCIENCE,
        },
        "target": {
            "task_key": "swallow|mmlu_pro_english|0",
        },
    },
    {
        "display_name": "mmlu_pro_english_philosophy",
        "func": micro_average,
        "func_args": {
            "metric_key": "extractive_match",
            "white_list": MMLU_PRO_ENGLISH_PHILOSOPHY,
        },
        "target": {
            "task_key": "swallow|mmlu_pro_english|0",
        },
    },
    {
        "display_name": "mmlu_pro_english_engineering",
        "func": micro_average,
        "func_args": {
            "metric_key": "extractive_match",
            "white_list": MMLU_PRO_ENGLISH_ENGINEERING,
        },
        "target": {
            "task_key": "swallow|mmlu_pro_english|0",
        },
    },
    {
        "display_name": "mmlu_pro_english",
        "func": micro_average,
        "func_args": {
            "metric_key": "extractive_match"
        },
        "target": {
            "task_key": "swallow|mmlu_pro_english|0",
        },
    },

    # MMLU English
    {
        "display_name": "mmlu_english_stem",
        "func": micro_average,
        "func_args": {
            "metric_key": "extractive_match",
            "white_list": MMLU_ENGLISH_STEM,
        },
        "target": {
            "task_key": "swallow|mmlu_english|0",
        },
    },
    {
        "display_name": "mmlu_english_humanities",
        "func": micro_average,
        "func_args": {
            "metric_key": "extractive_match",
            "white_list": MMLU_ENGLISH_HUMANITIES,
        },
        "target": {
            "task_key": "swallow|mmlu_english|0",
        },
    },
    {
        "display_name": "mmlu_english_social_sciences",
        "func": micro_average,
        "func_args": {
            "metric_key": "extractive_match",
            "white_list": MMLU_ENGLISH_SOCIAL_SCIENCES,
        },
        "target": {
            "task_key": "swallow|mmlu_english|0",
        },
    },
    {
        "display_name": "mmlu_english_other",
        "func": micro_average,
        "func_args": {
            "metric_key": "extractive_match",
            "white_list": MMLU_ENGLISH_OTHER,
        },
        "target": {
            "task_key": "swallow|mmlu_english|0",
        },
    },
    {
        "display_name": "mmlu_english",
        "func": micro_average,
        "func_args": {
            "metric_key": "extractive_match"
        },
        "target": {
            "task_key": "swallow|mmlu_english|0",
        },
    },

    # Japanese MTBench
    {
        "display_name": "japanese_mtbench_coding",
        "func": pick,
        "func_args": {
            "metric_key": "judge_score_coding_avg"
        },
        "target": {
            "task_key": "swallow|japanese_mt_bench|0,swallow|japanese_mt_bench_truncate_6144|0"
        },
    },
    {
        "display_name": "japanese_mtbench_extraction",
        "func": pick,
        "func_args": {
            "metric_key": "judge_score_extraction_avg"
        },
        "target": {
            "task_key": "swallow|japanese_mt_bench|0,swallow|japanese_mt_bench_truncate_6144|0"
        },
    },
    {
        "display_name": "japanese_mtbench_humanities",
        "func": pick,
        "func_args": {
            "metric_key": "judge_score_humanities_avg"
        },
        "target": {
            "task_key": "swallow|japanese_mt_bench|0,swallow|japanese_mt_bench_truncate_6144|0"
        },
    },
    {
        "display_name": "japanese_mtbench_math",
        "func": pick,
        "func_args": {
            "metric_key": "judge_score_math_avg"
        },
        "target": {
            "task_key": "swallow|japanese_mt_bench|0,swallow|japanese_mt_bench_truncate_6144|0"
        },
    },
    {
        "display_name": "japanese_mtbench_reasoning",
        "func": pick,
        "func_args": {
            "metric_key": "judge_score_reasoning_avg"
        },
        "target": {
            "task_key": "swallow|japanese_mt_bench|0,swallow|japanese_mt_bench_truncate_6144|0"
        },
    },
    {
        "display_name": "japanese_mtbench_roleplay",
        "func": pick,
        "func_args": {
            "metric_key": "judge_score_roleplay_avg"
        },
        "target": {
            "task_key": "swallow|japanese_mt_bench|0,swallow|japanese_mt_bench_truncate_6144|0"
        },
    },
    {
        "display_name": "japanese_mtbench_stem",
        "func": pick,
        "func_args": {
            "metric_key": "judge_score_stem_avg"
        },
        "target": {
            "task_key": "swallow|japanese_mt_bench|0,swallow|japanese_mt_bench_truncate_6144|0"
        },
    },
    {
        "display_name": "japanese_mtbench_writing",
        "func": pick,
        "func_args": {
            "metric_key": "judge_score_writing_avg"
        },
        "target": {
            "task_key": "swallow|japanese_mt_bench|0,swallow|japanese_mt_bench_truncate_6144|0"
        },
    },
    {
        "display_name": "japanese_mtbench_first_turn",
        "func": pick,
        "func_args": {
            "metric_key": "judge_score_overall_turn_1_avg"
        },
        "target": {
            "task_key": "swallow|japanese_mt_bench|0,swallow|japanese_mt_bench_truncate_6144|0"
        },
    },
    {
        "display_name": "japanese_mtbench_second_turn",
        "func": pick,
        "func_args": {
            "metric_key": "judge_score_overall_turn_2_avg"
        },
        "target": {
            "task_key": "swallow|japanese_mt_bench|0,swallow|japanese_mt_bench_truncate_6144|0"
        },
    },
    {
        "display_name": "japanese_mtbench_average",
        "func": pick,
        "func_args": {
            "metric_key": "judge_score_overall_avg"
        },
        "target": {
            "task_key": "swallow|japanese_mt_bench|0,swallow|japanese_mt_bench_truncate_6144|0"
        },
    },
    {
        "display_name": "japanese_mtbench_coding_japanese_ratio",
        "func": pick,
        "func_args": {
            "metric_key": "japanese_ratio_coding"
        },
        "target": {
            "task_key": "swallow|japanese_mt_bench|0,swallow|japanese_mt_bench_truncate_6144|0"
        },
    },
    {
        "display_name": "japanese_mtbench_extraction_japanese_ratio",
        "func": pick,
        "func_args": {
            "metric_key": "japanese_ratio_extraction"
        },
        "target": {
            "task_key": "swallow|japanese_mt_bench|0,swallow|japanese_mt_bench_truncate_6144|0"
        },
    },
    {
        "display_name": "japanese_mtbench_humanities_japanese_ratio",
        "func": pick,
        "func_args": {
            "metric_key": "japanese_ratio_humanities"
        },
        "target": {
            "task_key": "swallow|japanese_mt_bench|0,swallow|japanese_mt_bench_truncate_6144|0"
        },
    },
    {
        "display_name": "japanese_mtbench_math_japanese_ratio",
        "func": pick,
        "func_args": {
            "metric_key": "japanese_ratio_math"
        },
        "target": {
            "task_key": "swallow|japanese_mt_bench|0,swallow|japanese_mt_bench_truncate_6144|0"
        },
    },
    {
        "display_name": "japanese_mtbench_reasoning_japanese_ratio",
        "func": pick,
        "func_args": {
            "metric_key": "japanese_ratio_reasoning"
        },
        "target": {
            "task_key": "swallow|japanese_mt_bench|0,swallow|japanese_mt_bench_truncate_6144|0"
        },
    },
    {
        "display_name": "japanese_mtbench_roleplay_japanese_ratio",
        "func": pick,
        "func_args": {
            "metric_key": "japanese_ratio_roleplay"
        },
        "target": {
            "task_key": "swallow|japanese_mt_bench|0,swallow|japanese_mt_bench_truncate_6144|0"
        },
    },
    {
        "display_name": "japanese_mtbench_stem_japanese_ratio",
        "func": pick,
        "func_args": {
            "metric_key": "japanese_ratio_stem"
        },
        "target": {
            "task_key": "swallow|japanese_mt_bench|0,swallow|japanese_mt_bench_truncate_6144|0"
        },
    },
    {
        "display_name": "japanese_mtbench_writing_japanese_ratio",
        "func": pick,
        "func_args": {
            "metric_key": "japanese_ratio_writing"
        },
        "target": {
            "task_key": "swallow|japanese_mt_bench|0,swallow|japanese_mt_bench_truncate_6144|0"
        },
    },
    {
        "display_name": "japanese_mtbench_average_japanese_ratio",
        "func": pick,
        "func_args": {
            "metric_key": "japanese_ratio_overall"
        },
        "target": {
            "task_key": "swallow|japanese_mt_bench|0,swallow|japanese_mt_bench_truncate_6144|0"
        },
    },

    # English MTBench
    {
        "display_name": "english_mtbench_coding",
        "func": pick,
        "func_args": {
            "metric_key": "judge_score_coding_avg"
        },
        "target": {
            "task_key": "swallow|english_mt_bench|0"
        },
    },
    {
        "display_name": "english_mtbench_extraction",
        "func": pick,
        "func_args": {
            "metric_key": "judge_score_extraction_avg"
        },
        "target": {
            "task_key": "swallow|english_mt_bench|0"
        },
    },
    {
        "display_name": "english_mtbench_humanities",
        "func": pick,
        "func_args": {
            "metric_key": "judge_score_humanities_avg"
        },
        "target": {
            "task_key": "swallow|english_mt_bench|0"
        },
    },
    {
        "display_name": "english_mtbench_math",
        "func": pick,
        "func_args": {
            "metric_key": "judge_score_math_avg"
        },
        "target": {
            "task_key": "swallow|english_mt_bench|0"
        },
    },
    {
        "display_name": "english_mtbench_reasoning",
        "func": pick,
        "func_args": {
            "metric_key": "judge_score_reasoning_avg"
        },
        "target": {
            "task_key": "swallow|english_mt_bench|0"
        },
    },
    {
        "display_name": "english_mtbench_roleplay",
        "func": pick,
        "func_args": {
            "metric_key": "judge_score_roleplay_avg"
        },
        "target": {
            "task_key": "swallow|english_mt_bench|0"
        },
    },
    {
        "display_name": "english_mtbench_stem",
        "func": pick,
        "func_args": {
            "metric_key": "judge_score_stem_avg"
        },
        "target": {
            "task_key": "swallow|english_mt_bench|0"
        },
    },
    {
        "display_name": "english_mtbench_writing",
        "func": pick,
        "func_args": {
            "metric_key": "judge_score_writing_avg"
        },
        "target": {
            "task_key": "swallow|english_mt_bench|0"
        },
    },
    {
        "display_name": "english_mtbench_first_turn",
        "func": pick,
        "func_args": {
            "metric_key": "judge_score_overall_turn_1_avg"
        },
        "target": {
            "task_key": "swallow|english_mt_bench|0"
        },
    },
    {
        "display_name": "english_mtbench_second_turn",
        "func": pick,
        "func_args": {
            "metric_key": "judge_score_overall_turn_2_avg"
        },
        "target": {
            "task_key": "swallow|english_mt_bench|0"
        },
    },
    {
        "display_name": "english_mtbench_average",
        "func": pick,
        "func_args": {
            "metric_key": "judge_score_overall_avg"
        },
        "target": {
            "task_key": "swallow|english_mt_bench|0"
        },
    },
    # JEMHopQA
    {
        "display_name": "jemhopqa_cot_f1_score_quasi",
        "func": pick,
        "func_args": {
            'metric_key': 'f1_score_quasi'
        }, 
        'target': {
            'task_key': 'swallow|jemhopqa_cot|0'
        }
    },
    {
        "display_name": "jemhopqa_cot_f1_score",
        "func": pick,
        "func_args": {
            'metric_key': 'f1_score'
        }, 
        'target': {
            'task_key': 'swallow|jemhopqa_cot|0'
        }
    },
    # WMT20 En-Ja
    {
        "display_name": "wmt20_en_ja_bleu",
        "func": pick,
        "func_args": {
            'metric_key': 'bleu'
        }, 
        'target': {
            'task_key': 'swallow|wmt20:en-ja|0'
        }
    },
    {
        "display_name": "wmt20_en_ja_bleu_lmevalja",
        "func": pick,
        "func_args": {
            'metric_key': 'bleu_lmevalja'
        }, 
        'target': {
            'task_key': 'swallow|wmt20:en-ja|0'
        }
    },
    # WMT20 Ja-En
    {
        "display_name": "wmt20_ja_en_bleu",
        "func": pick,
        "func_args": {
            'metric_key': 'bleu'
        }, 
        'target': {
            'task_key': 'swallow|wmt20:ja-en|0'
        }
    },
    # M-IFEval 日本語版
    {
        "display_name": "mifeval_ja_inst_level_strict_acc",
        "func": pick,
        "func_args": {
            'metric_key': 'inst_level_strict_acc'
        }, 
        'target': {
            'task_key': 'swallow|mifeval_ja|0'
        }
    },
    {
        "display_name": "mifeval_ja_average",
        "func": average_in_one_task,        
        "func_args": {
            "metric_key_list": M_IFEVAL
        },
        'target': {
            'task_key': 'swallow|mifeval_ja|0'
        }
    },
    # MCLM MATH-100 日本語サブセット (=MATH-500邦訳版)
    {
        "display_name": "mclm_math_100_japanese",
        "func": pick,
        "func_args": {
            'metric_key': 'extractive_match'
        }, 
        'target': {
            'task_key': 'swallow|math_100_japanese|0'
        }
    },
    # BenchMAX Science Reasoning 日本語版 (=GPQA邦訳版)
    {
        "display_name": "gpqa_main_ja",
        "func": pick,
        "func_args": {
            'metric_key': 'extractive_match'
        }, 
        'target': {
            'task_key': 'swallow|swallow_gpqa_ja|0'
        }
    },
    # HellaSwag
    {
        "display_name": "swallow_hellaswag",
        "func": pick,
        "func_args": {
            'metric_key': 'extractive_match'
        }, 
        'target': {
            'task_key': 'swallow|hellaswag|0'
        }
    },
    # GPQA - Diamond subset
    {
        "display_name": "gpqa_diamond",
        "func": pick,
        "func_args": {
            'metric_key': 'extractive_match'
        }, 
        'target': {
            'task_key': 'swallow|gpqa:diamond|0'
        }
    },
    # MATH-500
    {
        "display_name": "math_500",
        "func": pick,
        "func_args": {
            'metric_key': 'extractive_match'
        }, 
        'target': {
            'task_key': 'swallow|math_500|0'
        }
    },
    # AIME 2024--2025    
    # タスク名を aime:{24,25} に変更したので --tasks "swallow|aime|0" で実行すればよい
    # スコア一覧表の Task列には swallow:aime:24:0, swallow:aime:25:0 と表示される
    {
        "display_name": "aime_2024_2025",
        "func": micro_average,
        "func_args": {
            'metric_key': 'extractive_match'
        }, 
        'target': {
            'task_key': 'swallow|aime|0'
        }
    },
    # LiveCodeBench v5 & v6 の設問
    # 指標名を Pass@k:10 （=10回試行で推定したPass@1）に変更した
    {
        "display_name": "livecodebench_v5_v6_pass@1",
        "func": pick,
        "func_args": {
            'metric_key': 'codegen_pass@1:10'
        }, 
        'target': {
            'task_key': 'swallow|lcb:codegeneration_v5_v6|0'
        }
    },
    {
        "display_name": "livecodebench_v5_v6_pass@10",
        "func": pick,
        "func_args": {
            'metric_key': 'codegen_pass@10:10'
        }, 
        'target': {
            'task_key': 'swallow|lcb:codegeneration_v5_v6|0'
        }
    },
    # HumanEval
    {
        'display_name': 'humaneval_pass@1', 
        'func': pick, 
        'func_args': {
            'metric_key': 'humaneval_pass@1:10'
        }, 
        'target': {
            'task_key': 'swallow|humaneval|0'
        },
    },
    {
        'display_name': 'humaneval_pass@10', 
        'func': pick, 
        'func_args': {
            'metric_key': 'humaneval_pass@10:10'
        }, 
        'target': {
            'task_key': 'swallow|humaneval|0'
        },
    },
    # HumanEvalPlus
    {
        'display_name': 'humanevalplus_pass@1', 
        'func': pick, 
        'func_args': {
            'metric_key': 'humaneval_pass@1:10'
        }, 
        'target': {
            'task_key': 'swallow|humanevalplus|0'
        },
    },
    {
        'display_name': 'humanevalplus_pass@10', 
        'func': pick, 
        'func_args': {
            'metric_key': 'humaneval_pass@10:10'
        }, 
        'target': {
            'task_key': 'swallow|humanevalplus|0'
        },
    },
]

## 集計関数 ##############################################################################################################

def main(
        model_name: str,
        raw_results_dir: str,
        aggregated_outputs_dir: str,
        ):
    # raw_results_dir 内のすべての JSON ファイルを取得する
    pattern = os.path.join(raw_results_dir, "*.json")
    all_results_files: list[str] = glob.glob(pattern)

    # 各結果ファイルからタスク名，メトリクス，必要時間，実行日時を抽出し集約する
    all_results_dicts: list[dict[str, Any]] = []
    for file_path in all_results_files:
        filename = os.path.basename(file_path)
        # ファイル名は "results_YYYY-MM-DDTHH-MM-SS.microsec.json" の形と仮定する
        try:
            # "results_" と ".json" を除きタイムスタンプ部分を取得
            timestamp_str = filename[len("results_"):-len(".json")]
            execution_dt = datetime.strptime(timestamp_str, "%Y-%m-%dT%H-%M-%S.%f")
        except Exception as e:
            print(f"Skipping file {filename} due to datetime parsing error: {e}")
            continue

        with open(file_path, "r") as f:
            data = json.load(f)
        # 結果が対象のモデルか確認
        if data.get("config_general", {}).get("model_name") != model_name:
            print(f"Skipping file {filename} as it does not match the model name {model_name}. config_general: {data.get('config_general', {})}, model_name: {model_name}")
            continue
        config = data.get("config_general", {})
        # total_evaluation_time_secondes があればそれを使用，なければ start_time と end_time の差分で計算
        if "total_evaluation_time_secondes" in config:
            try:
                required_time = float(config["total_evaluation_time_secondes"])
            except Exception:
                required_time = None
        else:
            start_time = config.get("start_time")
            end_time = config.get("end_time")
            if start_time is not None and end_time is not None:
                required_time = end_time - start_time
            else:
                required_time = None

        # results 内の各タスク毎の情報を抽出
        config_tasks = data["config_tasks"]
        for task_key, metrics in data.get("results", {}).items():
            # task_key の形式例: "swallow|swallow_jmmlu:abstract_algebra|0"
            # "|" で分割し、2 番目（subset）に "_average" が含まれるものや task_key=="all" はスキップする
            parts = task_key.split("|")
            if len(parts) >= 2:
                if ":" in parts[1]:
                    task_name, subset_name = parts[1].split(":")
                else:
                    task_name, subset_name = parts[1], ""
                if subset_name == "_average":
                    continue
            else:
                if task_key == "all":
                    continue
                task_name = task_key
                subset_name = ""
            entry = {
                "task_key": task_key,
                "task": task_name,
                "subset": subset_name,
                "sample_num": config_tasks['|'.join(task_key.split("|")[:-1])].get("effective_num_docs", -1) if type( config_tasks['|'.join(task_key.split("|")[:-1])]) is dict else -1,
                "execution_datetime": execution_dt.isoformat(),
                "required_time": required_time,
                "metrics": metrics
            }
            all_results_dicts.append(entry)

    # 公開用に警告を出す機構を追加
    assert len(all_results_dicts) > 0, f"{model_name}に対する結果が見つけられませんでした．\nvLLM serve → lighteval の方法で評価した場合は model_name の前に 'hosted_vllm' を付けてください．例: 'hosted_vllm/tokyotech-llm/Llama-3.1-Swallow-8B-Instruct-v0.5'．\nもし特定のプロバイダを用いた場合は必要に応じて model_name の前にプロバイダ名を付けてください．"

    # 各タスクごとに最新のエントリのみを抽出する（キーは "task:subset"）
    latest_results = {}
    for entry in all_results_dicts:
        task_and_subset = f"{entry['task']}:{entry['subset']}"
        current_dt = datetime.fromisoformat(entry["execution_datetime"])
        if task_and_subset not in latest_results:
            latest_results[task_and_subset] = entry
        else:
            prev_dt = datetime.fromisoformat(latest_results[task_and_subset]["execution_datetime"])
            if current_dt > prev_dt:
                latest_results[task_and_subset] = entry

    # aggregated_results の作成
    # AGGREGATE_CONF で定義されている各項目に対して、集約関数を適用する
    aggregated_results = {
        "model": model_name,
        "results": {},
        "overall": "",
        "tasks": [],
    }
    for conf in AGGREGATE_CONF:
        display_name = conf['display_name']
        func = conf['func']
        func_args = conf['func_args']
        target = conf['target']
        try:
            value = func(latest_results, target, **func_args)
            assert value is not None, f"Try calculating {display_name}, but received None."
            if value == -1:
                print(f"No samples found for {display_name}")
        except Exception as e:
            print(f"Error processing {display_name}: {e}")
            value = -1
        aggregated_results["results"][display_name] = value
        aggregated_results["tasks"].append(display_name)
    # overall: すべての結果の値をカンマ区切りの文字列として格納
    aggregated_results["overall"] = ",".join(str(v) for v in aggregated_results["results"].values())

    # aggregated_outputs_dir が存在しなければ作成する
    os.makedirs(aggregated_outputs_dir, exist_ok=True)
    aggregated_filepath = os.path.join(aggregated_outputs_dir, "aggregated_results.json")
    
    # 結果を aggregated_filepath に書き出す
    with open(aggregated_filepath, "w") as f:
        json.dump(aggregated_results, f, indent=2)
    print(f"Aggregated results saved to {aggregated_filepath}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, help="処理対象のモデル名")
    parser.add_argument("--raw_results_dir", type=str, help="個別結果ファイルが存在するディレクトリのパス", default="./lighteval/outputs/results")
    parser.add_argument("--aggregated_outputs_dir", type=str, help="集約結果を保存するディレクトリのパス", default="./aggregated_results")
    args = parser.parse_args()

    args.raw_results_dir = os.path.join(args.raw_results_dir, args.model_name)
    args.aggregated_outputs_dir = os.path.join(args.aggregated_outputs_dir, args.model_name)

    main(**vars(args))
