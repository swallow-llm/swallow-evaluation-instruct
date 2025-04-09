from lighteval.tasks.lighteval_task import LightevalTaskConfig


# SWALLOW JMMLU SUBSETS consist of all JMMLU subsets
# excluding ones with CC-BY-NC-ND license: japanese_civics, japanese_idiom, and japanese_geography.
SWALLOW_JMMLU_SUBSETS = [
    'japanese_history', 
    'miscellaneous', 
    'security_studies', 
    'virology', 
    'nutrition', 
    'human_sexuality', 
    'college_mathematics', 
    'econometrics', 
    'computer_security', 
    'clinical_knowledge', 
    'machine_learning', 
    'high_school_chemistry', 
    'human_aging', 
    'logical_fallacies', 
    'sociology', 
    'high_school_european_history', 
    'high_school_statistics', 
    'high_school_physics', 
    'high_school_microeconomics', 
    'college_physics', 
    'anatomy', 
    'high_school_psychology', 
    'business_ethics', 
    'professional_psychology', 
    'college_medicine', 
    'elementary_mathematics', 
    'moral_disputes', 
    'marketing', 
    'high_school_macroeconomics', 
    'world_religions',
    'conceptual_physics',
    'professional_medicine',
    'prehistory',
    'high_school_mathematics',
    'international_law',
    'philosophy',
    'management',
    'high_school_computer_science',
    'medical_genetics',
    'college_computer_science',
    'public_relations',
    'professional_accounting',
    'abstract_algebra',
    'global_facts',
    'college_biology',
    'high_school_geography',
    'world_history',
    'high_school_biology',
    'college_chemistry',
    'electrical_engineering',
    'astronomy',
    'jurisprudence',
    'formal_logic',
]


SWALLOW_JMMLU_QUERY_TEMPLATE = """
Answer the following multiple choice question. The last line of your response should be of the following format: 'Answer: $LETTER' (without quotes) where LETTER is one of ABCD. Think step by step before answering.

{Question}

A) {A}
B) {B}
C) {C}
D) {D}
""".strip()


def jmmlu_prompt_fn(line, task_name: str = None):
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
        instruction=query,
    )


multi_choice_metric = multilingual_extractive_match_metric(
    language=Language.JAPANESE,
    gold_extraction_target=[IndicesExtractionConfig(prefix_for_extraction="NativeLetters")],
    pred_extraction_target=[IndicesExtractionConfig(prefix_for_extraction="NativeLetters")],
    precision=5,
)


swallow_jmmlu_tasks = [
    LightevalTaskConfig(
        name=f"swallow_jmmlu:{subset}",
        prompt_function=jmmlu_prompt_fn,
        suite=["swallow", "swallow_jmmlu"],
        hf_repo="nlp-waseda/JMMLU",
        hf_subset=subset,
        evaluation_splits=["test"],
        hf_avail_split=["test"],
        trust_dataset=True,
        stop_sequence=[multi_choice_metric],
        metric=[],
        version=0,
    )
    for subset in SWALLOW_JMMLU_SUBSETS
]