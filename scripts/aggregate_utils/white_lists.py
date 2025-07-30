# White lists
# - are used to categorize subsets when calculating metrics, such as micro-average.

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
