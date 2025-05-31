from .white_lists import (
    JMMLU_SOCIAL_SCIENCES,
    JMMLU_HUMANITIES,
    JMMLU_STEM,
    JMMLU_OTHERS,
    MMLU_PROX_JAPANESE_BUSINESS,
    MMLU_PROX_JAPANESE_LAW,
    MMLU_PROX_JAPANESE_PSYCHOLOGY,
    MMLU_PROX_JAPANESE_BIOLOGY,
    MMLU_PROX_JAPANESE_CHEMISTRY,
    MMLU_PROX_JAPANESE_HISTORY,
    MMLU_PROX_JAPANESE_OTHER,
    MMLU_PROX_JAPANESE_HEALTH,
    MMLU_PROX_JAPANESE_ECONOMICS,
    MMLU_PROX_JAPANESE_MATH,
    MMLU_PROX_JAPANESE_PHYSICS,
    MMLU_PROX_JAPANESE_COMPUTER_SCIENCE,
    MMLU_PROX_JAPANESE_PHILOSOPHY,
    MMLU_PROX_JAPANESE_ENGINEERING,
)
from .funcs import micro_average

# Aggregate Config
# - Each entry must have 'display_name', 'func', and 'target'. 'target' must have 'task_key' and 'is_category'.
# - Each entry may have 'func_args' matching its 'func'.
AGGREGATE_CONF = [
    {
        "display_name": "jmmlu_social_sciences",
        "func": micro_average,
        "func_args": {"metric_key": "extractive_match", "white_list": JMMLU_SOCIAL_SCIENCES},
        "target": {"task_key": "swallow|swallow_jmmlu|0", "is_category": True},
    },
    {
        "display_name": "jmmlu_humanities",
        "func": micro_average,
        "func_args": {"metric_key": "extractive_match", "white_list": JMMLU_HUMANITIES},
        "target": {"task_key": "swallow|swallow_jmmlu|0", "is_category": True},
    },
    {
        "display_name": "jmmlu_stem",
        "func": micro_average,
        "func_args": {"metric_key": "extractive_match", "white_list": JMMLU_STEM},
        "target": {"task_key": "swallow|swallow_jmmlu|0", "is_category": True},
    },
    {
        "display_name": "jmmlu_other",
        "func": micro_average,
        "func_args": {"metric_key": "extractive_match", "white_list": JMMLU_OTHERS},
        "target": {"task_key": "swallow|swallow_jmmlu|0", "is_category": True},
    },
    {
        "display_name": "jmmlu",
        "func": micro_average,
        "func_args": {"metric_key": "extractive_match"},
        "target": {"task_key": "swallow|swallow_jmmlu|0", "is_category": True},
    },
    {
        "display_name": "mmlu_prox_japanese_business",
        "func": micro_average,
        "func_args": {
            "metric_key": "extractive_match",
            "white_list": MMLU_PROX_JAPANESE_BUSINESS,
        },
        "target": {
            "task_key": "swallow|mmlu_prox_japanese|0",
            "is_category": True,
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
            "is_category": True,
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
            "is_category": True,
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
            "is_category": True,
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
            "is_category": True,
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
            "is_category": True,
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
            "is_category": True,
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
            "is_category": True,
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
            "is_category": True,
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
            "is_category": True,
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
            "is_category": True,
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
            "is_category": True,
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
            "is_category": True,
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
            "is_category": True,
        },
    },
    {
        "display_name": "mmlu_prox_japanese",
        "func": micro_average,
        "func_args": {"metric_key": "extractive_match"},
        "target": {
            "task_key": "swallow|mmlu_prox_japanese|0",
            "is_category": True,
        },
    },
]
