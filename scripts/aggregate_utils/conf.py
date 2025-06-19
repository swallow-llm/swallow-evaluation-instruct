from .white_lists import *
from .funcs import micro_average, average_in_one_task, pick


# Aggregate Config
# - Each entry must have 'display_name', 'func', and 'target'. 'target' must have 'task_key' and 'is_category'.
# - Each entry may have 'func_args' matching its 'func'.
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
            'metric_key': 'jhumaneval_pass@1:10'
        }, 
        'target': {
            'task_key': 'swallow|swallow_jhumaneval|0'
        },
    },
    {
        'display_name': 'jhumaneval_pass@10', 
        'func': pick, 
        'func_args': {
            'metric_key': 'jhumaneval_pass@10:10'
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
        "func_args": {"metric_key": "extractive_match"},
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

    # Japanese MTBench
    {
        "display_name": "japanese_mtbench_coding",
        "func": average_in_one_task,
        "func_args": {
            "metric_key_list": JAPANESE_MTBENCH_CODING
        },
        "target": {
            "task_key": "swallow|japanese_mt_bench|0"
        },
    },
    {
        "display_name": "japanese_mtbench_extraction",
        "func": average_in_one_task,
        "func_args": {
            "metric_key_list": JAPANESE_MTBENCH_EXTRACTION
        },
        "target": {
            "task_key": "swallow|japanese_mt_bench|0"
        },
    },
    {
        "display_name": "japanese_mtbench_humanities",
        "func": average_in_one_task,
        "func_args": {
            "metric_key_list": JAPANESE_MTBENCH_HUMANITIES
        },
        "target": {
            "task_key": "swallow|japanese_mt_bench|0"
        },
    },
    {
        "display_name": "japanese_mtbench_math",
        "func": average_in_one_task,
        "func_args": {
            "metric_key_list": JAPANESE_MTBENCH_MATH
        },
        "target": {
            "task_key": "swallow|japanese_mt_bench|0"
        },
    },
    {
        "display_name": "japanese_mtbench_reasoning",
        "func": average_in_one_task,
        "func_args": {
            "metric_key_list": JAPANESE_MTBENCH_REASONING
        },
        "target": {
            "task_key": "swallow|japanese_mt_bench|0"
        },
    },
    {
        "display_name": "japanese_mtbench_roleplay",
        "func": average_in_one_task,
        "func_args": {
            "metric_key_list": JAPANESE_MTBENCH_ROLEPLAY
        },
        "target": {
            "task_key": "swallow|japanese_mt_bench|0"
        },
    },
    {
        "display_name": "japanese_mtbench_stem",
        "func": average_in_one_task,
        "func_args": {
            "metric_key_list": JAPANESE_MTBENCH_STEM
        },
        "target": {
            "task_key": "swallow|japanese_mt_bench|0"
        },
    },
    {
        "display_name": "japanese_mtbench_writing",
        "func": average_in_one_task,
        "func_args": {
            "metric_key_list": JAPANESE_MTBENCH_WRITING
        },
        "target": {
            "task_key": "swallow|japanese_mt_bench|0"
        },
    },
    {
        "display_name": "japanese_mtbench_first_turn",
        "func": average_in_one_task,
        "func_args": {
            "metric_key_list": JAPANESE_MTBENCH_FIRST_TURN
        },
        "target": {
            "task_key": "swallow|japanese_mt_bench|0"
        },
    },
    {
        "display_name": "japanese_mtbench_second_turn",
        "func": average_in_one_task,
        "func_args": {
            "metric_key_list": JAPANESE_MTBENCH_SECOND_TURN
        },
        "target": {
            "task_key": "swallow|japanese_mt_bench|0"
        },
    },
    {
        "display_name": "japanese_mtbench_average",
        "func": average_in_one_task,
        "func_args": {
            "metric_key_list": JAPANESE_MTBENCH_ALL
        },
        "target": {
            "task_key": "swallow|japanese_mt_bench|0"
        },
    },
    {
        "display_name": "japanese_mtbench_coding_japanese_ratio",
        "func": pick,
        "func_args": {
            "metric_key": "japanese_ratio_coding"
        },
        "target": {
            "task_key": "swallow|japanese_mt_bench|0"
        },
    },
    {
        "display_name": "japanese_mtbench_extraction_japanese_ratio",
        "func": pick,
        "func_args": {
            "metric_key": "japanese_ratio_extraction"
        },
        "target": {
            "task_key": "swallow|japanese_mt_bench|0"
        },
    },
    {
        "display_name": "japanese_mtbench_humanities_japanese_ratio",
        "func": pick,
        "func_args": {
            "metric_key": "japanese_ratio_humanities"
        },
        "target": {
            "task_key": "swallow|japanese_mt_bench|0"
        },
    },
    {
        "display_name": "japanese_mtbench_math_japanese_ratio",
        "func": pick,
        "func_args": {
            "metric_key": "japanese_ratio_math"
        },
        "target": {
            "task_key": "swallow|japanese_mt_bench|0"
        },
    },
    {
        "display_name": "japanese_mtbench_reasoning_japanese_ratio",
        "func": pick,
        "func_args": {
            "metric_key": "japanese_ratio_reasoning"
        },
        "target": {
            "task_key": "swallow|japanese_mt_bench|0"
        },
    },
    {
        "display_name": "japanese_mtbench_roleplay_japanese_ratio",
        "func": pick,
        "func_args": {
            "metric_key": "japanese_ratio_roleplay"
        },
        "target": {
            "task_key": "swallow|japanese_mt_bench|0"
        },
    },
    {
        "display_name": "japanese_mtbench_stem_japanese_ratio",
        "func": pick,
        "func_args": {
            "metric_key": "japanese_ratio_stem"
        },
        "target": {
            "task_key": "swallow|japanese_mt_bench|0"
        },
    },
    {
        "display_name": "japanese_mtbench_writing_japanese_ratio",
        "func": pick,
        "func_args": {
            "metric_key": "japanese_ratio_writing"
        },
        "target": {
            "task_key": "swallow|japanese_mt_bench|0"
        },
    },
    {
        "display_name": "japanese_mtbench_average_japanese_ratio",
        "func": pick,
        "func_args": {
            "metric_key": "japanese_ratio_overall"
        },
        "target": {
            "task_key": "swallow|japanese_mt_bench|0"
        },
    },

    # English MTBench
    {
        "display_name": "english_mtbench_coding",
        "func": average_in_one_task,
        "func_args": {
            "metric_key_list": ENGLISH_MTBENCH_CODING
        },
        "target": {
            "task_key": "swallow|english_mt_bench|0"
        },
    },
    {
        "display_name": "english_mtbench_extraction",
        "func": average_in_one_task,
        "func_args": {
            "metric_key_list": ENGLISH_MTBENCH_EXTRACTION
        },
        "target": {
            "task_key": "swallow|english_mt_bench|0"
        },
    },
    {
        "display_name": "english_mtbench_humanities",
        "func": average_in_one_task,
        "func_args": {
            "metric_key_list": ENGLISH_MTBENCH_HUMANITIES
        },
        "target": {
            "task_key": "swallow|english_mt_bench|0"
        },
    },
    {
        "display_name": "english_mtbench_math",
        "func": average_in_one_task,
        "func_args": {
            "metric_key_list": ENGLISH_MTBENCH_MATH
        },
        "target": {
            "task_key": "swallow|english_mt_bench|0"
        },
    },
    {
        "display_name": "english_mtbench_reasoning",
        "func": average_in_one_task,
        "func_args": {
            "metric_key_list": ENGLISH_MTBENCH_REASONING
        },
        "target": {
            "task_key": "swallow|english_mt_bench|0"
        },
    },
    {
        "display_name": "english_mtbench_roleplay",
        "func": average_in_one_task,
        "func_args": {
            "metric_key_list": ENGLISH_MTBENCH_ROLEPLAY
        },
        "target": {
            "task_key": "swallow|english_mt_bench|0"
        },
    },
    {
        "display_name": "english_mtbench_stem",
        "func": average_in_one_task,
        "func_args": {
            "metric_key_list": ENGLISH_MTBENCH_STEM
        },
        "target": {
            "task_key": "swallow|english_mt_bench|0"
        },
    },
    {
        "display_name": "english_mtbench_writing",
        "func": average_in_one_task,
        "func_args": {
            "metric_key_list": ENGLISH_MTBENCH_WRITING
        },
        "target": {
            "task_key": "swallow|english_mt_bench|0"
        },
    },
    {
        "display_name": "english_mtbench_first_turn",
        "func": average_in_one_task,
        "func_args": {
            "metric_key_list": ENGLISH_MTBENCH_FIRST_TURN
        },
        "target": {
            "task_key": "swallow|english_mt_bench|0"
        },
    },
    {
        "display_name": "english_mtbench_second_turn",
        "func": average_in_one_task,
        "func_args": {
            "metric_key_list": ENGLISH_MTBENCH_SECOND_TURN
        },
        "target": {
            "task_key": "swallow|english_mt_bench|0"
        },
    },
    {
        "display_name": "english_mtbench_average",
        "func": average_in_one_task,
        "func_args": {
            "metric_key_list": ENGLISH_MTBENCH_ALL
        },
        "target": {
            "task_key": "swallow|english_mt_bench|0"
        },
    },
]
