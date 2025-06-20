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
    # 指標名を Pass@1:10 （=10回試行で推定したPass@1）に変更した
    {
        "display_name": "livecodebench_v5_v6",
        "func": pick,
        "func_args": {
            'metric_key': 'codegen_pass@1:10'
        }, 
        'target': {
            'task_key': 'swallow|lcb:codegeneration_v5_v6|0'
        }
    },
    # ToDo: MMLU-Pro, MMLU の追加
]
