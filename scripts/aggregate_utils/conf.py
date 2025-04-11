from .white_lists import (
    JMMLU_SOCIAL_SCIENCES, JMMLU_HUMANITIES, JMMLU_STEM, JMMLU_OTHERS
)
from .funcs import (
    pick, micro_average
)

# Aggregate Config
# - Each entry must have 'display_name', 'func', and 'target'. 'target' must have 'task_key' and 'is_category'.
# - Each entry may have 'func_args' matching its 'func'.
AGGREGATE_CONF = [
    {
        'display_name': 'jmmlu_social_sciences', 
        'func': micro_average, 
        'func_args': {'metric_key': 'extractive_match', 'white_list': JMMLU_SOCIAL_SCIENCES}, 
        'target': {'task_key': 'swallow|swallow_jmmlu|0', 'is_category': True}
    },
    {
        'display_name': 'jmmlu_humanities', 
        'func': micro_average, 
        'func_args': {'metric_key': 'extractive_match', 'white_list': JMMLU_HUMANITIES}, 
        'target': {'task_key': 'swallow|swallow_jmmlu|0', 'is_category': True}
    },
    {
        'display_name': 'jmmlu_stem', 
        'func': micro_average, 
        'func_args': {'metric_key': 'extractive_match', 'white_list': JMMLU_STEM}, 
        'target': {'task_key': 'swallow|swallow_jmmlu|0', 'is_category': True}
    },
    {
        'display_name': 'jmmlu_other', 
        'func': micro_average, 
        'func_args': {'metric_key': 'extractive_match', 'white_list': JMMLU_OTHERS}, 
        'target': {'task_key': 'swallow|swallow_jmmlu|0', 'is_category': True}
    },
    {
        'display_name': 'jmmlu', 
        'func': micro_average, 
        'func_args': {'metric_key': 'extractive_match'}, 
        'target': {'task_key': 'swallow|swallow_jmmlu|0', 'is_category': True}
    },
]