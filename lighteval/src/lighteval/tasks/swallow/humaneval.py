from lighteval.metrics.metrics import Metrics
from lighteval.tasks.lighteval_task import Doc, LightevalTaskConfig


# Query template
SWALLOW_HUMANEVAL_QUERY_TEMPLATE = """
Complete the Python starter code below. Implement the missing parts so the behavior matches the docstring's specification and examples and passes all tests. Keep all given function and variable names and signatures unchanged.

Respond with ONE triple-backticked code block containing the FULL final code and nothing else.

Starter code:
```python
{code_snipet}
```

""".lstrip() # JHumanEvalに倣って，末尾の2重改行を残す


def swallow_humaneval_prompt_fn(line, task_name: str = "humaneval") -> Doc:
    # 1. get the prompt from the line
    query = SWALLOW_HUMANEVAL_QUERY_TEMPLATE.format(code_snipet=line['prompt'])

    # 2. return the doc with the same format as lcb
    return Doc(
        task_name=task_name,
        query=query,
        choices=[""],
        gold_index=0,
        specific={
            "fn_name": line['entry_point'],
            "task_id": line['task_id'],
            "check_fn": line['test'],
        },
    )

# Task config
# metric は jhumaneval.py で登録済みの "humaneval_pass_{1,10}" を使用する．
humaneval = LightevalTaskConfig(
    name="humaneval",
    prompt_function=swallow_humaneval_prompt_fn,
    suite=["swallow"],
    hf_repo="openai/openai_humaneval",
    hf_subset=None,
    hf_avail_splits=["test"],
    evaluation_splits=["test"],
    trust_dataset=True,
    stop_sequence=[],
    metric=[Metrics.humaneval_pass_1, Metrics.humaneval_pass_10],
    version=0,
)   