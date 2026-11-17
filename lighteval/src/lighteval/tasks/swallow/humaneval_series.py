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

# HumanEval+ の test フィールドの誤りを evalplus 本体の判定に合わせて修正する．
# HumanEval/32 (find_zero): test フィールドの `_poly(*candidate(*inp), inp)` は引数が逆で，
# 根 (float) を * で展開しようとして TypeError になるため，canonical solution を含むすべての解答が不合格になる．
# evalplus 本体の評価器は find_zero について `abs(_poly(*inp, out)) <= atol` を判定するので，それに置き換える．
# データセット側で修正された場合は置換対象の文字列が見つからず，何もしない．
HUMANEVALPLUS_TEST_FIXES = {
    "HumanEval/32": (
        "assert _poly(*candidate(*inp), inp) <= 0.0001",
        "assert abs(_poly(*inp, candidate(*inp))) <= 0.0001",
    ),
}


def fix_humanevalplus_test(task_id: str, test: str) -> str:
    if task_id not in HUMANEVALPLUS_TEST_FIXES:
        return test
    wrong, fixed = HUMANEVALPLUS_TEST_FIXES[task_id]
    return test.replace(wrong, fixed)


def swallow_humanevalplus_prompt_fn(line, task_name: str = "humaneval") -> Doc:
    doc = swallow_humaneval_prompt_fn(line=line, task_name=task_name)
    doc.specific["check_fn"] = fix_humanevalplus_test(line["task_id"], doc.specific["check_fn"])
    # HumanEval+はHumanEvalの約100倍のunittestがあるので，default timeout (=6 sec.)の20倍 = 120 sec.にする．
    # 600 sec. にすると時間がかかりすぎる
    doc.specific["timeout"] = 120
    return doc

# Task config
# metric は jhumaneval.py で登録済みの "humaneval_pass_{1,10}" を使用する．
# Doc.specific.timeout で単体テストのtimeoutを指定できる．
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

humanevalplus = LightevalTaskConfig(
    name="humanevalplus",
    prompt_function=swallow_humanevalplus_prompt_fn,
    suite=["swallow"],
    hf_repo="evalplus/humanevalplus",
    hf_subset=None,
    hf_avail_splits=["test"],
    evaluation_splits=["test"],
    trust_dataset=True,
    stop_sequence=[],
    metric=[Metrics.humaneval_pass_1, Metrics.humaneval_pass_10],
    version=0,
)