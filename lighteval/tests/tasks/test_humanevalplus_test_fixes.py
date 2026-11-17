from lighteval.tasks.extended.lcb.codegen_metrics import grade_check_fn
from lighteval.tasks.swallow.humaneval_series import swallow_humanevalplus_prompt_fn


# A minimal HumanEval/32-like problem whose test field has the same wrong assertion as evalplus/humanevalplus.
PROMPT = '''import math


def _poly(xs: list, x: float):
    return sum(coeff * math.pow(x, i) for i, coeff in enumerate(xs))


def find_zero(xs: list):
'''
SOLUTION = '''    lo, hi = -10.0, 10.0
    for _ in range(200):
        mid = (lo + hi) / 2
        if _poly(xs, lo) * _poly(xs, mid) <= 0:
            hi = mid
        else:
            lo = mid
    return lo
'''
WRONG_TEST = '''def check(candidate):
    for inp in [[[1, 2]], [[-6, 11, -6, 1]]]:
        assert _poly(*candidate(*inp), inp) <= 0.0001
'''


def make_line(task_id):
    return {"prompt": PROMPT, "entry_point": "find_zero", "task_id": task_id, "test": WRONG_TEST}


def test_humanevalplus_32_test_is_fixed():
    doc = swallow_humanevalplus_prompt_fn(make_line("HumanEval/32"), "humanevalplus")
    assert "assert abs(_poly(*inp, candidate(*inp))) <= 0.0001" in doc.specific["check_fn"]
    assert grade_check_fn(PROMPT + SOLUTION, "find_zero", doc.specific["check_fn"], timeout=6) == [True]


def test_wrong_test_fails_a_correct_solution():
    # the unpatched assertion unpacks a float, so even a correct solution fails
    assert grade_check_fn(PROMPT + SOLUTION, "find_zero", WRONG_TEST, timeout=6) != [True]


def test_other_tasks_are_unchanged():
    doc = swallow_humanevalplus_prompt_fn(make_line("HumanEval/0"), "humanevalplus")
    assert doc.specific["check_fn"] == WRONG_TEST
