# This file is copied from lighteval/src/lighteval/tasks/extended/mt_bench/main.py and adapted to Swallow LLM

# MIT License

# Copyright (c) 2024 The HuggingFace Team
# Copyright (c) 2025 Swallow LLM

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

# ruff: noqa: F405, F403, F401, I001
from lighteval.tasks.lighteval_task import LightevalTaskConfig
from lighteval.tasks.requests import Doc
from lighteval.metrics.metrics_sample import JudgeLLMMTBenchSwallow
from lighteval.metrics.utils.metric_utils import SampleLevelMetricGrouping, MetricCategory, MetricUseCase
from lighteval.tasks.swallow.japanese_mt_bench.judge_prompt_templates import (
    gpt_judge_prompt_mt_bench_for_single_v1,
    gpt_judge_prompt_mt_bench_for_single_v1_with_ref,
    gpt_judge_prompt_mt_bench_for_single_v1_multi_turn,
    gpt_judge_prompt_mt_bench_for_single_v1_with_ref_multi_turn,
)
import re
import numpy as np
import logging
import ast


logger = logging.getLogger(__name__)

CATEGORIRES = ["coding", "extraction", "humanities", "math", "reasoning", "roleplay", "stem", "writing"]
NEED_REF_CATEGORIRES = ["math", "reasoning", "coding"]
CATEGORY_TEMPERATURE_MAP = {
    "writing": 0.7,
    "roleplay": 0.7,
    "extraction": 0.0,
    "math": 0.0,
    "coding": 0.0,
    "reasoning": 0.0,
    "stem": 0.1,
    "humanities": 0.1,
    "arena-hard-200": 0.0,
}
NUM_SAMPLES = 5


def mt_bench_prompt(line, task_name: str = ""):
    return Doc(
        task_name=task_name,
        query=f"{line['turns'][0]}",
        choices=[],
        instruction=None,
        gold_index=[],
        specific={
            "reference": line["reference"],
            "category": line["category"],
            "multi_turn_queries": line["turns"],
            "id": line["question_id"],
            "num_samples": NUM_SAMPLES,
            "temperature": CATEGORY_TEMPERATURE_MAP.get(line["category"], 0.0),
        },
    )


def process_judge_response_gpt(x):
    score_pattern = re.compile("\[\[(\d+\.?\d*)\]\]")
    score_pattern_backup = re.compile("\[(\d+\.?\d*)\]")
    match = re.search(score_pattern, x)
    if not match:
        match = re.search(score_pattern_backup, x)
    if match:
        rating = ast.literal_eval(match.groups()[0])
    else:
        logger.error("Error processing judge response")
        rating = -1
    return rating


def gpt_judge_mt_bench_prompt(question, answer, options, gold):
    if type(question) is tuple:
        if options not in NEED_REF_CATEGORIRES:
            return gpt_judge_prompt_mt_bench_for_single_v1_multi_turn(question, options, answer, gold)
        else:
            return gpt_judge_prompt_mt_bench_for_single_v1_with_ref_multi_turn(question, options, answer, gold)
    elif type(question) is str:
        if options not in NEED_REF_CATEGORIRES:
            return gpt_judge_prompt_mt_bench_for_single_v1(question, options, answer, gold)
        else:
            return gpt_judge_prompt_mt_bench_for_single_v1_with_ref(question, options, answer, gold)
    else:
        raise ValueError(f"Unsupported question type: {type(question)}")


llm_judge_mt_bench_swallow_gpt4o_judge = SampleLevelMetricGrouping(
    metric_name=[f"judge_score_{category}_turn_1" for category in ["overall"] + CATEGORIRES]
    + [f"judge_score_{category}_turn_2" for category in ["overall"] + CATEGORIRES],
    higher_is_better={f"judge_score_{category}_turn_1": True for category in ["overall"] + CATEGORIRES}
    | {f"judge_score_{category}_turn_2": True for category in ["overall"] + CATEGORIRES},
    category=MetricCategory.LLM_AS_JUDGE_MULTI_TURN,
    use_case=MetricUseCase.SUMMARIZATION,
    sample_level_fn=JudgeLLMMTBenchSwallow(
        judge_model_name="gpt-4o-2024-08-06",
        template=gpt_judge_mt_bench_prompt,
        process_judge_response=process_judge_response_gpt,
        judge_backend="openai",
        short_judge_name="gpt-4o",
    ).compute,
    corpus_level_fn={f"judge_score_{category}_turn_1_avg": np.mean for category in ["overall"] + CATEGORIRES}
    | {f"judge_score_{category}_turn_2_avg": np.mean for category in ["overall"] + CATEGORIRES},
)

mt_bench_swallow_gpt4o = LightevalTaskConfig(
    name="japanese_mt_bench",
    prompt_function=mt_bench_prompt,  # must be defined in the file or imported from src/lighteval/tasks/tasks_prompt_formatting.py
    suite=["swallow"],
    hf_repo="tokyotech-llm/swallow_japanese_mt_bench",
    hf_subset="default",
    hf_avail_splits=["train"],
    evaluation_splits=["train"],
    few_shots_split="",
    few_shots_select="random",
    metric=[llm_judge_mt_bench_swallow_gpt4o_judge],
    generation_size=32768,
    stop_sequence=[],
)


TASKS_TABLE = [mt_bench_swallow_gpt4o]
