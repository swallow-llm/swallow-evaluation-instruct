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

import re
import logging
import ast

import numpy as np
from markdown import markdown
from bs4 import BeautifulSoup

from lighteval.tasks.lighteval_task import LightevalTaskConfig
from lighteval.tasks.requests import Doc
from lighteval.metrics.metrics_sample import JudgeLLMMTBenchSwallow
from lighteval.metrics.utils.metric_utils import (
    SampleLevelMetricGrouping,
    MetricCategory,
    MetricUseCase,
)
from lighteval.tasks.swallow.japanese_mt_bench.judge_prompt_templates import (
    gpt_judge_prompt_mt_bench_for_single_v1,
    gpt_judge_prompt_mt_bench_for_single_v1_with_ref,
    gpt_judge_prompt_mt_bench_for_single_v1_multi_turn,
    gpt_judge_prompt_mt_bench_for_single_v1_with_ref_multi_turn,
)


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


def make_mt_bench_prompt(max_gen_text_length: int = 8192):
    """
    generator for mt_bench_prompt variant with custom max_gen_text_length.
    """
    def mt_bench_prompt_variant(line, task_name: str = ""):
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
                "max_gen_text_length": max_gen_text_length,  # variantで指定
            },
        )
    return mt_bench_prompt_variant

# max_gen_text_length=8192にセットするデフォルトのプロンプト関数
mt_bench_prompt = make_mt_bench_prompt(8192)
# max_gen_text_length=6144にセットする，コンテキスト長が短いモデル(e.g., llm-jp-3.1)向けのプロンプト関数
mt_bench_prompt_truncate_6144 = make_mt_bench_prompt(6144)

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


def mt_bench_corpus_level_fn(score_list: list[float]) -> float:
    score_list = [score for score in score_list if score != -1]
    if not score_list:
        raise ValueError("No valid scores found in the list.")
    return np.mean(score_list) / 10


# Markdownからテキスト部分を抜き出す関数
def extract_plain_text(markdown_text: str) -> str:
    # MarkdownをHTMLに変換
    html = markdown(markdown_text)
    # HTMLから純粋なテキストを抽出
    soup = BeautifulSoup(html, "html.parser")
    plain_text = soup.get_text()
    # テキストの文字数をカウント
    return plain_text


# 日本語文字か否かを判定する関数
def is_japanese_char(ch):
    # Hiragana
    if "\u3041" <= ch <= "\u309f":
        return True
    # Katakana
    if "\u30a1" <= ch <= "\u30ff":
        return True
    # Kanji (CJK Unified Ideographs excl. Extenson A-I)
    if "\u4e00" <= ch <= "\u9fff":
        return True
    # Full-width characters
    if "\uff01" <= ch <= "\uff5e":
        return True
    # Japanese punctuation and symbols
    # https://qiita.com/YusukeHirao/items/099ab93bdbf47f0d7a02
    if ("\u3000" <= ch <= "\u3036") or (ch == "\u30fb") or (ch == "\uff5e"):
        return True
    return False


# テキスト中の日本語文字の割合を計算する関数
def count_japanese_chars(text: str) -> int:
    num_japanese_chars = sum(1 for ch in text if is_japanese_char(ch))
    return num_japanese_chars


def safe_divide(numerator: int, denominator: int) -> float:
    if denominator == 0:
        return 0.0
    else:
        return numerator / denominator


def compute_japanese_ratio_sample(
    sample_ids: list[str], responses: list, formatted_docs: list[Doc], **kwargs
) -> dict[str, int]:
    metrics = []
    for sample_idx in range(len(sample_ids)):
        category = formatted_docs[sample_idx].specific["category"]
        # 1ターン目と2ターン目は区別しないで連結する
        prediction = [
            (responses[sample_idx][0].result[0][turn], responses[sample_idx][0].result[1][turn])
            for turn in range(len(responses[sample_idx][0].result[0]))
        ]
        metrics.append(
            {
                f"japanese_ratio_{category}": prediction,
                "japanese_ratio_overall": prediction,
            }
        )
    return metrics


def compute_japanese_ratio_corpus(prediction_list: list[list[tuple[str]]]) -> dict[str, float]:
    total_num_chars = 0
    total_num_ja_chars = 0
    for prediction in prediction_list:
        for first_turn, second_turn in prediction:
            plain_text = extract_plain_text(first_turn + second_turn)
            num_chars = len(plain_text)
            num_ja_chars = count_japanese_chars(plain_text)
            total_num_chars += num_chars
            total_num_ja_chars += num_ja_chars

    return safe_divide(total_num_ja_chars, total_num_chars)


japanese_character_ratio_metric = SampleLevelMetricGrouping(
    metric_name=[f"japanese_ratio_{category}" for category in ["overall"] + CATEGORIRES],
    higher_is_better={f"japanese_ratio_{category}": True for category in ["overall"] + CATEGORIRES},
    category=MetricCategory.LLM_AS_JUDGE_MULTI_TURN,
    use_case=MetricUseCase.SUMMARIZATION,
    sample_level_fn=compute_japanese_ratio_sample,
    corpus_level_fn=compute_japanese_ratio_corpus,
)

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
    corpus_level_fn={
        f"judge_score_{category}_turn_1_avg": mt_bench_corpus_level_fn for category in ["overall"] + CATEGORIRES
    }
    | {f"judge_score_{category}_turn_2_avg": mt_bench_corpus_level_fn for category in ["overall"] + CATEGORIRES},
)

mt_bench_japanese_swallow_gpt4o = LightevalTaskConfig(
    name="japanese_mt_bench",
    prompt_function=mt_bench_prompt,
    suite=["swallow"],
    hf_repo="tokyotech-llm/swallow_japanese_mt_bench",
    hf_subset="default",
    hf_avail_splits=["train"],
    evaluation_splits=["train"],
    few_shots_split="",
    few_shots_select="random",
    metric=[llm_judge_mt_bench_swallow_gpt4o_judge, japanese_character_ratio_metric],
    stop_sequence=[],
)

# max_gen_text_lengthを6144に減らした，コンテキスト長が短いモデル(e.g., llm-jp-3.1)向けの派生版
mt_bench_japanese_swallow_gpt4o_truncate_6144 = LightevalTaskConfig(
    name="japanese_mt_bench_truncate_6144",
    prompt_function=mt_bench_prompt_truncate_6144,
    suite=["swallow"],
    hf_repo="tokyotech-llm/swallow_japanese_mt_bench",
    hf_subset="default",
    hf_avail_splits=["train"],
    evaluation_splits=["train"],
    few_shots_split="",
    few_shots_select="random",
    metric=[llm_judge_mt_bench_swallow_gpt4o_judge, japanese_character_ratio_metric],
    stop_sequence=[],
)

TASKS_TABLE = [mt_bench_japanese_swallow_gpt4o, mt_bench_japanese_swallow_gpt4o_truncate_6144]
