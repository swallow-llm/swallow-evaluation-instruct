import ast
import pycountry

from lighteval.tasks.lighteval_task import LightevalTaskConfig
from lighteval.metrics.metrics import Metrics
from lighteval.tasks.requests import Doc
from lighteval.utils.utils import as_list

from .metrics_translation_japanese import bleu_ja, chrf_ja, ter_ja

# 英日翻訳用プロンプト
LIGHTEVAL_MACHINE_TRANSLATION_ENJA_QUERY_TEMPLATE = """
English phrase: {source_text}\n
Japanese phrase:
""".strip()

# 日英翻訳用プロンプト
LIGHTEVAL_MACHINE_TRANSLATION_JAEN_QUERY_TEMPLATE = """
Japanese phrase: {source_text}\n
English phrase:
""".strip()

def wmt_enja(line, task_name: str = None):
    # nested object は ast.literal_eval で decode する必要がある
    if isinstance(line["translation"], str):
        line["translation"] = ast.literal_eval(line["translation"])
    # 参照訳については1番目のみ取得（WMT20 En-Ja/Ja-Enはもともと1つしかない）
    for k, v in line["translation"].items():
        line["translation"][k] = as_list(v)[0]

    query = LIGHTEVAL_MACHINE_TRANSLATION_ENJA_QUERY_TEMPLATE.format(source_text=line["translation"]["en"])

    return Doc(
        task_name=task_name,
        query=query,
        gold_index=0,
        choices=[line["translation"]["ja"].rstrip()],
    )

def wmt_jaen(line, task_name: str = None):
    if isinstance(line["translation"], str):
        line["translation"] = ast.literal_eval(line["translation"])
    for k, v in line["translation"].items():
        line["translation"][k] = as_list(v)[0]

    query = LIGHTEVAL_MACHINE_TRANSLATION_JAEN_QUERY_TEMPLATE.format(source_text=line["translation"]["ja"])

    return Doc(
        task_name=task_name,
        query=query,
        gold_index=0,
        choices=[line["translation"]["en"].rstrip()],
    )


wmt20_en_ja_swallow = LightevalTaskConfig(
    name="wmt20:en-ja",
    suite=["swallow"],
    prompt_function=wmt_enja,
    hf_repo="lighteval/sacrebleu_manual",
    hf_subset="wmt20_en-ja",
    hf_avail_splits=["test"],
    evaluation_splits=["test"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=None,
    metric=[bleu_ja, chrf_ja, ter_ja],
    stop_sequence=["\n"],
    trust_dataset=True,
    version=0,
)

wmt20_ja_en_swallow = LightevalTaskConfig(
    name="wmt20:ja-en",
    suite=["swallow"],
    prompt_function=wmt_jaen,
    hf_repo="lighteval/sacrebleu_manual",
    hf_subset="wmt20_ja-en",
    hf_avail_splits=["test"],
    evaluation_splits=["test"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=None,
    metric=[Metrics.bleu, Metrics.chrf, Metrics.ter],
    stop_sequence=["\n"],
    trust_dataset=True,
    version=0,
)