import ast
import pycountry

from lighteval.tasks.lighteval_task import LightevalTaskConfig
from lighteval.metrics.metrics import Metrics
from lighteval.tasks.requests import Doc
from lighteval.utils.utils import as_list

from .metrics_translation_japanese import bleu_ja, chrf_ja, ter_ja, bleu_en, chrf_en, ter_en, bleu_ja_nagisa, chrf_ja_nagisa, ter_ja_nagisa

# 英日翻訳用プロンプト
# 翻訳文の接頭辞は `日本語:` とする
MACHINE_TRANSLATION_ENJA_QUERY_TEMPLATE = """
以下に示す英文を日本語に翻訳せよ。
翻訳文の文体は、常体（だ、である）を用いること。
翻訳文を出力するときは、改行してから `日本語: 翻訳文` という形式で出力すること。

英語: {source_text}
""".lstrip()

# 日英翻訳用プロンプト
# 翻訳文の接頭辞は `English:` とする
MACHINE_TRANSLATION_JAEN_QUERY_TEMPLATE = """
Translate the following Japanese sentence into English.  
On a new line, output the result in this exact format: `English: <your translation>`

Japanese: {source_text}
""".strip()

def wmt_enja(line, task_name: str = None):
    # nested object は ast.literal_eval で decode する必要がある
    if isinstance(line["translation"], str):
        line["translation"] = ast.literal_eval(line["translation"])
    # 参照訳については1番目のみ取得（WMT20 En-Ja/Ja-Enはもともと1つしかない）
    for k, v in line["translation"].items():
        line["translation"][k] = as_list(v)[0]

    query = MACHINE_TRANSLATION_ENJA_QUERY_TEMPLATE.format(source_text=line["translation"]["en"])

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

    query = MACHINE_TRANSLATION_JAEN_QUERY_TEMPLATE.format(source_text=line["translation"]["ja"])

    return Doc(
        task_name=task_name,
        query=query,
        gold_index=0,
        choices=[line["translation"]["en"].rstrip()],
    )


wmt20_enja_swallow = LightevalTaskConfig(
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
    metric=[bleu_ja, bleu_ja_nagisa],
    stop_sequence=[],
    trust_dataset=True,
    version=0,
)

wmt20_jaen_swallow = LightevalTaskConfig(
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
    metric=[bleu_en],
    stop_sequence=[],
    trust_dataset=True,
    version=0,
)