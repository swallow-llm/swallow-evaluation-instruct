# ベンチマークの追加について
# 1. ./swallow/*.py を作成して LightevalTaskConfig のインスタンスを定義してください．
# Config の suite に "swallow" を含めてください．
# 2. # ./swallow/*.py から LightevalTaskConfig のインスタンスを import して SWALLOW_TASKS に追加してください．
# (配列の場合は SWALLOW_TASKS.extend() で追加してください)

# 以上の設定を行うと lighteval 実行時引数で `swallow|{ベンチマーク名}` が使えるようになります． `--custom-tasks` の指定は不要です．

from .mclm_MATH_100_japanese import math_100_japanese
from .jmmlu import JMMLU_SUBSET_TASKS
from .japanese_mt_bench.main import mt_bench_swallow_gpt4o
from .hellaswag import hellaswag_generative
from .jemhopqa import jemhopqa, jemhopqa_cot
from .gpqa_ja import gpqa_ja_instruct_lighteval
from .wmt20 import wmt20_en_ja_swallow, wmt20_ja_en_swallow

SWALLOW_TASKS = [
    math_100_japanese,
    hellaswag_generative,
    gpqa_ja_instruct_lighteval,
    jemhopqa,
    jemhopqa_cot,
    mt_bench_swallow_gpt4o,
    wmt20_en_ja_swallow,
    wmt20_ja_en_swallow
]
SWALLOW_TASKS.extend(JMMLU_SUBSET_TASKS)
