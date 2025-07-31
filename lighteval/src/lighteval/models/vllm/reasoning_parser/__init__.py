
from vllm.reasoning import ReasoningParserManager
from .string_based_markup_parser import StringBasedMarkupReasoningParser

# How to use
# ここには，文字列のマークアップのみに依存する StringBasedMarkupReasoningParser クラスを継承したRasoningParserクラスを定義します．
# 定義の方法は以下の通り． 
# @ReasoningParserManager.register_module(): parserのID．reasoning_parser引数で指定するIDとして使われます
# super().__init__(): think_start_exprには推論過程開始文字列，response_start_exprには最終出力開始文字列を定義します．  
# たとえばDeepSeek-R1形式ならば think_start_expr="<think>", response_start_expr="</think>" となります．

# deepseek_r1_markup用のパーサを登録
@ReasoningParserManager.register_module("deepseek_r1_markup")
class DeepseekR1MarkupParser(StringBasedMarkupReasoningParser):
    def __init__(self, tokenizer = None):
        super().__init__(think_start_expr="<think>", response_start_expr="</think>")
