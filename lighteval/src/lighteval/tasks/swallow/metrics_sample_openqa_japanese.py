import logging
import re
from typing import List, Optional, Callable, Literal, Dict, Any
from collections import Counter
import itertools

import neologdn
from fuzzywuzzy import fuzz

from lighteval.tasks.requests import Doc

logger = logging.getLogger(__name__)


def compute_char_f1(str_gold, str_pred) -> float:
    """
    標準的な文字F1指標．
    
    実装はJGLUE (出典は TransformersのSQuAD向けF1スコア) に基づく．
    https://github.com/yahoojapan/JGLUE/blob/release/v1.3.0/fine-tuning/patch/transformers-4.9.2_jglue-1.3.0.patch
    https://github.com/huggingface/transformers/blob/v4.9.2/src/transformers/data/metrics/squad_metrics.py

    Args:
        str_gold (_type_): 回答文字列
        str_pred (_type_): 正解文字列

    Returns:
        float: 文字F1
    """
    gold_toks = list(str_gold)
    pred_toks = list(str_pred)
    common = Counter(gold_toks) & Counter(pred_toks)
    num_same = sum(common.values())         
    
    if len(gold_toks) == 0 or len(pred_toks) == 0:
        # If either is no-answer, then F1 is 1 if they agree, 0 otherwise
        return int(gold_toks == pred_toks)
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(pred_toks)
    recall = 1.0 * num_same / len(gold_toks)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1


def compute_char_f1_llmjpeval(str_gold, str_pred) -> float:
    """
    normalized indel similarity に基づく文字F1指標．
       
    実装は llm-jp-eval (出典は fuzzywuzzy -> python-levenshtein) に基づく．
    https://github.com/llm-jp/llm-jp-eval/blob/v1.3.1/src/llm_jp_eval/utils.py#L154
    https://rapidfuzz.github.io/Levenshtein/levenshtein.html#Levenshtein.ratio

    Args:
        str_gold (_type_): 回答文字列
        str_pred (_type_): 正解文字列

    Returns:
        float: normalized indel similarity に基づく文字F1
    """    
    return fuzz.token_sort_ratio(str_pred, str_gold) / 100

def compute_match(str_gold, str_pred) -> float:
    return 1.0 if str_gold == str_pred else 0.0

def _regex_extractor(obj_regex, text: str, 
                     match_group_name: str,
                     extraction_mode: Literal["first_match", "last_match", "any_match"]) -> List[str]:
    """
    正規表現とテキストを渡すと extraction mode に従って抽出結果を返す関数．
    - extraction_mode="first_match": 最初のマッチのみを返す
    - extraction_mode="last_match": 最後のマッチのみを返す
    - extraction_mode="any_match": すべてのマッチを返す    
    """
    
    lst_matches_with_positions = [(match.group(match_group_name), match.start(), match.end()) for match in obj_regex.finditer(text)]
    # 出現箇所の昇順にソート．つまり最後に出現したものが配列の最後に入っている
    lst_matches_with_positions = sorted(lst_matches_with_positions, key=lambda x: (x[2], -x[1]), reverse=False)
    
    if len(lst_matches_with_positions) == 0:
        return []
    else:
        if extraction_mode == "first_match":
            return [lst_matches_with_positions[0][0]]
        elif extraction_mode == "last_match":
            return [lst_matches_with_positions[-1][0]]
        elif extraction_mode == "any_match":
            return [match[0] for match in lst_matches_with_positions]

def _boxed_match_extraction_function(text: str, extraction_mode: Literal["first_match", "last_match", "any_match"]) -> List[str]:
    """
    \boxed{あいうえお} や \boxed{\text{YES}} から，中括弧の中の文字列を抽出する正規表現
    """
    str_boxed_match_regex = (
        r'\\boxed\{'           # \boxed{ のリテラル
        r'(\\text\{)?'         #      ①「\text{」があればキャプチャ (GROUP 1)
        r'(?P<content>.+?)'    #      ② 中身を最短マッチで取る（Named Group）
        r'(?(1)\})'            #      ① が取れていたら「}」を１つ消費
        r'\}'                  # 外側の「}」
    )
    boxed_match_regex = re.compile(str_boxed_match_regex, re.DOTALL|re.UNICODE)
    
    return _regex_extractor(obj_regex=boxed_match_regex, match_group_name="content",
                            text=text, extraction_mode=extraction_mode)

# ToDo: prefixなしのvariantを実装
def _free_form_answer_extraction_function(text: str, extraction_mode: Literal["first_match", "last_match", "any_match"]) -> List[str]:
    """
    自由記述形式の応答文から回答スパンを抽出する正規表現。対応している応答文の例は以下の通り。
    正解: {回答スパン}, 正解は{回答スパン}。, 答えは{回答スパン}\n, ...
    """
    
    str_answer_match = r"""(?x)
    (?:                            # 前方一致
        (?:回答|正解|答え|解答)は    # 〇〇は...
      |                                           # または
        (?:回答|正解|答え|解答|解|答|【回答】)\s*    # 回答{プレースホルダ}〇〇
        (?:[:：]|→|->|=|＝|[．\.])\s*              # 区切り記号
    )
    (?P<answer>.+?)                               
    (?:です|だ|である|。|、|$|．|，|\s|\n)          # 後方一致
    """
    answer_regex = re.compile(str_answer_match, re.DOTALL | re.UNICODE)

    return _regex_extractor(obj_regex=answer_regex, match_group_name="answer",
                            text=text, extraction_mode=extraction_mode)
       
def _neologd_normalize(text: str) -> str:
    """
    neologdn package を使って文字列を正規化する．表記揺れを軽減する効果がある．
    quasi_exact_match として利用するとよい．
    """
    text = neologdn.normalize(text)
    return text


def _remove_paren_and_quote(text: str) -> str:
    """
    先頭と末尾が対応する括弧・引用符・強調記号で囲まれていたら外す．
    """
    
    text = text.strip()
    paren_pairs = {
        # 括弧
        '(': ')','（':'）','{': '}','<': '>',
        '「': '」', '『': '』','〈': '〉', '《': '》',
        '【': '】', '〖': '〗',
        '〔': '〕', '❨': '❩', '❪': '❫',
        '❴': '❵', '❬': '❭', '❰': '❱',
        '⦅': '⦆', '⦇': '⦈',
        '⟦': '⟧', '⟨': '⟩',

        # 引用符
        '"': '"',  "'": "'",  '`': '`',
        '“': '”',  '‘': '’',
        '«': '»',  '‹': '›',
        '„': '„',  '‚': '‚',
        '‟': '‟',  '‛': '‛',
        '〝': '〞', '〟': '〟',
        '＂': '＂', '＇': '＇',       
        
        # 強調
        '**':'**', '*':'*', '__':'__', '``':'``','```':'```',
    }
    # 文字数の多い順にテストしていく
    lst_paren_pairs = sorted(paren_pairs.items(), key=lambda tup: len(tup[0]), reverse=True)

    if len(text) <= 2:
        return text
    for str_open, str_close in lst_paren_pairs:
        if text.startswith(str_open) and text.endswith(str_close):
            return text[len(str_open):-len(str_close)]
    return text

def _canonicalize_binary_response(text: str) -> str:
    """
    肯定または否定のいずれかによる回答を "YES", "NO" に正準化する関数
    いずれにも該当しないものはそのまま返す
    """
    canonical_responses = {
        "YES": "はい,肯定,肯,◯,TRUE,OK,イエス".split(","),
        "NO": "いいえ,否,×,否定,ちがう,違う,FALSE,NG".split(",")
    }
    for canonical_response, lst_variants in canonical_responses.items():
        if text in lst_variants:
            return canonical_response
    
    return text

class JapaneseOpenQAExtractor(object):
    def __init__(
        self,
        use_boxed_match_extraction: bool = True,
        use_free_form_answer_extraction: bool = True,
        extraction_fallback_function: Optional[Callable[[str], List[str]]] = lambda text: [text],
        neologdn_normalize: bool = False,
        remove_paren_and_quote: bool = False,
        canonicalize_binary_response: bool = False,
        lowercase: bool = False,
        boxed_match_extraction_mode: Literal["first_match", "last_match", "any_match"] = "first_match",
        free_form_answer_match_extraction_mode: Literal["first_match", "last_match", "any_match"] = "last_match",
    ):
        """
        自由記述形式の設問に対する応答から回答スパンを抽出して正規化するクラス．  
        例：
        extractor = JapaneseQAExtractor()
        extractor("回答は東京都です。") == "東京都"
        
        回答スパンの抽出および正規化は，オプションで ON/OFF を変更できる．オプションは以下の通りで，上から順に適用される:
        以下，回答文字列を `ANSWER` としてオプションを説明する．
          1. boxed_match_extraction: \boxed{ANSWER} から回答を抽出
          2. free_form_answer_extraction: "回答：ANSWER" のような表現から回答を抽出
          3. extraction_fallback_function: 上記2つの抽出に失敗した場合に適用する回答スパン抽出関数．デフォルトは応答文全体を回答とみなす．
          4. neologdn_normalize: [neologdn](https://github.com/ikegami-yukino/neologdn) に従って回答文字列の表記ゆれを正規化．全角/半角，空白，記号類が正規化される．
          5. remove_paren_and_quote: 回答文字列前後の括弧や引用符を削除．E.g., 「東京都」 -> 東京都
          6. canonicalize_binary_response: 二択問題の回答を "YES", "NO" に正規化．E.g., いいえ -> NO
          7. lowercase: 回答を小文字化．E.g., TOKYO -> tokyo
        """
        self.use_boxed = use_boxed_match_extraction
        self.use_free  = use_free_form_answer_extraction
        self.fallback  = extraction_fallback_function
        self.do_neologd = neologdn_normalize
        self.do_strip  = remove_paren_and_quote
        self.do_bin    = canonicalize_binary_response
        self.do_lower  = lowercase
        self.boxed_match_extraction_mode = boxed_match_extraction_mode
        self.free_form_answer_match_extraction_mode = free_form_answer_match_extraction_mode

    def __call__(
        self,
        text: str
    ) -> List[str]:
        # 1–2. 抽出関数を順に適用
        results = []
        if self.use_boxed:
            results += _boxed_match_extraction_function(text, self.boxed_match_extraction_mode)
        if self.use_free:
            results += _free_form_answer_extraction_function(text, self.free_form_answer_match_extraction_mode)

        # 3. 両方ともヒットしなかったらフォールバック
        if len(results) == 0 and (self.fallback is not None):
            results = self.fallback(text) or []

        # 4–7. 取得した各スパンに対して後処理を適用
        if self.do_strip:
            results = list(map(_remove_paren_and_quote, results))
        if self.do_neologd:
            results = list(map(_neologd_normalize, results))
        if self.do_bin:
            results = list(map(_canonicalize_binary_response, results))
        if self.do_lower:
            results = [r.lower() for r in results]
        
        return results

def _pass_through(text: str) -> List[str]:
    return [text]

default_exact_match_pred_extractor = JapaneseOpenQAExtractor(
    use_boxed_match_extraction=True,
    use_free_form_answer_extraction=True,
    extraction_fallback_function=_pass_through
)

default_quasi_exact_match_pred_extractor = JapaneseOpenQAExtractor(
    use_boxed_match_extraction=True,
    use_free_form_answer_extraction=True,
    extraction_fallback_function=_pass_through,
    neologdn_normalize=True,
    remove_paren_and_quote=True,
    canonicalize_binary_response=True,
    lowercase=False
)

class JapaneseOpenQAExactMatchSamplingFunc(object):
    """
    自由記述回答形式の日本語タスクにおける評価関数をインスタンス化するクラス。
    インスタンス化の際に，回答スパン抽出・正規化・正準化といった前処理を行う JapaneseOpenQAExtractor の設定をカスタマイズできる．
    また抽出された正解および回答はそれぞれ複数個生じうるので，すべての組み合わせに対する集約関数を設定できる．   
    
    前処理は 完全一致用(exact_match)および疑似一致用(quasi_exact_match)の2種類 × 正解文字列用とLLM応答文用 の合計4種類が定義可能で，それぞれカスタマイズできる．
    疑似一致用は正規化や正準化を適用することを想定している．
    前処理のデフォルトは，正解文字列用はパススルー（=何もしない），LLM応答文用は回答スパン抽出・正規化・正準化をひととおり適用するようになっている．
    詳細は default_{quasi_}exact_match_pred_extractor を参照．       
    
    使い方:
      インスタンス化後、sample_level_fnをSampleLevelMetric.sample_level_fnに渡して利用します。
      
    sample_level_fnが返す指標は METRIC_NAMES() で定義されている通り．
      - exact_match: 完全一致 {0, 1}
      - f1_score: 文字F1 [0, 1]
      - llmjpeval_f1_score: llm-jp-eval と互換性のある，normalized indel similarity に基づく文字F1 [0, 1]
      - quasi_exact_match: 疑似一致 {0, 1}
      - f1_score_quasi: 疑似文字F1 [0, 1]
      - llmjpeval_f1_score_quasi: llmjpeval_f1_scoreの疑似版
    """
    
    def __init__(self,
        cfg_exact_match_gold_extractor: Optional[Dict[str, Any]] = None,
        cfg_exact_match_pred_extractor: Optional[Dict[str, Any]] = None,
        cfg_quasi_exact_match_gold_extractor: Optional[Dict[str, Any]] = None,
        cfg_quasi_exact_match_pred_extractor: Optional[Dict[str, Any]] = None,
        instance_level_aggregation_function: Callable[[List[float]], float] = max
    ):        
        if isinstance(cfg_exact_match_gold_extractor, dict):
            self._em_gold_extractor = JapaneseOpenQAExtractor(**cfg_exact_match_gold_extractor)
        else:
            self._em_gold_extractor = _pass_through
        
        if isinstance(cfg_quasi_exact_match_gold_extractor, dict):
            self._quasi_em_gold_extractor = JapaneseOpenQAExtractor(**cfg_quasi_exact_match_gold_extractor)
        else:
            self._quasi_em_gold_extractor = _pass_through
        
        if isinstance(cfg_exact_match_pred_extractor, dict):
            self._em_pred_extractor = JapaneseOpenQAExtractor(**cfg_exact_match_pred_extractor)
        else:
            self._em_pred_extractor = default_exact_match_pred_extractor
        
        if isinstance(cfg_quasi_exact_match_pred_extractor, dict):
            self._quasi_em_pred_extractor = JapaneseOpenQAExtractor(**cfg_quasi_exact_match_pred_extractor)
        else:
            self._quasi_em_pred_extractor = default_quasi_exact_match_pred_extractor
            
        self._metric_values_agg_func = instance_level_aggregation_function

    def add_to_doc_specifics(self, formatted_doc: Doc, attribute_name: str, attribute_values: Any) -> None:
        if formatted_doc.specific is None:
            formatted_doc.specific = {}

        formatted_doc.specific[attribute_name] = attribute_values
    
    def cross_apply_metric_function(self, lst_lst_golds: List[List[str]], lst_lst_preds: List[List[str]], metric_function: Callable[[str, str], float],
                                    if_empty = 0.0) -> List[float]:
        """
        1. golds / preds をそれぞれフラットにする
        2. 直積（cross product）を取り、「(gold_token, pred_token)」タプルを生成
        3. metric_function((gold_token, pred_token)) をすべてに適用して返す。ただし空配列の場合は [if_empty] を返す
        """
        flat_golds = itertools.chain.from_iterable(lst_lst_golds)
        flat_preds = itertools.chain.from_iterable(lst_lst_preds)

        lst_ret = [metric_function(gold, pred) for gold, pred in itertools.product(flat_golds, flat_preds)]
        if len(lst_ret) == 0:
            lst_ret = [if_empty]
        
        return lst_ret
            
    def sample_level_fn(self, golds: list[str], predictions: list[str], formatted_doc: Doc, **kwargs):
        # extractor は文字列から回答スパンや正解スパンのリストを返す
        lst_lst_em_golds = list(map(self._em_gold_extractor, golds))
        lst_lst_em_preds = list(map(self._em_pred_extractor, predictions))
        lst_lst_quasi_em_golds = list(map(self._quasi_em_gold_extractor, golds))
        lst_lst_quasi_em_preds = list(map(self._quasi_em_pred_extractor, predictions))
        
        dict_dict_golds_and_preds = {
            "exact_match": {
                "golds": lst_lst_em_golds,
                "preds": lst_lst_em_preds
            },
            "quasi_exact_match": {
                "golds": lst_lst_quasi_em_golds,
                "preds": lst_lst_quasi_em_preds
            }
        }
       
        # Assert on empty gold and warn on empty pred
        for exact_or_quasi, dict_golds_and_preds in dict_dict_golds_and_preds.items():
            for gold_or_pred, lst_lst_values in dict_golds_and_preds.items():
                if any(len(lst) == 0 for lst in lst_lst_values):
                    raw_values = golds if gold_or_pred == "golds" else predictions
                    logger.warning(f"We did not manage to extract a {gold_or_pred} for {exact_or_quasi} in the correct format from inputs: {raw_values}")

                # We store extracted golds and predictions into formatted_doc object. These will be stored into "detailed results" eventually.
                attribute_name = f"{exact_or_quasi}_{gold_or_pred}"
                self.add_to_doc_specifics(formatted_doc=formatted_doc, attribute_name=attribute_name, attribute_values=lst_lst_values)
                
        # 評価指標を計算する．compare_golds_and_preds(正解, 回答, 評価指標計算関数) -> List[評価指標の値]
        dict_metric_values = {}
        
        _lst_lst_golds = dict_dict_golds_and_preds["exact_match"]["golds"]
        _lst_lst_preds = dict_dict_golds_and_preds["exact_match"]["preds"]
        dict_metric_values["exact_match"] = self.cross_apply_metric_function(_lst_lst_golds, _lst_lst_preds, compute_match)
        dict_metric_values["f1_score"] = self.cross_apply_metric_function(_lst_lst_golds, _lst_lst_preds, compute_char_f1)
        dict_metric_values["llmjpeval_f1_score"] = self.cross_apply_metric_function(_lst_lst_golds, _lst_lst_preds, compute_char_f1_llmjpeval)
        
        _lst_lst_golds = dict_dict_golds_and_preds["quasi_exact_match"]["golds"]
        _lst_lst_preds = dict_dict_golds_and_preds["quasi_exact_match"]["preds"]
        dict_metric_values["quasi_exact_match"] = self.cross_apply_metric_function(_lst_lst_golds, _lst_lst_preds, compute_match)
        dict_metric_values["f1_score_quasi"] = self.cross_apply_metric_function(_lst_lst_golds, _lst_lst_preds, compute_char_f1)
        dict_metric_values["llmjpeval_f1_score_quasi"] = self.cross_apply_metric_function(_lst_lst_golds, _lst_lst_preds, compute_char_f1_llmjpeval)
        
        # すべての (gold, pred) pair に対して計算した評価指標の値に集約関数を適用する
        for metric_name in dict_metric_values.keys():
            dict_metric_values[metric_name] = self._metric_values_agg_func(dict_metric_values[metric_name])
            
        return dict_metric_values

    @classmethod
    def METRIC_NAMES(cls) -> List[str]:
        lst_metric_names = [
            "exact_match", "f1_score", "llmjpeval_f1_score",
            "quasi_exact_match", "f1_score_quasi", "llmjpeval_f1_score_quasi",
        ]
        return lst_metric_names
    
    @classmethod
    def METRICS_HIGHER_IS_BETTER(cls) -> Dict[str, bool]:
        dict_ret = {metric_name:True for metric_name in cls.METRIC_NAMES()}
        return dict_ret
    