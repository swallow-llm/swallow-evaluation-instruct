import re
import neologdn
from typing import List, Optional, Callable, Literal

def _regex_extractor(obj_regex, text: str, extraction_mode: Literal["first_match", "last_match", "any_match"]) -> List[str]:
    """
    正規表現とテキストを渡すと extraction mode に従って抽出結果を返す関数．
    - extraction_mode="first_match": 最初のマッチのみを返す
    - extraction_mode="last_match": 最後のマッチのみを返す
    - extraction_mode="any_match": すべてのマッチを返す    
    """
    
    lst_matches_with_positions = [(match.group("content"), match.start(), match.end()) for match in obj_regex.finditer(text)]
    # 出現箇所の昇順にソート．つまり最後に出現したものが配列の最後に入っている
    lst_matches_with_positions = sorted(lst_matches_with_positions, key=lambda x: (x[2], -x[1]), reverse=True)
    
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
    
    return _regex_extractor(obj_regex=boxed_match_regex, text=text, extraction_mode=extraction_mode)

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

    return _regex_extractor(obj_regex=answer_regex, text=text, extraction_mode=extraction_mode)
       
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
        オプションは上から順に適用される:
          1. boxed_match_extraction
          2. free_form_answer_extraction
          3. extraction_fallback_function (両者とも空なら)
          4. neologdn_normalize
          5. remove_paren_and_quote
          6. canonicalize_binary_response
          7. lowercase
        """
        self.use_boxed = use_boxed_match_extraction
        self.use_free  = use_free_form_answer_extraction
        self.fallback  = extraction_fallback_function
        self.do_neologd = neologdn_normalize
        self.do_strip  = remove_paren_and_quote
        self.do_bin    = canonicalize_binary_response
        self.do_lower  = lowercase
        self.boxed_match_extraction_mode = boxed_match_extraction_mode,
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
            results = list(map(_canonicalize_binary_response))
        if self.do_lower:
            results = [r.lower() for r in results]
        
        return results
