from typing import List, Literal, Callable, Optional

import unicodedata
import re
from janome.tokenizer import Tokenizer as JanomeTokenizer

from lighteval.metrics.utils.metric_utils import (
    CorpusLevelMetric,
    MetricCategory,
    MetricUseCase,
)
from lighteval.metrics.metrics_corpus import CorpusLevelTranslationMetric
from lighteval.metrics.sample_preparator import GenerativeCorpusMetricInput
from lighteval.tasks.requests import Doc

from .utils import _regex_extractor


def _fenced_codeblock_extraction_function(text: str,
    extraction_mode: Literal["first_match", "last_match", "any_match"],
    ) -> List[str]:
    """
    Markdownのコードブロックからテキストを抽出する．
    行数や言語指定などのバリエーションを許容する．
    """
    str_codeblock_regex = (
        r'```'                        # opening fence
        r'(?:[^\n\r]*\r?\n(?!```))?'  # optional info-string + newline
        r'(?P<content>.*?)'           # DOTALL, greedy
        r'(?:\r?\n)?'                 # optional newline before closing fence
        r'```'                        # closing fence
    )
    codeblock_regex = re.compile(str_codeblock_regex, re.DOTALL | re.UNICODE)

    return _regex_extractor(
        obj_regex=codeblock_regex,
        match_group_name="content",
        text=text,
        extraction_mode=extraction_mode,
    )

def _prefixed_line_extraction_function(text: str, prefix: str,
    extraction_mode: Literal["first_match", "last_match", "any_match"],
    ) -> List[str]:
    """
    文頭の接頭辞 `{prefix}: ` に続くテキストを抽出する．  
    たとえば prefix="日本語:" と指定すると `日本語: こんにちは` から `こんにちは` を抽出する．
    
    テキスト抽出の仕様
    * 接頭辞の前は文頭またはnewlineとする．文中で接頭辞が出現しても無視される．
    * 接頭辞直後のwhitespace有無は問わない
    * 末尾は改行または文末
    """
    prefix_escaped = re.escape(prefix)

    pattern = rf'(?m)^\s*{prefix_escaped}\s*(?P<content>[^\n\r]*)'
    prefix_regex   = re.compile(pattern, re.UNICODE)

    return _regex_extractor(
        obj_regex=prefix_regex,
        match_group_name="content",
        text=text,
        extraction_mode=extraction_mode,
    )

def _pass_through(text: str) -> List[str]:
    return [text]


class JanomeTextSegmenter(JanomeTokenizer):
    
    def __init__(self, remove_whitespace_tokens: bool = True, lowercase: bool = False, normalize_nfkc: bool = False, 
                 **kwargs_janome_tokenizer):
        
        """
        Janomeを用いて日本語テキストを分かち書きするクラス．
        sacrebleuのリファレンス分かち書きは MeCab+IPADICだがMeCabはPythonでインストールが完結しない．
        pure python でインストールが容易かつ MeCab+IPADIC と処理結果がほとんど同一であることから Janome を選択した．
        Ref. https://pypi.org/project/Janome/

        Args:
            remove_whitespace_tokens: 空白トークンを削除．MeCabと合わせるためTrueを推奨
            lowercase: トークンを小文字化
            normalize_nfkc: トークンをNFKC正規化

        Returns:
            JanomeTextSegmenterクラス．.segment() method で分かち書きを実行する
            
        Usage:
            segmenter = JanomeTextSegmenter(remove_whitespace_tokens=True)
            segmenter.segment(text="今日は晴れです。")
            >>> ["今日", "は", "晴れ", "です", "。"]
        """
        
        super().__init__(wakati=True, **kwargs_janome_tokenizer)
        self.remove_whitespace_tokens = remove_whitespace_tokens
        self.lowercase = lowercase
        self.normalize_nfkc = normalize_nfkc
    
    def _is_whitespace(self, token: str) -> bool:
        return token in (" ", "　")
    
    def _normalize_nfkc(self, token: str) -> str:
        return unicodedata.normalize("NFKC", token)
        
    def segment(self, text: str) -> List[str]:
        lst_tokens = list(self.tokenize(text))
        if self.remove_whitespace_tokens:
            lst_tokens = [token for token in lst_tokens if not self._is_whitespace(token)]
        if self.lowercase:
            lst_tokens = [token.lower() for token in lst_tokens]
        if self.normalize_nfkc:
            lst_tokens = list(map(self._normalize_nfkc, lst_tokens))
            
        return lst_tokens


class TranslationPreparator:
    
    def __init__(self, text_extraction_function: Callable[[str], List[str]], 
                 extraction_fallback_function: Optional[Callable[[str], List[str]]] = _pass_through):
        """_summary_
        BLEUやchrFなどを計算する際に使う前処理クラス．
        翻訳スパンの抽出，文字列の正規化に対応している．
        CorpusLevelMetric(sample_level_fn) に，インスタンスの .prepare() method を渡すことで前処理が行われるようになる．  

        Args:
            text_extraction_function (Callable[[str], List[str]]): 翻訳スパン抽出関数．翻訳指示プロンプトに整合させること．
            extraction_fallback_function (Callable[[str], List[str]]): 翻訳スパン抽出に失敗した場合のスパン抽出関数．デフォルトは応答文全体をそのまま翻訳とみなす．
        """        
        self.text_extraction_function = text_extraction_function
        self.extraction_fallback_function = extraction_fallback_function
        
    def prepare(self, golds: list[str], predictions: list[str], formatted_doc: Doc, **kwargs):
        """
        モデルの出力から翻訳文を抽出する

        Args:
            golds (list[str]): 1つの事例に対する参照訳のリスト
            predictions (list[str]): 1つの事例に対する翻訳文のリスト

        Returns:
            GenerativeCorpusMetricInput: Stores the golds and predictions as such
        """
        lst_translated = []
        for pred in predictions:
            _lst_extracted = self.text_extraction_function(text=pred)
            lst_translated.extend(_lst_extracted)
        
        # 抽出に失敗した場合はfallback
        if len(lst_translated) == 0:
            _fallback = self.extraction_fallback_function(pred)
            lst_translated.extend(_fallback)
            
        if formatted_doc.specific is None:
            formatted_doc.specific = {}
        formatted_doc.specific["extracted_predictions"] = lst_translated
        
        return GenerativeCorpusMetricInput(golds=golds, 
                                           preds=lst_translated)

    
class JapaneseTranslationPreparator:
    
    def __init__(self, 
        text_extraction_function: Callable[[str], List[str]],
        extraction_fallback_function: Optional[Callable[[str], List[str]]] = _pass_through,
        remove_whitespace_tokens: bool = False, lowercase: bool = False, normalize_nfkc: bool = False,  
        **kwargs_janome_tokenizer):
        """_summary_
        邦訳文に対してBLEUやchrFなどを計算する際に使う前処理クラス．
        翻訳スパンの抽出，Janomeによる分かち書き，トークン文字列の正規化に対応している．
        CorpusLevelMetric(sample_level_fn) に，インスタンスの .prepare() method を渡すことで前処理が行われるようになる．  

        Args:
            text_extraction_function (Callable[[str], List[str]]): 翻訳スパン抽出関数．翻訳指示プロンプトに整合させること．
            extraction_fallback_function (Callable[[str], List[str]]): 翻訳スパン抽出に失敗した場合のスパン抽出関数．デフォルトは応答文全体をそのまま翻訳とみなす．
            remove_whitespace_tokens (bool, optional): 空白トークンの削除. Defaults to False.
            lowercase (bool, optional): トークンの小文字化．Defaults to False.
            normalize_nfkc (bool, optional): トークンのNFKC正規化．Defaults to False.
            kwargs_janome_tokenizer: Janome.Tokenizer() に渡す引数．
        """
        self.text_extraction_function = text_extraction_function
        self.extraction_fallback_function = extraction_fallback_function
        
        self.segmenter = JanomeTextSegmenter(remove_whitespace_tokens=remove_whitespace_tokens, lowercase=lowercase, normalize_nfkc=normalize_nfkc, 
                                             **kwargs_janome_tokenizer)
    
    def prepare(self, golds: list[str], predictions: list[str], formatted_doc: Doc, **kwargs):
        """
        1. モデルの出力から翻訳文を抽出する
        2. 抽出した翻訳文および参照訳を分かち書きする

        Args:
            golds (list[str]): 1つの事例に対する参照訳のリスト
            predictions (list[str]): 1つの事例に対する翻訳文のリスト

        Returns:
            GenerativeCorpusMetricInput: Stores the golds and predictions as such
        """
        lst_translated = []
        for pred in predictions:
            _lst_extracted = self.text_extraction_function(text=pred)
            lst_translated.extend(_lst_extracted)
        
        # 抽出に失敗した場合はfallback
        if len(lst_translated) == 0:
            _fallback = self.extraction_fallback_function(pred)
            lst_translated.extend(_fallback)
            
        if formatted_doc.specific is None:
            formatted_doc.specific = {}
        formatted_doc.specific["extracted_predictions"] = lst_translated
        
        lst_reference_tokenzed = [" ".join(self.segmenter.segment(gold)) for gold in golds]
        lst_translated_tokenized = [" ".join(self.segmenter.segment(translated)) for translated in lst_translated]
        
        return GenerativeCorpusMetricInput(golds=lst_reference_tokenzed, 
                                           preds=lst_translated_tokenized)


def wmt20_enja_translation_span_extractor(text: str):
    return _prefixed_line_extraction_function(text=text, prefix="日本語:", extraction_mode="last_match")

def wmt20_jaen_translation_span_extractor(text: str):
    return _prefixed_line_extraction_function(text=text, prefix="English:", extraction_mode="last_match")


wmt20_enja_translation_preparator = JapaneseTranslationPreparator(
    text_extraction_function=wmt20_enja_translation_span_extractor, 
    extraction_fallback_function=_pass_through,
    remove_whitespace_tokens=False, lowercase=False, normalize_nfkc=False)

wmt20_jaen_translation_preparator = TranslationPreparator(
    text_extraction_function=wmt20_jaen_translation_span_extractor, 
    extraction_fallback_function=_pass_through)

# 邦訳文向けBLEU
bleu_ja = CorpusLevelMetric(
    metric_name="bleu",
    sample_level_fn=wmt20_enja_translation_preparator.prepare,
    category=MetricCategory.GENERATIVE,
    use_case=MetricUseCase.TRANSLATION,
    corpus_level_fn=CorpusLevelTranslationMetric("bleu", lang="").compute,
    higher_is_better=True,
)    

# 邦訳文向けchrF
chrf_ja = CorpusLevelMetric(
    metric_name="chrf",
    sample_level_fn=wmt20_enja_translation_preparator.prepare,
    category=MetricCategory.GENERATIVE,
    use_case=MetricUseCase.TRANSLATION,
    corpus_level_fn=CorpusLevelTranslationMetric("chrf", lang="").compute,
    higher_is_better=True,
)

# 邦訳文向けTER
ter_ja = CorpusLevelMetric(
    metric_name="ter",
    sample_level_fn=wmt20_enja_translation_preparator.prepare,
    category=MetricCategory.GENERATIVE,
    use_case=MetricUseCase.TRANSLATION,
    corpus_level_fn=CorpusLevelTranslationMetric("ter", lang="").compute,
    higher_is_better=False,
)

# 英訳向けBLEU
bleu_en = CorpusLevelMetric(
    metric_name="bleu",
    sample_level_fn=wmt20_jaen_translation_preparator.prepare,
    category=MetricCategory.GENERATIVE,
    use_case=MetricUseCase.TRANSLATION,
    corpus_level_fn=CorpusLevelTranslationMetric("bleu", lang="en").compute,
    higher_is_better=True,
)    

# 英訳向けchrF
chrf_en = CorpusLevelMetric(
    metric_name="chrf",
    sample_level_fn=wmt20_jaen_translation_preparator.prepare,
    category=MetricCategory.GENERATIVE,
    use_case=MetricUseCase.TRANSLATION,
    corpus_level_fn=CorpusLevelTranslationMetric("chrf", lang="en").compute,
    higher_is_better=True,
)

# 英訳向けTER
ter_en = CorpusLevelMetric(
    metric_name="ter",
    sample_level_fn=wmt20_jaen_translation_preparator.prepare,
    category=MetricCategory.GENERATIVE,
    use_case=MetricUseCase.TRANSLATION,
    corpus_level_fn=CorpusLevelTranslationMetric("ter", lang="en").compute,
    higher_is_better=False,
)
