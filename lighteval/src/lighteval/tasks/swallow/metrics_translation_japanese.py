from typing import List, Literal, Callable

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



class JanomeTextSegmenter(JanomeTokenizer):
    
    def __init__(self, remove_whiespace_tokens: bool = True, lowercase: bool = False, normalize_nfkc: bool = False, *, 
                 udic = '', udic_enc = 'utf8', udic_type = 'ipadic', **kwargs_janome_tokenizer):
        
        """
        Janomeを用いて日本語テキストを分かち書きするクラス．
        sacrebleuのリファレンス分かち書きは MeCab+IPADICだがMeCabはPythonでインストールが完結しない．
        pure python でインストールが容易かつ MeCab+IPADIC と処理結果がほとんど同一であることから Janome を選択した．
        Ref. https://pypi.org/project/Janome/

        Args:
            remove_whiespace_tokens: 空白トークンを削除．MeCabと合わせるためTrueを推奨
            lowercase: トークンを小文字化
            normalize_nfkc: トークンをNFKC正規化

        Returns:
            JanomeTextSegmenterクラス．.segment() method で分かち書きを実行する
            
        Usage:
            segmenter = JanomeTextSegmenter(remove_whiespace_tokens=True)
            segmenter.segment(text="今日は晴れです。")
            >>> ["今日", "は", "晴れ", "です", "。"]
        """
        
        super().__init__(udic, udic_enc=udic_enc, udic_type=udic_type, wakati=True, **kwargs_janome_tokenizer)
        self.remove_whiespace_tokens = remove_whiespace_tokens
        self.lowercase = lowercase
        self.normalize_nfkc = normalize_nfkc
    
    def _is_whitespace(self, token: str) -> bool:
        return token in (" ", "　")
    
    def _normalize_nfkc(self, token: str) -> str:
        return unicodedata.normalize("NFKC", token)
        
    def segment(self, text: str) -> List[str]:
        lst_tokens = list(self.tokenize(text))
        if self.remove_whiespace_tokens:
            lst_tokens = [token for token in lst_tokens if not self._is_whitespace(token)]
        if self.lowercase:
            lst_tokens = [token.lower() for token in lst_tokens]
        if self.normalize_nfkc:
            lst_tokens = list(map(self._normalize_nfkc, lst_tokens))
            
        return lst_tokens
    
    
class JapaneseTranslationPreparator:
    
    def __init__(self, 
        text_extraction_function: Callable[[str], List[str]] = lambda text: _prefixed_line_extraction_function(text=text, prefix="日本語:", extraction_mode="last_match"),
        remove_whiespace_tokens: bool = False, lowercase: bool = False, normalize_nfkc: bool = False, *, 
        udic = '', udic_enc = 'utf8', udic_type = 'ipadic', **kwargs_janome_tokenizer):
        
        self.text_extraction_function = text_extraction_function
        
        self.segmenter = JanomeTextSegmenter(remove_whiespace_tokens=remove_whiespace_tokens, lowercase=lowercase, normalize_nfkc=normalize_nfkc, 
                                             udic=udic, udic_enc=udic_enc, udic_type=udic_type, **kwargs_janome_tokenizer)
    
    def prepare(self, golds: list[str], predictions: list[str], **kwargs):
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
        
        # 抽出に失敗した場合はそのまま入力する
        if len(lst_translated) == 0:
            lst_translated.append(pred)
        
        lst_reference_tokenzed = [" ".join(self.segmenter.segment(gold)) for gold in golds]
        lst_translated_tokenized = [" ".join(self.segmenter.segment(translated)) for translated in lst_translated]
        
        return GenerativeCorpusMetricInput(golds=lst_reference_tokenzed, 
                                           preds=lst_translated_tokenized)
    

# 邦訳文向けBLEU
bleu_ja = CorpusLevelMetric(
    metric_name="bleu",
    sample_level_fn=JapaneseTranslationPreparator(remove_whiespace_tokens=True, lowercase=False, normalize_nfkc=False).prepare,
    category=MetricCategory.GENERATIVE,
    use_case=MetricUseCase.TRANSLATION,
    corpus_level_fn=CorpusLevelTranslationMetric("bleu").compute,
    higher_is_better=True,
)    

# 邦訳文向けchrF
chrf_ja = CorpusLevelMetric(
    metric_name="chrf",
    sample_level_fn=JapaneseTranslationPreparator(remove_whiespace_tokens=True, lowercase=False, normalize_nfkc=False).prepare,
    category=MetricCategory.GENERATIVE,
    use_case=MetricUseCase.TRANSLATION,
    corpus_level_fn=CorpusLevelTranslationMetric("chrf").compute,
    higher_is_better=True,
)

# 邦訳文向けTER
ter_ja = CorpusLevelMetric(
    metric_name="ter",
    sample_level_fn=JapaneseTranslationPreparator(remove_whiespace_tokens=True, lowercase=False, normalize_nfkc=False).prepare,
    category=MetricCategory.GENERATIVE,
    use_case=MetricUseCase.TRANSLATION,
    corpus_level_fn=CorpusLevelTranslationMetric("ter").compute,
    higher_is_better=False,
)
