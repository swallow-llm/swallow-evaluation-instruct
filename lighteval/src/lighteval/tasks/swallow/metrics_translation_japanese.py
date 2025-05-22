from typing import List

from janome.tokenizer import Tokenizer as JanomeTokenizer
import unicodedata

from lighteval.metrics.utils.metric_utils import (
    CorpusLevelMetric,
    MetricCategory,
    MetricUseCase,
)
from lighteval.metrics.metrics_corpus import CorpusLevelTranslationMetric
from lighteval.metrics.sample_preparator import GenerativeCorpusMetricInput


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
    
    def __init__(self, remove_whiespace_tokens: bool = False, lowercase: bool = False, normalize_nfkc: bool = False, *, 
                 udic = '', udic_enc = 'utf8', udic_type = 'ipadic', **kwargs_janome_tokenizer):
        self.segmenter = JanomeTextSegmenter(remove_whiespace_tokens=remove_whiespace_tokens, lowercase=lowercase, normalize_nfkc=normalize_nfkc, 
                                             udic=udic, udic_enc=udic_enc, udic_type=udic_type, **kwargs_janome_tokenizer)
    
    def prepare(self, golds: list[str], predictions: list[str], **kwargs):
        """
        出力された翻訳文 (predictions) および 参照訳 (golds) を JanomeTextSegmenter で分かち書きする前処理クラス

        Args:
            golds (list[str]): 1つの事例に対する参照訳のリスト
            predictions (list[str]): 1つの事例に対する翻訳文のリスト

        Returns:
            GenerativeCorpusMetricInput: Stores the golds and predictions as such
        """
        golds = [" ".join(self.segmenter.segment(gold)) for gold in golds]
        predictions = [" ".join(self.segmenter.segment(pred)) for pred in predictions]
        
        return GenerativeCorpusMetricInput(golds=golds, preds=predictions)
    
    
bleu_ja = CorpusLevelMetric(
    metric_name="bleu_ja",
    sample_level_fn=JapaneseTranslationPreparator(remove_whiespace_tokens=True, lowercase=False, normalize_nfkc=False).prepare,
    category=MetricCategory.GENERATIVE,
    use_case=MetricUseCase.TRANSLATION,
    corpus_level_fn=CorpusLevelTranslationMetric("bleu").compute,
    higher_is_better=True,
)    
