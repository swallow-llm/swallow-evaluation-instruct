# MIT License

# Copyright (c) 2024 The HuggingFace Team

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

"""This module manages all the metrics occurring at the corpus level.
Some metrics (such as corpus BLEU) are not computed at the individual item level, but over all the corpus.
A number of these aggregations come from the EleutherAIHarness
"""
import logging
import math
from typing import Literal

import numpy as np
import sacrebleu
import sklearn.metrics

from lighteval.metrics.sample_preparator import (
    GenerativeCorpusMetricInput,
    LogprobCorpusMetricInput,
    PerplexityCorpusMetricInput,
)
from lighteval.utils.utils import as_list


logger = logging.getLogger(__name__)


# General aggregations
def matthews_corrcoef(items: list[GenerativeCorpusMetricInput]) -> float:
    """Computes the Matthews Correlation Coefficient, using scikit learn ([doc](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.matthews_corrcoef.html)).

    Args:
        items (list[dict]): List of GenerativeCorpusMetricInput

    Returns:
        float: Score
    """
    golds = [i.golds for i in items]
    preds = [i.preds for i in items]
    return sklearn.metrics.matthews_corrcoef(golds, preds)


class CorpusLevelF1Score:
    def __init__(self, average: str, num_classes: int = 2):
        """Stores the relevant parameters for the task's corpus level f1 score.

        Args:
            average (str): Method to use to compute the f1 score. Can be weighted, macro, micro.
            num_classes (int, optional): Num of possible choice classes. Defaults to 2. If this parameter is above 2, we'll compute multi f1 corpus score
        """
        if average not in ["weighted", "macro", "micro", None]:
            raise ValueError(
                f"A CorpusLevelF1Score must be initialized with weighted, macro, micro, or None as an average function. {average} was used."
            )
        self.average = average
        self.num_classes = num_classes

    def compute(self, items: list[LogprobCorpusMetricInput]):
        """Computes the metric score over all the corpus generated items, by using the scikit learn implementation."""
        golds = [i.golds for i in items]
        preds = [i.preds for i in items]
        # Single f1
        if self.num_classes == 2:
            fscore = sklearn.metrics.f1_score(golds, preds, average=self.average)
            return np.max(fscore)

        # Multi f1
        f1s = []
        for i in range(self.num_classes):
            f1s.append(sklearn.metrics.f1_score(y_true=golds == i, y_pred=preds == i))
        return float(np.mean(f1s))


class CorpusLevelTranslationMetric:
    def __init__(self, metric_type: str, lang: Literal["zh", "ja", "ko", ""] = ""):
        """Stores the relevant parameters for a corpus level translation metric.

        Args:
            metric_type (str): Can be any of bleu, chrf, or ter depending on the metric to use.
        """
        self.metric_type = metric_type
        self.lang = lang

    def get_metric(self):
        if self.metric_type == "bleu":
            return sacrebleu.BLEU(trg_lang=self.lang)
        elif self.metric_type == "chrf":
            return sacrebleu.CHRF()
        elif self.metric_type == "ter":
            return sacrebleu.TER(asian_support=True if self.lang != "" else False)
        else:
            raise ValueError(f"Unknown corpus level translation metric type : {self.metric_type}")

    def compute(self, items: list[GenerativeCorpusMetricInput]) -> float:
        """Computes the metric score over all the corpus generated items, by using the sacrebleu implementation.
        
        # BLEU    
        BLEU is a metric used to evaluate the quality of machine-generated sentences by comparing them to reference translations [1]. 
        It works by measuring how many word n-grams (regardless of order) in the generated sentence match those in the reference sentence. 
        Typically, a weighted average of unigram to 4-gram precision is used.
        It is recommended to use the SacreBLEU implementation [5], as BLEU is sensitive to tokenization choices [2,3,4]. 
        
        References:
        [1] PAPINENI, Kishore, et al. Bleu: a method for automatic evaluation of machine translation. In: ACL2002, 2002.
        [2] Comparing the Uncomparable to Claim the State of the Art: A Concerning Trend. https://kaitchup.substack.com/p/comparing-the-uncomparable-to-claim-the-state-of-the-art-a-concerning-trend-3d864522a0ba
        [3] POST, Matt. A Call for Clarity in Reporting BLEU Scores. WMT 2018, 2018, 186.
        [4] GRUSKY, Max. Rogue scores. In: ACL2023. 2023. p. 1914-1934.
        [5] https://github.com/mjpost/sacrebleu
        
        # SacreBLEU spec:
        SacreBLEU expects input as (generated sentences, reference sentences) as (List[str], List[List[str]]). 
        Note that each inner list corresponds to one reference across all generations.  
        In other words, for N predictions and M reference sentences each, the input shapes are:
        - preds [N]: List of N predicted sentences
        - refs [M, N]:  List of M lists, each containing N reference sentences       
        """
                
        metric = self.get_metric()        
        
        # Assert: The number of reference sentences must be identical across all predictions.
        first_item = next(iter(items))
        num_references = len(first_item.golds)
        for item in items:
            if num_references != len(item.golds):
                logger.error(
                    f"Error: inconsistent number of reference setences detected. Expected: {num_references}, actual: {len(item.golds)}"
                )
                raise ValueError
        
        golds = []
        for idx in range(num_references):
            _golds = [item.golds[idx] for item in items]
            golds.append(_golds)
        
        preds = []
        for i in items:
            pred = as_list(i.preds)
            if len(pred) > 1:
                logger.info(
                    f"Multiple predictions present, keeping only the first prediction (when computing sacrebleu.{metric.__name__})."
                )
            preds.append(pred[0])
        
        return float(metric.corpus_score(hypotheses=preds, references=golds).score)


class CorpusLevelPerplexityMetric:
    def __init__(self, metric_type: str):
        """Stores the relevant parameter for a corpus level perplexity metric.
        Perplexity metrics compute more or less the same thing, which is a variation on the
        average of log-probabilities over a sequence, but the normalization and processing applied
        is different depending on the metric type.
        Perplexity uses an exponential and no weights for the average, weighted perplexity uses an exponential
        and the number of words as weights for the log-prob average, and bits per byte uses the number of bits
        for normalization and divides the results by log(2).

        Args:
            metric_type (str): Can be any of `perplexity`, `weighted_perplexity` or `bits_per_byte`
        """
        if metric_type not in ["perplexity", "weighted_perplexity", "bits_per_byte"]:
            raise ValueError(f"Unknown corpus level perplexity metric type : {metric_type}")

        self.metric_type = metric_type

    def compute(self, items: list[PerplexityCorpusMetricInput]):
        """Computes the metric score over all the corpus generated items."""
        logprobs = [i.logprobs for i in items]
        weights = [i.weights for i in items]

        if self.metric_type == "perplexity":
            return math.exp(-np.mean(logprobs))
        if self.metric_type == "weighted_perplexity":
            return math.exp(-sum(logprobs) / sum(weights))
        if self.metric_type == "bits_per_byte":
            return -sum(logprobs) / sum(weights) * 1 / math.log(2)
