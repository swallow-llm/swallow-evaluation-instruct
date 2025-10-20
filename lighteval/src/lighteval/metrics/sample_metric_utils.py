
"""
Pass@K and Maj@K metric utilities for creating sampling-based versions of existing SampleLevelMetric.
This module provides utilities to convert any SampleLevelMetric into Pass@K or Maj@K metrics 
that can be used across different benchmarks.
"""

from typing import List, Literal
from collections import defaultdict
import copy
from lighteval.metrics.pass_at_k_solutions import estimate_pass_at_k

from lighteval.metrics.utils.metric_utils import (
    MetricUseCase,
    MetricCategory,
    SampleLevelMetric,
)
from lighteval.metrics.maj_at_k_solutions import maj_at_k_exact_dp_scipy
from lighteval.tasks.requests import Doc


def powers_of_two_up_to_n(N):
    """Generates all powers of two up to a given number N.

    Args:
        N (int): The upper limit (inclusive) for generating powers of two.

    Returns:
        List[int]: A list of all powers of two up to N.
    """
    result = []
    value = 1
    while value <= N:
        result.append(value)
        value *= 2
    if result[-1] != N:
        result.append(N)
    return result


def normalize_multiple_extractions(extracted_list) -> str:
    """抽出結果をdeduplicateおよびstringに変換
    
    Args:
        extracted_list: 抽出結果のリスト
        
    Returns:
        str: ハッシュ化された結果
    """
    if not extracted_list:
        return "EMPTY"
    if len(extracted_list) == 1:
        return str(extracted_list[0])

    normalized = [str(x) for x in extracted_list]
    unique_extractions = sorted(set(normalized))

    if len(unique_extractions) == 1:
        return unique_extractions[0]

    # 複数ある場合は重複を除去したうえでソートして連結した文字列を返す
    combined = "|".join(unique_extractions)
    return combined

def create_sampling_metric_fn(base_metric: SampleLevelMetric, k: int, metric_type: str) -> callable:
    """Pass@KとMaj@K共通のメトリクス関数作成
    
    Args:
        base_metric: 基となるSampleLevelMetric
        k: K値
        metric_type: "pass" または "maj"
        
    Returns:
        サンプリングベースのメトリクス関数
    """
    
    supports_extracted = getattr(base_metric, "supports_return_extracted_predictions", False)
    if not supports_extracted and metric_type == "maj":
        raise ValueError("Metrics declaring supports_return_extracted_predictions=True must be used for Maj@K.")
    
    def sampling_metric_fn(golds: List[str], predictions: List[str], formatted_doc: Doc, **kwargs) -> float:
        if len(predictions) < k:
            raise ValueError(f"Number of predictions ({len(predictions)}) is less than k ({k}) for {metric_type}@{k}")

        counts_freq_snapshot = None

        if metric_type == "pass":
            # Pass@K計算
            scores = []
            for pred in predictions:
                temp_doc = copy.deepcopy(formatted_doc)
                score = base_metric.sample_level_fn(golds=golds, predictions=[pred], formatted_doc=temp_doc, **kwargs)
                scores.append(int(score))
            
            num_samples = len(predictions)
            num_correct = sum(scores)
            result = estimate_pass_at_k(num_samples=num_samples, num_correct=num_correct, k=k)
            
        elif metric_type == "maj":
            # Maj@K計算

            # counts_dictとcorrect_dictを作成
            counts_dict = defaultdict(int)
            correct_dict = {}
            for pred in predictions:
                temp_doc = copy.deepcopy(formatted_doc)
                score, extracted_list = base_metric.sample_level_fn(
                    golds=golds,
                    predictions=[pred],
                    formatted_doc=temp_doc,
                    return_extracted_predictions=True,
                    **kwargs,
                )
                is_correct = bool(score)
                
                # 抽出結果の取得
                if extracted_list is None:
                    extracted_list = []
                elif not isinstance(extracted_list, list):
                    extracted_list = list(extracted_list)
                
                normalized_result = normalize_multiple_extractions(extracted_list)
                counts_dict[normalized_result] += 1
                correct_dict[normalized_result] = is_correct

            # Maj@K計算
            result = maj_at_k_exact_dp_scipy(counts_dict, correct_dict, k)
            counts_freq_snapshot = [
                {"answer": key, "count": int(value), "is_correct": bool(correct_dict.get(key, None))}
                for key, value in sorted(counts_dict.items())
            ]
        else:
            raise ValueError(f"Unknown metric_type: {metric_type}")

        # 全てのpredictionsでbase_metricを実行してformatted_doc.specificを正しく設定        
        _ = base_metric.sample_level_fn(golds=golds, predictions=predictions, formatted_doc=formatted_doc, **kwargs)

        # Maj@Kの場合，回答の頻度情報をformatted_doc.specific.extracted_predictions_freqに保存
        if counts_freq_snapshot is not None:
            if not hasattr(formatted_doc, "specific") or formatted_doc.specific is None:
                formatted_doc.specific = {}
            formatted_doc.specific["extracted_predictions_freq"] = counts_freq_snapshot

        return result
    
    return sampling_metric_fn

def create_sampling_metrics(base_metric: SampleLevelMetric, k_values: List[int], num_samples: int, metric_type: Literal["pass", "maj"]) -> List[SampleLevelMetric]:
    """
    SampleLevelMetricから複数のK値に対してサンプリングベースのメトリクス（Pass@KまたはMaj@K）を作成します。
    
    Args:
        base_metric: 基となるSampleLevelMetric
        k_values: K値のリスト
        num_samples: サンプリング数
        metric_type: メトリクスタイプ（"pass" または "maj"）
    
    Returns:
        サンプリングベースのメトリクスのリスト
    """
    assert base_metric.use_case == MetricUseCase.ACCURACY, "Base metric must be an accuracy-type metric."

    # Maj@Kの場合は抽出結果取得に対応しているかをチェック
    if metric_type == "maj" and not getattr(base_metric, "supports_return_extracted_predictions", False):
        raise ValueError(
            f"Base metric '{base_metric.metric_name}' is not compatible with Maj@K. "
            f"Set supports_return_extracted_predictions=True and implement the optional "
            f"return_extracted_predictions interface to use Maj@K."
        )

    metrics = []
    for k in k_values:
        # メトリクスタイプに応じて適切な関数を選択
        if metric_type == "pass":
            sampling_fn = create_sampling_metric_fn(base_metric, k, metric_type="pass")
        elif metric_type == "maj":
            sampling_fn = create_sampling_metric_fn(base_metric, k, metric_type="maj")
        else:
            raise ValueError(f"Unknown metric_type: {metric_type}")
        
        # メトリクス名を生成
        base_name = base_metric.metric_name
        metric_name = f"{base_name}_{metric_type}@{k}:{num_samples}"
        
        # SampleLevelMetricを作成
        metric = SampleLevelMetric(
            metric_name=metric_name,
            category=MetricCategory.GENERATIVE_SAMPLING,
            use_case=base_metric.use_case,
            higher_is_better=True,
            sample_level_fn=sampling_fn,
            corpus_level_fn=base_metric.corpus_level_fn,
        )
        
        metrics.append(metric)
    
    return metrics


def create_passk_metrics(base_metric: SampleLevelMetric, k_values: List[int], num_samples: int) -> List[SampleLevelMetric]:
    """
    SampleLevelMetricから複数のK値に対してPass@Kメトリクスを作成します。
    
    Args:
        base_metric: 基となるSampleLevelMetric
        k_values: Pass@KのK値のリスト
        num_samples: サンプリング数
    
    Returns:
        Pass@Kメトリクスのリスト
    """
    return create_sampling_metrics(base_metric, k_values, num_samples, metric_type="pass")


def create_majk_metrics(base_metric: SampleLevelMetric, k_values: List[int], num_samples: int) -> List[SampleLevelMetric]:
    """
    SampleLevelMetricから複数のK値に対してMaj@Kメトリクスを作成します。
    
    Args:
        base_metric: 基となるSampleLevelMetric
        k_values: Maj@KのK値のリスト
        num_samples: サンプリング数
    
    Returns:
        Maj@Kメトリクスのリスト
    """
    return create_sampling_metrics(base_metric, k_values, num_samples, metric_type="maj")
