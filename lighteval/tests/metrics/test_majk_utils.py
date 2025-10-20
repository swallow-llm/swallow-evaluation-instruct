"""
Tests for Maj@K metric utilities.
"""

import pytest
import numpy as np
from unittest.mock import Mock

from lighteval.metrics.sample_metric_utils import (
    normalize_multiple_extractions,
    create_sampling_metric_fn,
    create_majk_metrics,
    create_sampling_metrics,
)
from lighteval.metrics.utils.metric_utils import (
    MetricCategory,
    MetricUseCase,
    SampleLevelMetric,
)
from lighteval.tasks.requests import Doc


class TestHashMultipleExtractions:
    """Test the hash_multiple_extractions function."""
    
    def test_empty_list(self):
        """Test with empty list."""
        result = normalize_multiple_extractions([])
        assert result == "EMPTY"
    
    def test_single_extraction(self):
        """Test with single extraction."""
        result = normalize_multiple_extractions(["5"])
        assert result == "5"
    
    def test_multiple_extractions(self):
        """Test with multiple extractions."""
        result = normalize_multiple_extractions(["5", "3", "7"])
        assert result == "3|5|7"

    def test_same_extractions_same_hash(self):
        """Test that same extractions produce same hash."""
        result1 = normalize_multiple_extractions(["5", "3"])
        result2 = normalize_multiple_extractions(["3", "5"])  # Different order
        assert result1 == result2  # Should be same due to sorting


class TestCreateMajKMetricFn:
    """Test the create_majk_metric_fn function."""
    
    def setup_method(self):
        """Set up test fixtures."""
        # Create a mock extractive metric
        self.base_metric = Mock(spec=SampleLevelMetric)
        self.base_metric.sample_level_fn = Mock()
        self.base_metric.use_case = MetricUseCase.ACCURACY
        self.base_metric.supports_return_extracted_predictions = True
        
        # Create a mock formatted_doc
        self.formatted_doc = Mock(spec=Doc)
    
    def test_insufficient_predictions_raises_error(self):
        """Test that insufficient predictions raises ValueError."""
        majk_fn = create_sampling_metric_fn(self.base_metric, k=5, metric_type="maj")
        predictions = ["A", "B", "C"]  # Only 3 predictions, but k=5
        
        with pytest.raises(ValueError, match="Number of predictions \\(3\\) is less than k \\(5\\)"):
            majk_fn(golds=["A"], predictions=predictions, formatted_doc=self.formatted_doc)
    
    def test_majk_calculation_with_extractions(self):
        """Test Maj@K calculation with extracted results."""
        # Mock the base metric to return different scores and extractions
        def mock_sample_level_fn(
            golds,
            predictions,
            formatted_doc,
            return_extracted_predictions=False,
            **kwargs,
        ):
            pred = predictions[0]
            # Set up mock extraction results - use a real dict instead of Mock
            if not hasattr(formatted_doc, 'specific') or formatted_doc.specific is None:
                formatted_doc.specific = {}
            
            if pred == "answer is 5":
                extracted = ["5"]
                formatted_doc.specific["extracted_predictions"] = extracted
                score = 1.0  # Correct
            elif pred == "answer is 3":
                extracted = ["3"]
                formatted_doc.specific["extracted_predictions"] = extracted
                score = 0.0  # Incorrect
            elif pred == "the result is 5":
                extracted = ["5"]
                formatted_doc.specific["extracted_predictions"] = extracted
                score = 1.0  # Correct
            else:
                extracted = [pred]
                formatted_doc.specific["extracted_predictions"] = extracted
                score = 0.0

            if return_extracted_predictions:
                return score, extracted
            return score

        self.base_metric.sample_level_fn.side_effect = mock_sample_level_fn
        
        # Initialize formatted_doc.specific as a real dict
        self.formatted_doc.specific = {}

        majk_fn = create_sampling_metric_fn(self.base_metric, k=3, metric_type="maj")
        predictions = ["answer is 5", "answer is 3", "the result is 5"]
        
        result = majk_fn(golds=["5"], predictions=predictions, formatted_doc=self.formatted_doc)
        
        # Should be > 0 since "5" appears twice and is correct
        assert result > 0.0
        assert result <= 1.0
        assert self.formatted_doc.specific["extracted_predictions_freq"] == [
            {"answer": "3", "count": 1},
            {"answer": "5", "count": 2},
        ]


class TestCreateMajKMetrics:
    """Test the create_majk_metrics function."""
    
    def setup_method(self):
        """Set up test fixtures."""
        # Create a mock extractive metric
        self.base_metric = Mock(spec=SampleLevelMetric)
        self.base_metric.metric_name = "extractive_test"
        self.base_metric.use_case = MetricUseCase.ACCURACY
        self.base_metric.sample_level_fn = Mock()
        self.base_metric.corpus_level_fn = np.mean
        self.base_metric.supports_return_extracted_predictions = False
    
    def test_create_multiple_k_values(self):
        """Test creating metrics for multiple k values."""
        self.base_metric.supports_return_extracted_predictions = True

        k_values = [1, 3, 5]
        num_samples = 10

        metrics = create_majk_metrics(self.base_metric, k_values, num_samples)

        assert len(metrics) == 3

        # Check metric names
        expected_names = [
            "extractive_test_maj@1:10",
            "extractive_test_maj@3:10",
            "extractive_test_maj@5:10"
        ]

        for i, metric in enumerate(metrics):
            assert isinstance(metric, SampleLevelMetric)
            assert metric.metric_name == expected_names[i]
            assert metric.category == MetricCategory.GENERATIVE_SAMPLING
            assert metric.use_case == MetricUseCase.ACCURACY
            assert metric.higher_is_better is True

    def test_incompatible_metric_raises_error(self):
        """Test that incompatible metric raises error."""
        self.base_metric.supports_return_extracted_predictions = False

        k_values = [1, 3]
        num_samples = 10

        with pytest.raises(ValueError, match="is not compatible with Maj@K"):
            create_majk_metrics(self.base_metric, k_values, num_samples)


class TestCreateSamplingMetrics:
    """Test the create_sampling_metrics function."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.base_metric = Mock(spec=SampleLevelMetric)
        self.base_metric.metric_name = "test_metric"
        self.base_metric.use_case = MetricUseCase.ACCURACY
        self.base_metric.corpus_level_fn = np.mean
        self.base_metric.supports_return_extracted_predictions = False
    
    def test_pass_metrics_creation(self):
        """Test creating Pass@K metrics."""
        k_values = [1, 3]
        num_samples = 10
        
        metrics = create_sampling_metrics(self.base_metric, k_values, num_samples, metric_type="pass")
        
        assert len(metrics) == 2
        assert metrics[0].metric_name == "test_metric_pass@1:10"
        assert metrics[1].metric_name == "test_metric_pass@3:10"
    
    def test_maj_metrics_creation(self):
        """Test creating Maj@K metrics."""
        self.base_metric.supports_return_extracted_predictions = True

        k_values = [1, 3]
        num_samples = 10

        metrics = create_sampling_metrics(self.base_metric, k_values, num_samples, metric_type="maj")

        assert len(metrics) == 2
        assert metrics[0].metric_name == "test_metric_maj@1:10"
        assert metrics[1].metric_name == "test_metric_maj@3:10"
