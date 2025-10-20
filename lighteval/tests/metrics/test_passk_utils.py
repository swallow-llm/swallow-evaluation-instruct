"""
Tests for Pass@K metric utilities.
"""

import pytest
import numpy as np
from unittest.mock import Mock

from lighteval.metrics.sample_metric_utils import (
    create_passk_metrics,
    create_sampling_metric_fn
)
from lighteval.metrics.utils.metric_utils import (
    MetricCategory,
    MetricUseCase,
    SampleLevelMetric,
)
from lighteval.metrics.pass_at_k_solutions import estimate_pass_at_k
from lighteval.tasks.requests import Doc


class TestEstimatePassAtK:
    """Test the estimate_pass_at_k function."""
    
    def test_all_correct(self):
        """Test when all samples are correct."""
        for k in [1, 5, 10]:
            result = estimate_pass_at_k(num_samples=10, num_correct=10, k=k)
            assert result == 1.0
        
    def test_none_correct(self):
        """Test when no samples are correct."""
        for k in [1, 5, 10]:
            result = estimate_pass_at_k(num_samples=10, num_correct=0, k=k)
            assert result == 0.0
        
    def test_partial_correct_exact_values(self):
        """Test exact values for specific scenarios."""
        for num_correct in [1,3,5]:
            result = estimate_pass_at_k(num_samples=10, num_correct=num_correct, k=1)
            expected = pytest.approx(num_correct / 10, abs=1e-10)
            assert result == expected

    def test_pass_at_k_monotonicity(self):
        """Test that Pass@K increases monotonically with K."""
        num_samples = 10
        num_correct = 3
        
        results = []
        for k in range(1, 8):  # k from 1 to 7 (less than num_samples - num_correct)
            result = estimate_pass_at_k(num_samples, num_correct, k)
            results.append(result)
        
        # Each result should be >= the previous one
        for i in range(1, len(results)):
            assert results[i] >= results[i-1], f"Pass@{i+1} ({results[i]}) should be >= Pass@{i} ({results[i-1]})"
    
    def test_k_greater_than_failures(self):
        """Test when k is greater than the number of failures."""
        # 10 samples, 8 correct, 2 failures, k=3
        # Since failures (2) < k (3), should return 1.0
        result = estimate_pass_at_k(num_samples=10, num_correct=8, k=3)
        assert result == 1.0
        
        # 10 samples, 7 correct, 3 failures, k=4
        # Since failures (3) < k (4), should return 1.0
        result = estimate_pass_at_k(num_samples=10, num_correct=7, k=4)
        assert result == 1.0
    
    def test_edge_cases(self):
        """Test edge cases."""
        # Single sample, correct
        result = estimate_pass_at_k(num_samples=1, num_correct=1, k=1)
        assert result == 1.0
        
        # Single sample, incorrect
        result = estimate_pass_at_k(num_samples=1, num_correct=0, k=1)
        assert result == 0.0
        
        # k equals num_samples
        result = estimate_pass_at_k(num_samples=5, num_correct=2, k=5)
        assert result == 1.0  # Must include all samples, so guaranteed to get correct ones
        
        # k equals num_correct
        result = estimate_pass_at_k(num_samples=10, num_correct=3, k=3)
        expected = 1.0 - np.prod(1.0 - 3 / np.arange(8, 11))  # 1 - (5/8 * 6/9 * 7/10)
        assert result == pytest.approx(expected, abs=1e-10)
    
    def test_known_mathematical_values(self):
        """Test against known mathematical values."""
        # 10 samples, 2 correct, k=2
        # P(at least 1 correct in 2 draws) = 1 - P(both incorrect)
        # P(both incorrect) = (8/10) * (7/9)
        # P(at least 2 correct) = 1 - (8/10) * (7/9)
        result = estimate_pass_at_k(num_samples=10, num_correct=2, k=2)
        expected = 1.0 - (8/10) * (7/9)
        assert result == pytest.approx(expected, abs=1e-10)
        
        # 5 samples, 1 correct, k=3
        # P(correct in 3 draws) = 1 - P(all 3 incorrect)
        # P(all 3 incorrect) = (4/5) * (3/4) * (2/3)
        # P(at least 1 correct) = 1 - (4/5) * (3/4) * (2/3)
        result = estimate_pass_at_k(num_samples=5, num_correct=1, k=3)
        expected = 1.0 - (4/5) * (3/4) * (2/3)
        assert result == pytest.approx(expected, abs=1e-10)


class TestCreatePassKMetricFn:
    """Test the create_passk_metric_fn function."""
    
    def setup_method(self):
        """Set up test fixtures."""
        # Create a mock base metric that returns 1.0 for correct answers, 0.0 for incorrect
        self.base_metric = Mock(spec=SampleLevelMetric)
        self.base_metric.sample_level_fn = Mock()
        self.base_metric.use_case = MetricUseCase.ACCURACY
        
        # Create a mock formatted_doc
        self.formatted_doc = Mock(spec=Doc)
        self.formatted_doc.get_golds.return_value = ["A"]
    
    def test_all_correct_predictions(self):
        """Test when all predictions are correct."""
        # Mock the base metric to always return 1.0 (correct)
        self.base_metric.sample_level_fn.return_value = 1.0

        passk_fn = create_sampling_metric_fn(self.base_metric, k=3, metric_type="pass")
        predictions = ["A", "A", "A", "A", "A"]  # 5 predictions, all correct

        result = passk_fn(golds=["A"], predictions=predictions, formatted_doc=self.formatted_doc)

        # With all correct, Pass@3 should be 1.0
        assert result == 1.0
    
    def test_some_correct_predictions(self):
        """Test when some predictions are correct."""
        # Mock the base metric to return 1.0 for first 2 calls, 0.0 for the rest
        # Need extra call for final base_metric execution to save details
        self.base_metric.sample_level_fn.side_effect = [1.0, 1.0, 0.0, 0.0, 0.0, 1.0]

        passk_fn = create_sampling_metric_fn(self.base_metric, k=3, metric_type="pass")
        predictions = ["A", "A", "B", "C", "D"]  # 5 predictions, 2 correct
        
        result = passk_fn(golds=["A"], predictions=predictions, formatted_doc=self.formatted_doc)
        
        # With 5 samples, 2 correct, k=3: should be > 0 but < 1
        assert 0.0 < result < 1.0
    
    def test_no_correct_predictions(self):
        """Test when no predictions are correct."""
        # Mock the base metric to always return 0.0 (incorrect)
        self.base_metric.sample_level_fn.return_value = 0.0

        passk_fn = create_sampling_metric_fn(self.base_metric, k=3, metric_type="pass")
        predictions = ["B", "C", "D", "E", "F"]  # 5 predictions, none correct
        
        result = passk_fn(golds=["A"], predictions=predictions, formatted_doc=self.formatted_doc)
        
        # With no correct predictions, Pass@3 should be 0.0
        assert result == 0.0
    
    def test_insufficient_predictions_raises_error(self):
        """Test that insufficient predictions raises ValueError."""
        passk_fn = create_sampling_metric_fn(self.base_metric, k=5, metric_type="pass")
        predictions = ["A", "B", "C"]  # Only 3 predictions, but k=5
        
        with pytest.raises(ValueError, match="Number of predictions \\(3\\) is less than k \\(5\\)"):
            result = passk_fn(golds=["A"], predictions=predictions, formatted_doc=self.formatted_doc)
    
    def test_exact_k_predictions(self):
        """Test when number of predictions equals k."""
        # Need extra call for final base_metric execution to save details
        self.base_metric.sample_level_fn.side_effect = [1.0, 0.0, 0.0, 1.0]

        passk_fn = create_sampling_metric_fn(self.base_metric, k=3, metric_type="pass")
        predictions = ["A", "B", "C"]  # Exactly 3 predictions for k=3
        
        result = passk_fn(golds=["A"], predictions=predictions, formatted_doc=self.formatted_doc)
        
        # Should work without error
        # With k=3 and 3 predictions where 1 is correct, Pass@3 should be 1.0
        # (since we're guaranteed to include the correct answer when selecting all 3)
        assert result == 1.0


class TestCreatePassKMetrics:
    """Test the create_passk_metrics function."""
    
    def setup_method(self):
        """Set up test fixtures."""
        # Create a mock base metric
        self.base_metric = Mock(spec=SampleLevelMetric)
        self.base_metric.metric_name = "test_metric"
        self.base_metric.use_case = MetricUseCase.ACCURACY
        self.base_metric.sample_level_fn = Mock(return_value=1.0)
        self.base_metric.corpus_level_fn = lambda scores: sum(scores) / len(scores) if scores else 0.0

    def test_create_multiple_k_values(self):
        """Test creating metrics for multiple k values."""
        k_values = [1, 3, 5]
        num_samples = 10
        
        metrics = create_passk_metrics(self.base_metric, k_values, num_samples)
        
        assert len(metrics) == 3
        
        # Check metric names
        expected_names = [
            "test_metric_pass@1:10",
            "test_metric_pass@3:10",
            "test_metric_pass@5:10"
        ]
        
        for i, metric in enumerate(metrics):
            assert isinstance(metric, SampleLevelMetric)
            assert metric.metric_name == expected_names[i]
            assert metric.category == MetricCategory.GENERATIVE_SAMPLING
            assert metric.use_case == MetricUseCase.ACCURACY
            assert metric.higher_is_better is True
    
    def test_single_k_value(self):
        """Test creating metric for single k value."""
        k_values = [1]
        num_samples = 5
        
        metrics = create_passk_metrics(self.base_metric, k_values, num_samples)
        
        assert len(metrics) == 1
        assert metrics[0].metric_name == "test_metric_pass@1:5"
    
    def test_custom_num_samples(self):
        """Test with custom num_samples."""
        k_values = [1, 5]
        num_samples = 20
        
        metrics = create_passk_metrics(self.base_metric, k_values, num_samples)
        
        assert len(metrics) == 2
        assert metrics[0].metric_name == "test_metric_pass@1:20"
        assert metrics[1].metric_name == "test_metric_pass@5:20"


class TestIntegration:
    """Integration tests for the complete Pass@K workflow."""
    
    def test_end_to_end_workflow(self):
        """Test the complete workflow from base metric to Pass@K evaluation."""
        # Create a simple base metric that checks if prediction equals "correct"
        def simple_metric_fn(golds, predictions, formatted_doc, **kwargs):
            return 1.0 if predictions[0] == "correct" else 0.0
        
        base_metric = SampleLevelMetric(
            metric_name="simple_test",
            category=MetricCategory.GENERATIVE,
            use_case=MetricUseCase.ACCURACY,
            higher_is_better=True,
            sample_level_fn=simple_metric_fn,
            corpus_level_fn=np.mean,
        )
        
        # Create Pass@K metrics
        passk_metrics = create_passk_metrics(base_metric, k_values=[1, 3], num_samples=10)
        
        # Create test data
        formatted_doc = Mock(spec=Doc)
        formatted_doc.get_golds.return_value = ["correct"]
        
        # Test Pass@1 with mixed predictions (5 samples, 1 correct)
        predictions_mixed = ["correct", "wrong", "wrong", "wrong", "wrong"]
        result_pass1 = passk_metrics[0].sample_level_fn(golds=["correct"], predictions=predictions_mixed, formatted_doc=formatted_doc)
        result_pass3 = passk_metrics[1].sample_level_fn(golds=["correct"], predictions=predictions_mixed, formatted_doc=formatted_doc)
        
        # Calculate expected values using estimate_pass_at_k
        expected_pass1 = estimate_pass_at_k(num_samples=5, num_correct=1, k=1)
        expected_pass3 = estimate_pass_at_k(num_samples=5, num_correct=1, k=3)
        
        # Verify exact values
        assert result_pass1 == pytest.approx(expected_pass1, abs=1e-10)
        assert result_pass3 == pytest.approx(expected_pass3, abs=1e-10)
        
        # Pass@3 should be higher than Pass@1 when there's at least one correct answer
        assert result_pass1 > 0.0
        assert result_pass3 > result_pass1
        
        # Test with all wrong predictions (5 samples, 0 correct)
        predictions_wrong = ["wrong", "wrong", "wrong", "wrong", "wrong"]
        result_pass1_wrong = passk_metrics[0].sample_level_fn(golds=["correct"], predictions=predictions_wrong, formatted_doc=formatted_doc)
        result_pass3_wrong = passk_metrics[1].sample_level_fn(golds=["correct"], predictions=predictions_wrong, formatted_doc=formatted_doc)
        
        # Calculate expected values (should be 0.0 for both)
        expected_pass1_wrong = estimate_pass_at_k(num_samples=5, num_correct=0, k=1)
        expected_pass3_wrong = estimate_pass_at_k(num_samples=5, num_correct=0, k=3)
        
        # Verify exact values
        assert result_pass1_wrong == pytest.approx(expected_pass1_wrong, abs=1e-10)
        assert result_pass3_wrong == pytest.approx(expected_pass3_wrong, abs=1e-10)
        
        # Both should be 0.0 when all predictions are wrong
        assert result_pass1_wrong == 0.0
        assert result_pass3_wrong == 0.0
    
    def test_integration_with_different_scenarios(self):
        """Test integration with various prediction scenarios."""
        def simple_metric_fn(golds, predictions, formatted_doc, **kwargs):
            return 1.0 if predictions[0] == "correct" else 0.0
        
        base_metric = SampleLevelMetric(
            metric_name="simple_test",
            category=MetricCategory.GENERATIVE,
            use_case=MetricUseCase.ACCURACY,
            higher_is_better=True,
            sample_level_fn=simple_metric_fn,
            corpus_level_fn=np.mean,
        )
        
        # Create Pass@K metrics
        passk_metrics = create_passk_metrics(base_metric, k_values=[1, 2, 5], num_samples=10)
        
        # Create test data
        formatted_doc = Mock(spec=Doc)
        formatted_doc.get_golds.return_value = ["correct"]
        
        # Test scenario: 10 samples, 3 correct
        predictions_scenario1 = ["correct", "correct", "correct", "wrong", "wrong", 
                               "wrong", "wrong", "wrong", "wrong", "wrong"]
        
        results = []
        expected_results = []
        for i, k in enumerate([1, 2, 5]):
            result = passk_metrics[i].sample_level_fn(golds=["correct"], predictions=predictions_scenario1, formatted_doc=formatted_doc)
            expected = estimate_pass_at_k(num_samples=10, num_correct=3, k=k)
            results.append(result)
            expected_results.append(expected)
            
            # Verify exact match
            assert abs(result - expected) < 1e-10, f"Pass@{k}: got {result}, expected {expected}"
        
        # Verify monotonicity: Pass@1 <= Pass@2 <= Pass@5
        assert results[0] <= results[1] <= results[2]
        assert expected_results[0] <= expected_results[1] <= expected_results[2]
        
        # Test scenario: 10 samples, 7 correct
        predictions_scenario2 = ["correct"] * 7 + ["wrong"] * 3
        
        for i, k in enumerate([1, 2, 5]):
            result = passk_metrics[i].sample_level_fn(golds=["correct"], predictions=predictions_scenario2, formatted_doc=formatted_doc)
            expected = estimate_pass_at_k(num_samples=10, num_correct=7, k=k)
            
            # Verify exact match
            assert abs(result - expected) < 1e-10, f"Scenario 2 Pass@{k}: got {result}, expected {expected}"
            
            # With 7 correct out of 10, all Pass@K should be very high
            assert result > 0.5
