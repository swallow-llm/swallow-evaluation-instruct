
"""
Tests for Maj@K metric solutions.
"""

import pytest
import random
from typing import Dict, Tuple

from lighteval.metrics.maj_at_k_solutions import (
    maj_at_k_exact_dp_scipy,
    maj_at_k_bruteforce,
)


def generate_test_case(N: int, M: int, max_count: int, num_correct: int, seed: int = None) -> Tuple[Dict[str, int], Dict[str, bool]]:
    """
    Generate test case with specified parameters.
    
    Args:
        N: Total number of samples
        M: Number of classes
        max_count: Maximum count for any single class
        num_correct: Number of classes that are correct
        seed: Random seed for reproducibility
        
    Returns:
        Tuple of (counts_dict, correct_dict)
    """
    if seed is not None:
        random.seed(seed)
    
    # Generate M class names
    keys = [f"class_{i}" for i in range(M)]
    
    # Generate counts that sum to N
    if M == 1:
        counts = [N]
    else:
        # Start with minimum count of 1 for each class
        counts = [1] * M
        remaining = N - M
        
        # Distribute remaining counts randomly, respecting max_count
        for _ in range(remaining):
            valid_indices = [i for i in range(M) if counts[i] < max_count]
            if valid_indices:
                idx = random.choice(valid_indices)
                counts[idx] += 1
            else:
                # If all classes are at max_count, distribute evenly
                idx = random.randint(0, M - 1)
                counts[idx] += 1
    
    counts_dict = {keys[i]: counts[i] for i in range(M)}
    
    # Generate correct_dict with exactly num_correct True values
    correct_keys = random.sample(keys, num_correct)
    correct_dict = {k: (k in correct_keys) for k in keys}
    
    return counts_dict, correct_dict

class TestMajAtKExactDP:
    """Test maj_at_k_exact_dp_scipy against brute-force ground truth."""
    
    @pytest.mark.parametrize("N,M,max_count,num_correct", [
        # Basic cases
        (5, 2, 3, 1),   # Simple case with one correct class
        (6, 3, 3, 0),   # All classes incorrect
        (7, 3, 4, 2),   # Two correct classes
        (8, 4, 3, 1),   # Many classes, one correct
        (9, 3, 5, 3),   # All classes correct
        (10, 5, 3, 2),  # Maximum N with multiple classes
        
        # Edge cases
        (4, 1, 4, 1),   # Single class, correct
        (4, 1, 4, 0),   # Single class, incorrect
        (6, 6, 1, 3),   # Many small classes
        (8, 2, 6, 1),   # Unbalanced classes
    ])
    def test_parametrized_cases(self, N: int, M: int, max_count: int, num_correct: int):
        """Test with parametrized cases covering various scenarios."""
        for seed in range(3):  # Test multiple random instances for each parameter set
            counts_dict, correct_dict = generate_test_case(N, M, max_count, num_correct, seed)
            
            # Test multiple K values
            K_values = [1, N//2, N] if N > 2 else [1, N]
            K_values = sorted(set(k for k in K_values if 1 <= k <= N))
            
            for K in K_values:
                exact_result = maj_at_k_exact_dp_scipy(counts_dict, correct_dict, K)
                bruteforce_result = maj_at_k_bruteforce(counts_dict, correct_dict, K)
                
                assert exact_result == pytest.approx(bruteforce_result, abs=1e-7), \
                    f"N={N}, M={M}, max_count={max_count}, num_correct={num_correct}, K={K}, seed={seed}: " \
                    f"exact={exact_result}, bruteforce={bruteforce_result}"
    
    def test_all_false_cases(self):
        """Test cases where all classes are incorrect (num_correct=0)."""
        test_params = [
            (5, 2, 3),   # N=5, M=2, max_count=3
            (7, 3, 3),   # N=7, M=3, max_count=3
            (10, 4, 4),  # N=10, M=4, max_count=4
        ]
        
        for N, M, max_count in test_params:
            counts_dict, correct_dict = generate_test_case(N, M, max_count, num_correct=0, seed=42)
            
            for K in [1, N//2, N]:
                if 1 <= K <= N:
                    exact_result = maj_at_k_exact_dp_scipy(counts_dict, correct_dict, K)
                    bruteforce_result = maj_at_k_bruteforce(counts_dict, correct_dict, K)
                    
                    # When all classes are incorrect, result should be 0.0
                    assert exact_result == 0.0
                    assert bruteforce_result == 0.0
                    assert exact_result == pytest.approx(bruteforce_result, abs=1e-7)
    
    @pytest.mark.parametrize("num_correct", [1, 2, 3])
    def test_few_correct_cases(self, num_correct: int):
        """Test cases with 1-3 correct classes."""
        N, M, max_count = 9, 5, 3
        counts_dict, correct_dict = generate_test_case(N, M, max_count, num_correct, seed=123)
        
        # Verify we have the expected number of correct classes
        actual_correct = sum(correct_dict.values())
        assert actual_correct == num_correct
        
        for K in [1, 3, 5, 9]:
            exact_result = maj_at_k_exact_dp_scipy(counts_dict, correct_dict, K)
            bruteforce_result = maj_at_k_bruteforce(counts_dict, correct_dict, K)
            
            assert exact_result == pytest.approx(bruteforce_result, abs=1e-7), \
                f"num_correct={num_correct}, K={K}: exact={exact_result}, bruteforce={bruteforce_result}"
    
    @pytest.mark.parametrize("M", [1, 2, 3, 4, 5])
    def test_wide_range_of_M(self, M: int):
        """Test wide range of M (number of classes)."""
        N = min(10, M + 3)  # Ensure N is reasonable for the given M
        max_count = N // M + 2
        
        for num_correct in [0, 1, min(M, 2)]:
            counts_dict, correct_dict = generate_test_case(N, M, max_count, num_correct, seed=M*10 + num_correct)
            
            for K in [1, N//2, N]:
                if 1 <= K <= N:
                    exact_result = maj_at_k_exact_dp_scipy(counts_dict, correct_dict, K)
                    bruteforce_result = maj_at_k_bruteforce(counts_dict, correct_dict, K)
                    
                    assert exact_result == pytest.approx(bruteforce_result, abs=1e-7), \
                        f"M={M}, N={N}, num_correct={num_correct}, K={K}: " \
                        f"exact={exact_result}, bruteforce={bruteforce_result}"
    
    def test_edge_cases(self):
        """Test specific edge cases."""
        # Single sample cases
        for is_correct in [True, False]:
            counts_dict = {"a": 1}
            correct_dict = {"a": is_correct}
            K = 1
            
            exact_result = maj_at_k_exact_dp_scipy(counts_dict, correct_dict, K)
            bruteforce_result = maj_at_k_bruteforce(counts_dict, correct_dict, K)
            expected = 1.0 if is_correct else 0.0
            
            assert exact_result == expected
            assert bruteforce_result == expected
            assert exact_result == pytest.approx(bruteforce_result, abs=1e-7)
        
        # K equals N case
        counts_dict, correct_dict = generate_test_case(N=6, M=3, max_count=3, num_correct=1, seed=999)
        K = 6
        
        exact_result = maj_at_k_exact_dp_scipy(counts_dict, correct_dict, K)
        bruteforce_result = maj_at_k_bruteforce(counts_dict, correct_dict, K)
        
        assert exact_result == pytest.approx(bruteforce_result, abs=1e-7)


class TestHelperFunctions:
    """Test helper functions."""
    
    def test_generate_test_case_constraints(self):
        """Test that generate_test_case respects constraints."""
        N, M, max_count, num_correct = 8, 3, 4, 2
        counts_dict, correct_dict = generate_test_case(N, M, max_count, num_correct, seed=42)
        
        # Check constraints
        assert sum(counts_dict.values()) == N
        assert len(counts_dict) == M
        assert all(1 <= count <= max_count for count in counts_dict.values())
        assert sum(correct_dict.values()) == num_correct
        assert set(counts_dict.keys()) == set(correct_dict.keys())
    
    def test_generate_test_case_reproducibility(self):
        """Test that generate_test_case is reproducible with same seed."""
        params = (7, 3, 3, 1, 123)
        counts1, correct1 = generate_test_case(*params)
        counts2, correct2 = generate_test_case(*params)
        
        assert counts1 == counts2
        assert correct1 == correct2
