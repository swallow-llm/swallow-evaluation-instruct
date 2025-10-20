import numpy as np


def estimate_pass_at_k(num_samples: int, num_correct: int, k: int) -> float:
    """
    Estimates pass@k for a single problem.
    Ref. https://arxiv.org/abs/2107.03374 (Formula 1)
    """
    if num_samples - num_correct < k:
        return 1.0
    return 1.0 - np.prod(1.0 - k / np.arange(num_samples - num_correct + 1, num_samples + 1))