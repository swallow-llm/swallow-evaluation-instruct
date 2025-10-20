from typing import Any, Dict
from collections import defaultdict, Counter
from scipy.special import comb as sp_comb
import random

from math import comb
from itertools import combinations

def maj_at_k_exact_dp_scipy(counts_dict: Dict[Any, int], correct_dict: Dict[Any, bool], K: int):
    """
    Exact Maj@K (plurality, uniform tie-break) via Dynamic Programming (DP).
    Refer to [Maj@K の計算式](./docs/source/majoirity-at-k-metric.mdx) for the mathematical formulation.
    
    SciPy is used to precompute binomial coefficients comb(Ni, j) and denominator comb(N, K).    
      - use_exact_int=False (default): float64 binomials (fast). Good for N<=128.
      - use_exact_int=True: Python big-int binomials for full exactness (slower).
    Args:
        counts_dict: {answer: frequency}
        correct_dict: {answer: True/False} (missing -> False)
        K: 1..N
    Returns:
        Majority@K accuracy (float)
    """
    
    # 1) 入力の正規化（クラスごとの総数 Ns と正解フラグ Cs を作る）
    items = [(k, int(v), bool(correct_dict.get(k, False)))
             for k, v in counts_dict.items() if v > 0]
    items.sort(key=lambda x: x[0])       # 再現性確保のためにソート
    lst_num_freqs = [v for _, v, _ in items]        # N_m
    lst_is_correct = [c for _, _, c in items]       # True if {m∈C}
    N = sum(lst_num_freqs)                          # 試行回数 N
    M = len(lst_num_freqs)                          # クラス数 M
    assert 1 <= K <= N

    # 2) comb(Ni, j) をキャッシュ化
    binoms = []
    for N_m in lst_num_freqs:
        jmax = min(N_m, K)
        vals = [float(sp_comb(N_m, j, exact=False)) for j in range(jmax+1)]
        binoms.append(vals)

    # 分母 comb(N, K)
    denom = float(sp_comb(N, K, exact=False))

    # 3) DP 初期化：F_0(0, -1, 0, 0) = 1
    dp_table = defaultdict(float)
    dp_table[(0, -1, 0, 0)] = 1.0

    # 4) 各クラス i を順に畳み込む
    for i in range(M):
        N_m = lst_num_freqs[i]
        is_corr = 1 if lst_is_correct[i] else 0 # 1_{i∈C}
        vals = binoms[i]                 # [comb(N_i, j)]
        next_dp_table = defaultdict(float)

        # 4a) 現在の全状態 (k, m, t, c_w) を走査
        for (k_used, m, t, cw), ways in dp_table.items():
            max_j = min(N_m, K - k_used)  # j の上限

            # 4b) j=0..max_j を割り当て（遷移）
            for j in range(max_j + 1):
                new_k = k_used + j

                # 4c) (m,t,cw) の更新
                if m < 0: # 最初のクラス
                    new_m = j
                    new_t = 1
                    new_cw = is_corr
                else: # 2番目以降のクラス
                    if j > m: # 最頻クラスを更新
                        new_m = j; new_t = 1; new_cw = is_corr
                    elif j == m: # 最頻クラスを追加
                        new_m = m; new_t = t + 1; new_cw = cw + is_corr
                    else: # 最頻クラスを維持
                        new_m = m; new_t = t; new_cw = cw

                # 4d) F_{i+1}(...) += F_i(...) * comb(N_i, j)
                next_dp_table[(new_k, new_m, new_t, new_cw)] += ways * vals[j]

        # 次のクラスに進む．古いテーブルは不要なので上書き
        dp_table = next_dp_table

    # 5) 終了：k=K の状態の重みを c_w/t を掛けて合算し，最後に comb(N,K) で割る
    total_num = 0.0
    for (k_used, m, t, cw), ways in dp_table.items():
        if k_used != K or t <= 0 or cw <= 0:
            continue
        total_num += ways * (cw / t)

    return float(total_num / denom)


# Monte Carlo baseline (sampling without replacement)
def maj_at_k_monte_carlo(counts_dict: Dict[Any, int], correct_dict: Dict[Any, bool], K: int, trials: int = 40000, random_seed: int = 0):
    rng = random.Random(random_seed)
    population = []
    for k, f in counts_dict.items():
        population.extend([k]*int(f))
    N = len(population)
    assert 1 <= K <= N
    win = 0.0
    for t in range(trials):
        sample = rng.sample(population, K)
        freqs = Counter(sample)
        top = max(freqs.values())
        majority_classes = [c for c, freq in freqs.items() if freq == top]
        cw = sum(1 for c in majority_classes if correct_dict.get(c, False))
        if cw > 0:
            win += cw / len(majority_classes)
    return win / trials

# Brute-force baseline (sampling without replacement)
# This can be used for small N and K to verify the DP implementation.
def maj_at_k_bruteforce(counts_dict: Dict[Any, int], correct_dict: Dict[Any, bool], K: int):
    population = []
    for k, f in counts_dict.items():
        population.extend([k]*int(f))
    N = len(population)
    assert 1 <= K <= N
    win = 0.0
    for idxs in combinations(range(N), K):
        sample = [population[i] for i in idxs]
        cnt = Counter(sample)
        top = max(cnt.values())
        winners = [a for a, c in cnt.items() if c == top]
        t = len(winners)
        cw = sum(1 for a in winners if correct_dict.get(a, False))
        if cw > 0:
            win += cw / t
    return win / comb(N, K)