import logging
import matplotlib.pyplot as plt
import numpy as np

from functools import partial
from typing import Any, Optional


def compute_prefix_sums(D: np.ndarray) -> np.ndarray: 
    return D.cumsum(axis=0).cumsum(axis=1)


def block_sum(S: np.ndarray, a: int, b: int):
    """
    Sum over D[a:b, a:b]
    """
    total = S[b-1, b-1]
    if a > 0:
        total = total - S[a - 1, b - 1] - S[b - 1, a - 1] + S[a - 1, a - 1]
    return total


def normalized_block_sum(S: np.ndarray, a: int, b: int) -> float:
    length = b - a
    if length <= 1:
        return 0.0
    total = block_sum(S, a, b)
    return total / (length * length)


def weighted_block_sum(D: np.ndarray, a: int, b: int, weight_exp: float) -> float:
    """
    Weighted average of D[a:b, a:b]. Pair (i, j) gets weight
    ((i - a + 1) * (j - a + 1)) ** weight_exp, so distances near the end of
    the segment contribute more.
    """
    length = b - a
    if length <= 1:
        return 0.0
    block = D[a:b, a:b]
    p = np.arange(1, length + 1, dtype=float) ** weight_exp
    weights = np.outer(p, p)
    return float((block * weights).sum() / weights.sum())


def run_dp(
    D: np.ndarray,
    max_k: int,
    weight_exp: float = 0.0,
    record_costs: bool = False,
) -> tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
    """
    Fill the segmentation DP arrays.
    Returns (dp, prev, M), where M[k, t] = dp[k] + cost(k, t) + penalty
    """
    n = D.shape[0]
    if weight_exp == 0.0:
        # With prefix sum of squares, normalized_block_sum corresponds to within-cluster dispersion
        S = compute_prefix_sums(D ** 2)
        cost_fn = lambda s, t: normalized_block_sum(S, s, t)
    else:
        cost_fn = lambda s, t: weighted_block_sum(D, s, t, weight_exp)

    # dp[k, n] -- Minimum-cost clustering of [0, n] into k segments
    dp = np.full((max_k + 1, n + 1), np.inf)
    prev = np.zeros((max_k + 1, n + 1), dtype=int)
    
    dp[0, 0] = 0

    cost_matrix = np.full((max_k + 1, n + 1, n + 1), np.nan) if record_costs else None

    for k in range(1, max_k + 1):
        for t in range(1, n + 1):
            for s in range(k - 1, t):
                cost = dp[k - 1, s] + cost_fn(s, t)
                if cost < dp[k, t]:
                    dp[k, t] = cost
                    prev[k, t] = s

    return dp, prev, cost_matrix


def bic_selection(dp: np.ndarray, max_k: int, n: int) -> int:
    bic_scores = []

    for k in range(1, max_k + 1):
        rss = dp[k, n]
        bic = n * np.log(rss / n + 1e-12) + k * np.log(n)
        bic_scores.append(bic)
    
    logging.debug("BIC scores: {}".format(bic_scores))

    return int(np.argmin(bic_scores)) + 1


def backtrack(prev: np.ndarray, opt_k: int, n: int) -> list[int]:
    breakpoints: list[int] = []
    t = n

    for k in range(opt_k, 0, -1):
        s = prev[k, t]
        breakpoints.append(s)
        t = s
    breakpoints.reverse()
    return breakpoints[1:]


def dp_clustering(
    D: np.ndarray,
    max_k: int = 10,
    weight_exp: float = 0.0,
) -> list[int]:
    n = D.shape[0]

    min_time_idx = n // 2
    subD = D[min_time_idx:, min_time_idx:]
    subn = subD.shape[0]

    max_k = min(max_k, subn // 2)

    dp, prev, _ = run_dp(subD, max_k=max_k, weight_exp=weight_exp)
    opt_k = bic_selection(dp, max_k, subn)
    return [x + min_time_idx for x in backtrack(prev, opt_k, subn)]


def cluster_distance_matrix(distances: np.ndarray, 
                            method: str = "dp") -> list[int]:
    methods = ["dp", "rupture"]
    if method == "dp":
        return dp_clustering(distances)
    else:
        raise ValueError(f"Only {methods} clustering methods are supported")