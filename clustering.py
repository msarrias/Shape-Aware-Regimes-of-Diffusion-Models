import logging
import matplotlib.pyplot as plt
import numpy as np

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

def segment_cost(S: np.ndarray, a: int, b: int) -> float:
    length = b - a
    if length <= 1:
        return 0.0
    total = block_sum(S, a, b)
    return total / (length * length)


def weighted_segment_cost(D: np.ndarray, a: int, b: int, weight_exp: float = 1.0) -> float:
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


def _run_dp(
    D: np.ndarray,
    penalty: float,
    weight_exp: float = 0.0,
    record_costs: bool = False,
) -> tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
    """
    Fill the segmentation DP arrays.
    Returns (dp, prev, M), where M[k, t] = dp[k] + cost(k, t) + penalty
    """
    n = D.shape[0]

    if weight_exp == 0.0:
        S = compute_prefix_sums(D)
        cost_fn = lambda k, t: segment_cost(S, k, t)
    else:
        cost_fn = lambda k, t: weighted_segment_cost(D, k, t, weight_exp)

    dp = np.zeros(n + 1)
    prev = np.zeros(n + 1, dtype=int)
    dp[0] = -penalty

    M = np.full((n + 1, n + 1), np.nan) if record_costs else None

    for t in range(1, n + 1):
        best_cost = np.inf
        best_k = 0
        for k in range(t):
            cost = dp[k] + cost_fn(k, t) + penalty
            if M is not None:
                M[k, t] = cost
            if cost < best_cost:
                best_cost = cost
                best_k = k
        dp[t] = best_cost
        prev[t] = best_k

    return dp, prev, M


def _backtrack(prev: np.ndarray) -> list[int]:
    breakpoints: list[int] = []
    t = len(prev) - 1
    while t > 0:
        k = int(prev[t])
        breakpoints.append(k)
        t = k
    breakpoints.reverse()
    return breakpoints[1:]


def dp_clustering(
    D: np.ndarray,
    penalty: np.floating[Any],
    weight_exp: float = 0.0,
) -> list[int]:
    logging.debug("Penalty: {}".format(penalty))
    _, prev, _ = _run_dp(D, float(penalty), weight_exp=weight_exp)
    return _backtrack(prev)

def cluster_distance_matrix(distances: np.ndarray, 
                            method: str = "dp", 
                            weight_exp: float = 0.0, 
                            penalty_coeff: float = 1.0) -> list[int]:
    methods = ["dp", "rupture"]
    if method == "dp":
        return dp_clustering(distances, penalty=penalty_coeff * np.std(distances), weight_exp=weight_exp)
    else:
        raise ValueError(f"Only {methods} clustering methods are supported")


# This function was generated with Claude Code for debugging purposes
def plot_segment_dp(
    D: np.ndarray,
    penalty: Optional[float] = None,
    weight_exp: float = 0.0,
    ax=None,
):
    eff_penalty: float = float(penalty) if penalty is not None else float(0.5 * np.std(D))

    n = D.shape[0]
    dp, prev, M = _run_dp(D, eff_penalty, weight_exp=weight_exp, record_costs=True)
    assert M is not None
    segments = _backtrack(prev)

    if ax is None:
        _, ax = plt.subplots(figsize=(8, 7))

    im = ax.imshow(M, origin="lower", aspect="auto", cmap="viridis")
    ax.figure.colorbar(im, ax=ax, label="dp[k] + segment_cost(k, t) + penalty")

    ts = np.arange(1, n + 1)
    ax.plot(ts, prev[1:], ".", color="white", markersize=4, label="argmin prev[t]")

    path_t = [t for _, t in segments]
    path_k = [k for k, _ in segments]
    ax.plot(path_t, path_k, "o-", color="red", markersize=8,
            markerfacecolor="none", linewidth=1.5, label="backtracking path")
    print(path_t, path_k)

    ax.set_xlabel("t (segment end)")
    ax.set_ylabel("k (segment start)")
    ax.set_title(
        f"Segment DP cost matrix — {len(segments)} segments, "
        f"penalty={eff_penalty:.4g}, weight_exp={weight_exp:g}"
    )
    ax.legend(loc="upper left")

    return ax, segments, dp, prev, M

# Example usage
# segments = segment_dp(D, penalty=0.5)