import logging
import numpy as np
from scipy.stats import wasserstein_distance
from lib.spectral import normalized_laplacian, unnormalized_laplacian, solve_and_sort_std_eigv_problem


_LAPLACIAN_BUILDERS = {
    "normalized": normalized_laplacian,
    "unnormalized": unnormalized_laplacian,
}

def normalize_eigenvectors(
        Phi: np.ndarray,
):
    """
    Min-max normalize each eigenvector (column) independently to [0, 1].
    """
    col_min = Phi.min(axis=0)          # (n-1,)
    col_max = Phi.max(axis=0)          # (n-1,)
    col_range = col_max - col_min      # (n-1,)
    
    # avoid division by zero for constant columns
    col_range = np.where(col_range == 0, 1.0, col_range)
    
    return (Phi - col_min) / col_range


def eigen_decompose_job(
        W: np.ndarray,
        laplacian_type: str,
        norm: bool = True
):
    try:
        build_laplacian = _LAPLACIAN_BUILDERS[laplacian_type]
    except KeyError:
        raise ValueError(f"Unsupported laplacian_type: {laplacian_type!r}")

    L = build_laplacian(W=W)
    eigvs, eigvecs = solve_and_sort_std_eigv_problem(matrix=L)

    if eigvs[1] < 1e-10:
        raise ValueError("SGD requires a single connected component.")

    Phi = eigvecs[:, 1:]
    return normalize_eigenvectors(Phi=Phi) if norm else Phi


def compute_sgd(
        Phi1: np.ndarray,
        Phi2: np.ndarray,
):
    Ni, Nj = Phi1.shape[1], Phi2.shape[1]
    Mij    = min(Ni, Nj)
    signs = [(1, 1), (1, -1)]
    cum_sum = 0
    for k in range(Mij):
        best = float('inf')
        for s1, s2 in signs:
            v1 = np.sort(s1 * Phi1[:, k])
            v2 = np.sort(s2 * Phi2[:, k])
            d  = wasserstein_distance(v1, v2)
            if d < best:
                best = d
        cum_sum += best
    return cum_sum / (Mij - 1) if Mij > 1 else 0.0
