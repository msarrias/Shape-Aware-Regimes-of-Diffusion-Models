import logging
import numpy as np
from scipy.stats import wasserstein_distance
from spectral import normalized_laplacian, unnormalized_laplacian
import scipy.linalg as la


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

def solve_and_sort_std_eigv_problem(
        matrix: np.ndarray,
):
    # eigvc[:, i] is the i-th eigenvector
    eigv, eigvc = la.eig(matrix)

    # la.eig returns complex types
    eigv = eigv.real
    eigvc = eigvc.real

    # la.eig is unordered
    idx = eigv.argsort()
    eigv = eigv[idx]
    eigvc = eigvc[:, idx]  # column-wise

    return eigv, eigvc

def eigen_decompose_job(
        W: np.ndarray,
        laplacian_type: str,
        norm:bool =True
):
    if laplacian_type not in ("normalized", "unnormalized"):
        raise ValueError("Unsupported laplacian_type")

    if laplacian_type == "normalized":
        L = normalized_laplacian(W=W)

    elif laplacian_type == "unnormalized":
        L = unnormalized_laplacian(W=W)
    else:
        raise ValueError("Unsupported laplacian_type")

    # Solve Eigen-problem
    eigvs, eigvecs = solve_and_sort_std_eigv_problem(matrix=L)

    if np.sum(eigvs < 1e-10) > 1:
        raise ValueError(
            "SGD requires a single connected component."
        )
    Phi = eigvecs[:, 1:]
    if norm:
        norm_Phi = normalize_eigenvectors(Phi=Phi)
        return norm_Phi

    return Phi

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
