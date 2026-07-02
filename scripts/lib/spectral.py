import numpy as np
import scipy.linalg as la


def unnormalized_laplacian(W):
    D = np.diag(np.sum(W, axis=1))
    return D - W


def normalized_laplacian(W):
    d = np.sum(W, axis=1)
    d_inv_sqrt = 1.0 / np.sqrt(d)
    D_inv_sqrt = np.diag(d_inv_sqrt)
    # L_sym = I - D^(-1/2) W D^(-1/2)
    return np.eye(len(W)) - D_inv_sqrt @ W @ D_inv_sqrt


def solve_and_sort_std_eigv_problem(matrix):
    """Solves the eigenvalue problem and returns sorted (values, vectors)."""
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


def scale_and_shift(arr):
    v_min, v_max = arr.min(), arr.max()
    return (arr - v_min) / (v_max - v_min) if v_max != v_min else arr


def normalize(list_values, norm_type, Vol=None):
    arr = np.asarray(list_values)
    if norm_type == "scale_and_shift":
        return scale_and_shift(arr)

    if norm_type == "log_scale_and_shift":
        log_ctd = np.log1p(arr)
        return scale_and_shift(log_ctd)

    if norm_type == "norm_wrt_volume":
        if Vol is None or Vol == 0:
            raise ValueError("Vol must be provided and non-zero for 'norm_wrt_volume'")
        return arr / Vol

    if norm_type == "norm_wrt_avg_ctd":
        arr_mean = np.mean(arr)
        if arr_mean == 0:
            raise ValueError("Mean of input values must be non-zero for 'norm_wrt_avg_ctd'")
        return arr / arr_mean

    raise ValueError("Unsupported norm_type")
    