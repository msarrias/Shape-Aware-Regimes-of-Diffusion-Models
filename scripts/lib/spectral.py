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
