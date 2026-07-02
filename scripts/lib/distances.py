import numpy as np
from lib.spectral import solve_and_sort_std_eigv_problem, normalized_laplacian, unnormalized_laplacian


def ctd_matrix(
    W,
    laplacian_type="unnormalized"
):
    """Calculates the Commute Time Distance (CTD) matrix."""
    d = np.sum(W, axis=1)
    Vol = np.sum(d)

    if laplacian_type not in ("normalized", "unnormalized"):
        raise ValueError("Unsupported laplacian_type")

    if laplacian_type == "normalized":
        L = normalized_laplacian(W)

    elif laplacian_type == "unnormalized":
        L = unnormalized_laplacian(W)

    # Solve Eigen-problem
    eigvs, eigvecs = solve_and_sort_std_eigv_problem(L)

    if np.sum(eigvs < 1e-10) > 1:
        raise ValueError(
            "CTD requires a single connected component."
        )

    # Skip the first eigenvalue/vector
    # eigvecs are column-wise
    vals = eigvs[1:]
    Phi = eigvecs[:, 1:]

    if laplacian_type == "unnormalized":
        # Standard CTE: sqrt(Vol) * Phi * diag(1/sqrt(Lambda))
        inv_sqrt_vals = 1.0 / np.sqrt(vals + 1e-10)
        CTE = np.sqrt(Vol) * (Phi * inv_sqrt_vals)  # broadcast

    elif laplacian_type == "normalized":
        # Normalized CTE: sqrt(Vol) * D^(-1/2) * Phi * diag(1/sqrt(Lambda))
        d_inv_sqrt = 1.0 / np.sqrt(d + 1e-12)
        inv_sqrt_vals = 1.0 / np.sqrt(vals + 1e-10)
        # Apply degree scaling to the eigenvectors
        CTE = np.sqrt(Vol) * (d_inv_sqrt[:, np.newaxis] * Phi) * inv_sqrt_vals

    # Squared Euclidean Distance
    sq_norms = np.sum(CTE**2, axis=1)
    C = sq_norms[:, np.newaxis] + sq_norms[np.newaxis, :] - 2 * np.dot(CTE, CTE.T)

    return C
