"""
Density-peak clustering of a SAGD distance matrix.

Implements Rodriguez & Laio (2014)
"""

import logging
import matplotlib.pyplot as plt
import numpy as np

from dataclasses import dataclass
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from pathlib import Path


@dataclass
class DensityPeakResult:
    d_c: float
    rho: np.ndarray       # (n,) local density
    delta: np.ndarray     # (n,) distance to the closest point of higher density
    nneigh: np.ndarray    # (n,) index of that point, -1 for the global density maximum
    gamma: np.ndarray     # (n,) rho * delta
    centers: np.ndarray   # (k,) cluster center indices, ordered by descending gamma
    labels: np.ndarray    # (n,) labels in 0..k-1
    halo: np.ndarray      # (n,) bool, True where the point belongs to the cluster halo

    def to_dict(self, assign_halo: bool = False) -> dict[int, int]:
        """
        Map snapshot index -> cluster label. With `assign_halo`, points in the halo
         are reported as -1 instead of their cluster label.
        """
        labels = self.labels
        if assign_halo:
            labels = np.where(self.halo, -1, labels)
        return {int(i): int(label) for i, label in enumerate(labels)}


def check_distance_matrix(distances: np.ndarray) -> np.ndarray:
    """
    Validate a snapshot x snapshot distance matrix and return it as float.
    """
    D = np.asarray(distances, dtype=float)

    if D.ndim != 2 or D.shape[0] != D.shape[1]:
        raise ValueError(f"Expected a square distance matrix, got shape {D.shape}")
    if D.shape[0] < 2:
        raise ValueError("Need at least 2 points to cluster")
    if not np.allclose(D, D.T):
        raise ValueError("Distance matrix must be symmetric")
    if not np.allclose(np.diag(D), 0.0):
        raise ValueError("Distance matrix must have a zero diagonal")

    return D


def estimate_dc(D: np.ndarray, neighbor_frac: float = 0.02) -> float:
    """
    Pick d_c so that the average number of neighbours
    is 1-2% of the total number of points, i.e. the `neighbor_frac` quantile of the
    off-diagonal distances.
    """
    if not 0.0 < neighbor_frac < 1.0:
        raise ValueError(f"neighbor_frac must be in (0, 1), got {neighbor_frac}")

    n = D.shape[0]
    offdiag = D[~np.eye(n, dtype=bool)]
    d_c = float(np.quantile(offdiag, neighbor_frac))

    if d_c <= 0.0:
        # Degenerate: many exactly-zero distances. Fall back to the smallest positive one.
        positive = offdiag[offdiag > 0.0]
        if positive.size == 0:
            raise ValueError("All pairwise distances are zero, nothing to cluster")
        d_c = float(positive.min())

    return d_c


def local_density(D: np.ndarray, d_c: float, kernel: str = "gaussian") -> np.ndarray:
    """
    Local density rho_i. `cutoff` is Eq. 1 of the paper (number of points closer than
    d_c); `gaussian` is the exponential kernel the paper recommends when the number of
    points is small, which is the case for a SAGD matrix (~10^2 snapshots).
    Self-contributions are excluded in both cases.
    """
    kernels = ["gaussian", "cutoff"]

    if kernel == "gaussian":
        rho = np.exp(-((D / d_c) ** 2)).sum(axis=1) - 1.0
    elif kernel == "cutoff":
        rho = (D < d_c).sum(axis=1).astype(float) - 1.0
    else:
        raise ValueError(f"Only {kernels} kernels are supported")

    return rho


def density_order(rho: np.ndarray) -> np.ndarray:
    """
    Indices sorted by decreasing density, ties broken by index. This gives a strict
    total order, so "higher density" is never mutual and every point has a
    well-defined nearest neighbour of higher density.
    """
    return np.lexsort((np.arange(rho.size), -rho))


def min_higher_density_distance(
    D: np.ndarray,
    rho: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Eq. 2: delta_i is the distance to the closest point of higher density. For the
    global maximum, delta is conventionally max_j d_ij. Also returns nneigh, the index
    of that closer-and-denser point (-1 for the global maximum).
    """
    n = D.shape[0]
    order = density_order(rho)

    delta = np.zeros(n)
    nneigh = np.full(n, -1, dtype=int)

    # order[0] is the global density maximum; every other point has at least one
    # predecessor in `order` that is strictly denser.
    for rank, i in enumerate(order[1:], start=1):
        denser = order[:rank]
        best = denser[np.argmin(D[i, denser])]
        delta[i] = D[i, best]
        nneigh[i] = best

    peak = order[0]
    delta[peak] = D[peak].max()

    return delta, nneigh


def select_centers(
    rho: np.ndarray,
    delta: np.ndarray,
    n_clusters: int | None = None,
    rho_min: float | None = None,
    delta_min: float | None = None,
) -> np.ndarray:
    """
    Pick the cluster centers off the decision graph: points with both high rho and
    high delta. Precedence is explicit rho_min/delta_min thresholds, then n_clusters
    (top-k by gamma), then the automatic rule -- rank gamma = rho * delta in
    decreasing order and cut at its largest drop (paper Fig. 4B).

    Centers are returned ordered by descending gamma.
    """
    gamma = rho * delta
    order = np.argsort(-gamma, kind="stable")

    if rho_min is not None or delta_min is not None:
        mask = np.ones(rho.size, dtype=bool)
        if rho_min is not None:
            mask &= rho > rho_min
        if delta_min is not None:
            mask &= delta > delta_min
        centers = order[mask[order]]
        if centers.size == 0:
            raise ValueError(
                "No point passes the rho_min/delta_min thresholds; "
                "loosen them or read them off the decision graph"
            )
        return centers

    if n_clusters is not None:
        if not 1 <= n_clusters <= rho.size:
            raise ValueError(
                f"n_clusters must be in [1, {rho.size}], got {n_clusters}"
            )
        return order[:n_clusters]

    ranked = gamma[order]
    # The gap between consecutive ranked gammas; cutting after the largest one
    # separates the peaks from the bulk. k=1 if the top point dominates.
    gaps = ranked[:-1] - ranked[1:]
    k = int(np.argmax(gaps)) + 1

    logging.debug("Ranked gamma: {}".format(ranked))
    logging.debug("Largest gamma gap after rank {}, selecting {} centers".format(k, k))

    return order[:k]


def assign_labels(
    rho: np.ndarray,
    nneigh: np.ndarray,
    centers: np.ndarray,
) -> np.ndarray:
    """
    Single-pass assignment: walking points in order of decreasing density, each
    non-center takes the label of its nearest neighbour of higher density (which has
    necessarily already been labelled). No objective function is iterated.
    """
    n = rho.size
    labels = np.full(n, -1, dtype=int)
    labels[centers] = np.arange(centers.size)

    for i in density_order(rho):
        if labels[i] == -1:
            labels[i] = labels[nneigh[i]]

    assert (labels >= 0).all(), "Every point should have been assigned a cluster"

    return labels


def compute_halo(
    D: np.ndarray,
    labels: np.ndarray,
    rho: np.ndarray,
    d_c: float,
) -> np.ndarray:
    """
    Split each cluster into a core and a halo. The border region of a cluster is the
    set of its points lying within d_c of a point assigned to another cluster; rho_b
    is the highest density in that border region, and points below it are halo
    (suitable to be considered noise).
    """
    n = D.shape[0]
    halo = np.zeros(n, dtype=bool)

    n_clusters = labels.max() + 1
    if n_clusters < 2:
        # A single cluster has no border region.
        return halo

    close = D <= d_c
    np.fill_diagonal(close, False)
    other_cluster = labels[:, None] != labels[None, :]
    on_border = (close & other_cluster).any(axis=1)

    for c in range(n_clusters):
        in_cluster = labels == c
        border = in_cluster & on_border
        if not border.any():
            continue
        rho_b = rho[border].max()
        halo |= in_cluster & (rho < rho_b)

    return halo


def density_peaks(
    distances: np.ndarray,
    d_c: float | None = None,
    neighbor_frac: float = 0.02,
    kernel: str = "gaussian",
    n_clusters: int | None = None,
    rho_min: float | None = None,
    delta_min: float | None = None,
) -> DensityPeakResult:
    """
    Run density-peak clustering on a snapshot x snapshot distance matrix and return the
    full result, including the quantities needed for the decision graph.

    `d_c` overrides the `neighbor_frac` rule of thumb. Center selection follows the
    precedence documented in `select_centers`.
    """
    D = check_distance_matrix(distances)

    if d_c is None:
        d_c = estimate_dc(D, neighbor_frac=neighbor_frac)
    elif d_c <= 0.0:
        raise ValueError(f"d_c must be positive, got {d_c}")

    rho = local_density(D, d_c, kernel=kernel)
    delta, nneigh = min_higher_density_distance(D, rho)
    centers = select_centers(
        rho, delta,
        n_clusters=n_clusters,
        rho_min=rho_min,
        delta_min=delta_min,
    )
    labels = assign_labels(rho, nneigh, centers)
    halo = compute_halo(D, labels, rho, d_c)

    logging.debug(
        "d_c={:.6g}, {} centers at {}, {} halo points".format(
            d_c, centers.size, centers.tolist(), int(halo.sum())
        )
    )

    return DensityPeakResult(
        d_c=d_c,
        rho=rho,
        delta=delta,
        nneigh=nneigh,
        gamma=rho * delta,
        centers=centers,
        labels=labels,
        halo=halo,
    )


def cluster_sagd_matrix(
    distances: np.ndarray,
    d_c: float | None = None,
    neighbor_frac: float = 0.02,
    kernel: str = "gaussian",
    n_clusters: int | None = None,
    rho_min: float | None = None,
    delta_min: float | None = None,
    assign_halo: bool = False,
) -> dict[int, int]:
    """
    Cluster a SAGD distance matrix and return {snapshot index -> cluster label}.

    Keys are row indices into `distances`; row i corresponds to the i-th snapshot time
    (times are not stored alongside `SAGD.jbl`, so the caller maps them back). Labels
    run 0..k-1 ordered by descending gamma, or -1 for halo points if `assign_halo`.

    Use `density_peaks` instead if you also want rho/delta for the decision graph.
    """
    result = density_peaks(
        distances,
        d_c=d_c,
        neighbor_frac=neighbor_frac,
        kernel=kernel,
        n_clusters=n_clusters,
        rho_min=rho_min,
        delta_min=delta_min,
    )
    return result.to_dict(assign_halo=assign_halo)


def _figure_and_axes(
    ax: Axes | None,
    figsize: tuple[float, float],
) -> tuple[Figure, Axes]:
    """
    Draw into the caller's axes if given, otherwise make a new figure of `figsize`.
    """
    if ax is None:
        return plt.subplots(figsize=figsize)

    fig = ax.get_figure(root=True)
    assert fig is not None, "The given axes must belong to a figure"

    return fig, ax


def draw_markers(ax: Axes, markers: dict) -> None:
    """
    Draw labelled reference times (t_SAGD, t*_SAGD, ...) as dashed vertical lines on
    a time axis. `markers` maps a legend label to either a time, or a (time, color)
    pair if you want to fix the color; a None time is skipped.
    """
    default_colors = ["orange", "red", "magenta", "cyan", "lime"]

    for i, (label, spec) in enumerate(markers.items()):
        if isinstance(spec, (tuple, list)):
            t, color = spec
        else:
            t, color = spec, default_colors[i % len(default_colors)]
        if t is None:
            continue

        ax.axvline(t, color=color, linestyle="--", alpha=0.8,
                   label=f"{label} = {t:.2f}")


def plot_decision_graph(
    result: DensityPeakResult,
    ts: np.ndarray | None = None,
    ax: Axes | None = None,
    annotate: bool = True,
    save_path: Path | None = None,
) -> tuple[Figure, Axes]:
    """
    The decision graph of the paper (Fig. 1B): delta against rho. Cluster centers are
    the points that stand out towards the top right -- high density and anomalously far
    from any denser point. Isolated outliers sit top left (high delta, low rho).

    Pass `ts` (the snapshot times, row i of the distance matrix <-> ts[i]) to annotate
    the centers with their diffusion time instead of their snapshot index.

    Returns (fig, ax); writes a PNG only if `save_path` is given.
    """
    fig, ax = _figure_and_axes(ax, (6, 5))

    is_center = np.zeros(result.rho.size, dtype=bool)
    is_center[result.centers] = True

    ax.scatter(
        result.rho[~is_center], result.delta[~is_center],
        s=25, c="0.4", alpha=0.7, edgecolors="none", label="snapshots",
    )
    ax.scatter(
        result.rho[is_center], result.delta[is_center],
        s=120, c=np.arange(result.centers.size)[np.argsort(result.centers)],
        cmap="tab10", vmin=0, vmax=9,
        edgecolors="black", linewidths=1.0, zorder=3, label="cluster centers",
    )

    if annotate:
        for i in result.centers:
            text = f"t={ts[i]:.2f}" if ts is not None else f"{i}"
            ax.annotate(
                text,
                (float(result.rho[i]), float(result.delta[i])),
                textcoords="offset points", xytext=(8, 4),
                fontsize=9, fontweight="bold",
            )

    ax.set_xlabel(r"$\rho$  (local density)", fontsize=12)
    ax.set_ylabel(r"$\delta$  (distance to closest denser point)", fontsize=12)
    ax.set_title(
        r"Decision graph  ($d_c$ = {:.3g}, {} clusters)".format(
            result.d_c, result.centers.size
        ),
        fontsize=12,
    )
    ax.legend(frameon=False, fontsize=9)

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")

    return fig, ax


def plot_labels_over_time(
    result: DensityPeakResult,
    ts: np.ndarray | None = None,
    markers: dict | None = None,
    assign_halo: bool = False,
    ax: Axes | None = None,
    save_path: Path | None = None,
) -> tuple[Figure, Axes]:
    """
    The cluster label of every snapshot along the diffusion trajectory. Pass `ts` to
    put actual diffusion time on the x axis (running from t=T on the left down to
    t~0 on the right); without it the x axis is the snapshot index.

    `markers` overlays labelled reference times -- e.g.
    `{"$t_{SAGD}$": 2.94, "$t^*_{SAGD}$": 1.15}` -- see `draw_markers`.

    Returns (fig, ax); writes a PNG only if `save_path` is given.
    """
    fig, ax = _figure_and_axes(ax, (9, 2.2))

    labels = np.where(result.halo, -1, result.labels) if assign_halo else result.labels
    x = np.asarray(ts, dtype=float) if ts is not None else np.arange(labels.size)

    ax.scatter(
        x, np.zeros(labels.size),
        c=labels, cmap="tab10", vmin=0, vmax=9,
        marker="|", s=400,
    )

    if markers:
        draw_markers(ax, markers)
        ax.legend(frameon=False, fontsize=9, loc="upper right")

    ax.set_yticks([])
    ax.set_xlabel("diffusion time $t$" if ts is not None else "snapshot index",
                  fontsize=12)
    ax.set_title("Density-peak label per snapshot", fontsize=12)
    if ts is not None:
        ax.invert_xaxis()

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")

    return fig, ax
