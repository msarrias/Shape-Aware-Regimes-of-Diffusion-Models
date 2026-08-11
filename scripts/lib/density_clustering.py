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

from lib.adaptive_knn import AdaptiveKNNGraph


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
    Pick d_c so that the average number of neighbours is 1-2% of the total number of points
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


def local_density(
    D: np.ndarray,
    d_c: float,
    kernel: str = "gaussian",
    min_k: int = 5,
    sigma: float | None = None,
) -> np.ndarray:
    """
    Local density of the point.
    `cutoff` counts the points closer than d_c. `gaussian` and `knn` both weight a pair
    with gaussian kernel. `gaussian` sums those
    weights over every other point, `knn` only over the edges of the adaptive KNN graph. 
    
    Left unset, `sigma` is d_c / sqrt(2) for
    `gaussian`, recovering the paper's exp(-(d / d_c)^2), and the median distance to the
    k-th neighbour of that graph for `knn`.
    """
    kernels = ["gaussian", "cutoff", "knn"]

    graph = AdaptiveKNNGraph(dist_matrix=D, min_k=min_k, kernel="gaussian")

    if kernel == "gaussian":
        rho = graph.gaussian_kernel(
            sigma=sigma if sigma is not None else d_c / np.sqrt(2.0)
        ).sum(axis=1) - 1.0
    elif kernel == "cutoff":
        rho = (D < d_c).sum(axis=1).astype(float) - 1.0
    elif kernel == "knn":
        rho = graph.compute_W(sigma=sigma).sum(axis=1)
    else:
        raise ValueError(f"Only {kernels} kernels are supported")

    return rho


def density_order(rho: np.ndarray) -> np.ndarray:
    """
    Indices sorted by decreasing density, ties broken by index.
    """
    return np.lexsort((np.arange(rho.size), -rho))


def min_higher_density_distance(
    D: np.ndarray,
    rho: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    The distance to the closest point of higher density and its index. For the
    global maximum, delta is conventionally max_j d_ij.
    """
    n = D.shape[0]
    order = density_order(rho)

    delta = np.zeros(n)
    nneigh = np.full(n, -1, dtype=int)

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
    n_clusters: int | None = None
) -> np.ndarray:
    """
    Pick the cluster centers off the decision graph based on rank gamma = rho * delta.
    If `n_clusters` is not given, sort in decreasing order and cut at its largest drop.
    """
    gamma = rho * delta
    order = np.argsort(-gamma, kind="stable")

    if n_clusters is not None:
        if not 1 <= n_clusters <= rho.size:
            raise ValueError(
                f"n_clusters must be in [1, {rho.size}], got {n_clusters}"
            )
        return order[:n_clusters]

    ranked = gamma[order]
    gaps = ranked[:-1] - ranked[1:]
    k = int(np.argmax(gaps)) + 1

    logging.debug("Ranked gamma: {}".format(ranked))
    logging.debug("Largest gamma gap after rank {}, selecting {} centers".format(k, k))

    return order[:k]


def gamma_gap_score(
    gamma: np.ndarray,
    max_clusters: int = 10,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Score the candidate number of clusters as,

        score(K) = log(gamma_K / gamma_{K+1}) / median gap

    Returns (ks, scores) -- the candidate cluster counts and the score at each.
    """
    if max_clusters < 2:
        raise ValueError(f"max_clusters must be at least 2, got {max_clusters}")

    ranked = np.sort(np.asarray(gamma, dtype=float))[::-1][:max_clusters + 1]
    ranked = ranked[ranked > 0.0]

    if ranked.size < 3:
        return np.empty(0, dtype=int), np.empty(0)

    gaps = np.log(ranked[:-1]) - np.log(ranked[1:])
    positive = gaps[gaps > 0.0]

    if positive.size == 0:
        return np.empty(0, dtype=int), np.empty(0)

    ks = np.arange(2, gaps.size + 1)
    scores = gaps[1:] / np.median(positive)

    return ks, scores


def num_cluster_candidates(
    gamma: np.ndarray,
    max_clusters: int = 10,
) -> np.ndarray:
    """
    Candidate cluster counts, ranked by `gamma_gap_score`.
    """
    ks, scores = gamma_gap_score(gamma, max_clusters=max_clusters)

    if ks.size == 0:
        logging.debug("No gamma gap to rank below K={}".format(max_clusters))
        return np.empty(0, dtype=int)

    strongest_first = np.argsort(-scores, kind="stable")

    logging.debug(
        "Cluster count candidates {} with gap scores {}".format(
            ks[strongest_first].tolist(),
            np.round(scores[strongest_first], 4).tolist(),
        )
    )

    return ks[strongest_first]


def assign_labels(
    rho: np.ndarray,
    nneigh: np.ndarray,
    centers: np.ndarray,
) -> np.ndarray:
    """
    Iterating over points in order of decreasing density; each
    non-center takes the label of its nearest neighbour of higher density.
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
    is the highest density in that border region, and points below it are halo (~ noise).
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
    min_k: int = 5,
    sigma: float | None = None,
    n_clusters: int | None = None,
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

    rho = local_density(D, d_c, kernel=kernel, min_k=min_k, sigma=sigma)
    delta, nneigh = min_higher_density_distance(D, rho)
    centers = select_centers(rho, delta, n_clusters=n_clusters)
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
    min_k: int = 5,
    sigma: float | None = None,
    n_clusters: int | None = None,
    assign_halo: bool = False,
) -> dict[int, int]:
    """
    Cluster a SAGD distance matrix and return {snapshot index -> cluster label}.

    Keys are row indices into `distances`; row i corresponds to the i-th snapshot time. 
    Labels run 0..k-1 ordered by descending gamma, or -1 for halo points if `assign_halo`.
    """
    result = density_peaks(
        distances,
        d_c=d_c,
        neighbor_frac=neighbor_frac,
        kernel=kernel,
        min_k=min_k,
        sigma=sigma,
        n_clusters=n_clusters,
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


def _symlog_threshold(values: np.ndarray) -> float:
    """
    Where a symlog axis over `values` should switch from linear to logarithmic
    """
    positive = values[values > 0.0]
    return float(positive.min()) if positive.size else 1.0


def plot_decision_graph(
    result: DensityPeakResult,
    ts: np.ndarray | None = None,
    ax: Axes | None = None,
    annotate: bool = True,
    to_annotate: int | None = None,
    save_path: Path | None = None,
) -> tuple[Figure, Axes]:
    """
    The decision graph of the paper (Fig. 1B): delta against rho. Cluster centers are
    the points that stand out towards the top right -- high density and anomalously far
    from any denser point. Isolated outliers sit top left (high delta, low rho).

    Pass `ts` (the snapshot times, row i of the distance matrix <-> ts[i]) to annotate
    the centers with their diffusion time instead of their snapshot index. `to_annotate`
    labels that many top points by gamma = rho * delta -- the paper's own ranking, so it
    shows which points would become centers at a larger `n_clusters`; left unset, only
    the centers are labelled.

    Returns (fig, ax); writes a PNG only if `save_path` is given.
    """
    own_figure = ax is None
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
        if to_annotate is None:
            labelled = result.centers
        else:
            labelled = np.argsort(-result.gamma, kind="stable")[:to_annotate]

        for i in labelled:
            text = f"t={ts[i]:.2f}" if ts is not None else f"{i}"
            ax.annotate(
                text,
                (float(result.rho[i]), float(result.delta[i])),
                textcoords="offset points", xytext=(8, 4),
                fontsize=9, fontweight="bold",
            )

    ax.set_xscale("symlog", linthresh=_symlog_threshold(result.rho))
    ax.set_yscale("symlog", linthresh=_symlog_threshold(result.delta))
    ax.set_xlabel(r"$\rho$  (local density)", fontsize=12)
    ax.set_ylabel(r"$\delta$  (distance to closest denser point)", fontsize=12)
    ax.set_title(
        r"Decision graph  ({} clusters)".format(result.centers.size),
        fontsize=12,
    )
    ax.legend(frameon=False, fontsize=9)

    if own_figure:
        fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")

    return fig, ax


def plot_gamma_gap(
    result: DensityPeakResult,
    max_clusters: int = 10,
    to_annotate: int = 5,
    ax: Axes | None = None,
    save_path: Path | None = None,
) -> tuple[Figure, Axes]:
    own_figure = ax is None
    fig, ax = _figure_and_axes(ax, (5, 5))

    ks, scores = gamma_gap_score(result.gamma, max_clusters=max_clusters)
    candidates = num_cluster_candidates(result.gamma, max_clusters=max_clusters)

    ax.set_xlabel("number of clusters $k$", fontsize=12)
    ax.set_ylabel(r"gap$(k)$  /  median gap", fontsize=12)
    ax.set_title(r"$\gamma$ gap", fontsize=12)

    if ks.size == 0:
        ax.text(0.5, 0.5, "no gap to score",
                transform=ax.transAxes, ha="center", va="center", fontsize=11)
        ax.set_xticks([])
        ax.set_yticks([])
        if own_figure:
            fig.tight_layout()
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches="tight")
        return fig, ax

    labelled = candidates[:to_annotate]
    is_labelled = np.isin(ks, labelled)

    ax.bar(ks[~is_labelled], scores[~is_labelled], color="0.7", zorder=2)
    ax.bar(ks[is_labelled], scores[is_labelled], color="crimson", zorder=2)
    # ax.axhline(1.0, color="0.4", linestyle="--", linewidth=1.0, zorder=1,
    #            label="typical gap")

    for k in labelled:
        ax.annotate(
            f"K={k}",
            (float(k), float(scores[ks == k][0])),
            textcoords="offset points", xytext=(0, 4), ha="center",
            fontsize=10 if k == candidates[0] else 9,
            fontweight="bold" if k == candidates[0] else "normal",
            color="crimson",
        )

    ax.set_yscale("symlog", linthresh=_symlog_threshold(scores))
    ax.set_xticks(ks)
    ax.legend(frameon=False, fontsize=9)

    if own_figure:
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
    own_figure = ax is None
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

    if own_figure:
        fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")

    return fig, ax
