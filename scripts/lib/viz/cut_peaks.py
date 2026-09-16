"""Between-cluster graph cut vs. normalized-CTD mode separation, over diffusion time.

Extracted from `speciation_time_nsweep.ipynb` so any notebook can plot the pair for a
completed run; the run (`cell`, `run_dir`) is passed in explicitly rather than looked up
in notebook globals.
"""
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from scipy.ndimage import gaussian_filter1d

from lib.ou_model import theoretical_bimodal_gaussian_ts
from lib.viz.dist_animation import get_projection_and_cut, mixture_plane_basis


def ctd_peak_separation(values, bins=120, smooth=2.0, prominence=0.02, min_ctd=0.05):
    """Distance between the two highest peaks of one snapshot's *normalized* CTD
    distribution, or 0.0 when fewer than two peaks are found.

    `stats.normalize` already puts the values on [0, 1], so the histogram is taken on a
    fixed [0, 1] grid and separations are directly comparable across snapshots. Peaks are
    ranked by height rather than prominence -- the two tallest modes are the ones that
    stand out in the distribution panel. The smoothed histogram is zero-padded at both
    ends so that a mode sitting hard against 0 or 1 is still found, `find_peaks` being
    blind to edge maxima.

    Peaks below `min_ctd` are discarded: percentile clipping in `stats.normalize` piles
    the bottom 5% of CTDs onto a single value, giving a spike at ~0 that is an artefact of
    the clipping rather than a mode of the distribution, and whose near-constant height
    otherwise wins the second slot whenever the real interior mode dips below it.
    """
    counts, edges = np.histogram(np.asarray(values), bins=bins, range=(0.0, 1.0))
    centers = (edges[:-1] + edges[1:]) / 2
    smoothed = gaussian_filter1d(counts.astype(float), smooth)
    idx, _ = find_peaks(np.r_[0.0, smoothed, 0.0],
                        prominence=smoothed.max() * prominence)
    idx = idx - 1
    idx = idx[centers[idx] >= min_ctd]
    if idx.size < 2:
        return 0.0
    top = idx[np.argsort(smoothed[idx])[::-1][:2]]
    return float(abs(centers[top[0]] - centers[top[1]]))


def plot_cut_and_ctd_peaks(cell, run_dir, d, n, std=1.0, label=None, t_range=None,
                           ax=None, save_fig_path=None, **peak_kwargs):
    """Between-cluster cut weight fraction and CTD mode separation over time.

    `cell` is a loaded run entry -- keys `ctds`, `tsagd`, `tstar`, `mu` -- and `run_dir`
    its output directory; `d`, `n` and the optional `label` (e.g. the graph-construction
    mode) only feed the title, so runs from different experiments plot through the same
    call.

    Left axis: the weight of the kNN edges joining the two clusters as a fraction of the
    graph's total weight (`get_projection_and_cut`). Right axis: the gap between the two
    highest peaks of the normalized CTD distribution (`ctd_peak_separation`). Both ask
    when the graph stops mixing the two clusters, so the point of the plot is whether they
    turn over together.

    `t_range=(t_lo, t_hi)` restricts the plot to snapshots inside that window. Pass `ax`
    to draw into an existing axis (its figure is returned); `save_fig_path` writes the
    figure to disk.
    """
    ctds = cell["ctds"]["CTDs"]
    mu_star = np.full(d, cell["mu"])
    proj = get_projection_and_cut(Path(run_dir), mixture_plane_basis(mu_star))

    snaps = np.array([t for t in ctds
                      if t_range is None or t_range[0] <= t <= t_range[1]])
    if snaps.size == 0:
        raise ValueError(
            f"No snapshots inside t_range={t_range}; available times span "
            f"[{min(ctds):.3g}, {max(ctds):.3g}]"
        )
    ratio = np.array([proj[t]["cut"] / proj[t]["weight"] for t in snaps])
    separation = np.array([ctd_peak_separation(ctds[t]["norm_ctds"], **peak_kwargs)
                           for t in snaps])
    t_s, _ = theoretical_bimodal_gaussian_ts(mu_star, std, snaps)

    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 4.5))
    else:
        fig = ax.figure
    ax.plot(snaps, ratio, color="crimson", lw=1.8)
    ax.set_xlabel("diffusion time  t  (T → 0)")
    ax.set_ylabel("cut weight / total weight", color="crimson")
    ax.tick_params(axis="y", labelcolor="crimson")
    ax.set_ylim(0, max(ratio.max() * 1.08, 1e-9))

    ax2 = ax.twinx()
    ax2.plot(snaps, separation, color="steelblue", lw=1.8)
    ax2.set_ylabel("separation of the two highest normalized-CTD peaks",
                   color="steelblue")
    ax2.tick_params(axis="y", labelcolor="steelblue")
    ax2.set_ylim(0, max(separation.max() * 1.08, 1e-9))

    ax.set_xlim(snaps.min(), snaps.max())
    for t_mark, color, ls, mark_label in (
            (cell["tsagd"], "darkcyan", "-", fr"$t_{{sagd}}$={cell['tsagd']:.2f}"),
            (cell["tstar"], "darkorange", "--", fr"$t^*$={cell['tstar']:.2f}"),
            (t_s, "0.35", ":", fr"$t_s$={t_s:.2f}")):
        if snaps.min() <= t_mark <= snaps.max():
            ax.axvline(t_mark, color=color, ls=ls, lw=1.5, label=mark_label)
    if ax.get_legend_handles_labels()[0]:
        ax.legend(loc="lower right", fontsize=9, frameon=False)
    prefix = f"{label}   |   " if label else ""
    ax.set_title(f"{prefix}D={d}, N={n}: KNN graph cut  vs  CTD peak distance",
                 fontsize=12)
    fig.tight_layout()

    if save_fig_path:
        Path(save_fig_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_fig_path, dpi=150, bbox_inches="tight")
    return fig
