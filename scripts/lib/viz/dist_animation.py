"""Per-snapshot distance-distribution animations for a completed SAGD run.

Extracted from `notebooks/speciation_time_nsweep.ipynb` so several notebooks can
animate the same run outputs. Everything here works off one run directory
(`<save_path>/<exp_name>/D{d}_N{n}_T{T}/`) plus that run's entry in the
notebook-side `grid_data` dict, so runs from different graph-construction modes
(random edge injection vs. MST fallback) can be animated side by side.
"""

from pathlib import Path

import joblib
import numpy as np
import matplotlib.animation as animation
import matplotlib.pyplot as plt
from IPython.display import HTML
from joblib import Parallel, delayed
from matplotlib.collections import LineCollection
from scipy.spatial.distance import pdist

from lib.ou_model import mixture_at_time
from lib.stats import compress_equal_frequency

EUCLID_FILE = "euclid_dists.jbl"
PROJ_FILE = "proj2d.jbl"
PROJ_VERSION = 2

# (title, key, x-axis color, log-x?) for the two distribution panels sitting to
# the right of the mixture-contour panel.
_PANELS = [
    ("Normalized CTD",     "norm_ctds", "seagreen",  False),
    ("Euclidean distance", "euclid",    "indianred", False),
]


def _snapshot_euclid_quantiles(points, n_bins):
    """All pairwise Euclidean distances at one snapshot, compressed to `n_bins`
    equal-frequency quantiles (same scheme the pipeline uses for CTDs)."""
    return compress_equal_frequency(pdist(np.asarray(points), metric="euclidean"), n_bins)


def get_euclidean_quantiles(run_dir: Path, n_bins: int = 5000, n_jobs: int = 16) -> dict:
    """Per-snapshot pairwise-Euclidean-distance distributions for one run.

    Cached as `euclid_dists.jbl` next to the other pipeline outputs, so the
    large `history.jbl` is deserialized only the first time. Returns a dict
    mapping snapshot time -> compressed quantile array, ordered T -> 0.
    """
    cache = Path(run_dir) / EUCLID_FILE
    if cache.exists():
        return joblib.load(cache)["euclid"]

    hist = joblib.load(Path(run_dir) / "history.jbl")["history"]
    ts = list(hist.keys())
    dists = Parallel(n_jobs=n_jobs)(
        delayed(_snapshot_euclid_quantiles)(hist[t], n_bins) for t in ts
    )
    euclid = dict(zip(ts, dists))
    joblib.dump(
        {"euclid": euclid, "params": {"ts": ts, "n_bins": n_bins}},
        cache,
        compress=3,
    )
    return euclid


def get_projection_and_cut(run_dir: Path, basis: np.ndarray) -> dict:
    """Per-snapshot 2-D projection of the samples plus the kNN between-cluster cut.

    A point's cluster is the centre it is closer to, which for the bimodal
    mixture -- equal weights, one shared isotropic variance -- is exactly the
    sign of its coordinate along the mean-difference direction, i.e. of the
    first projected coordinate (the same rule as `ou_model.classify`).

    Returns a dict mapping snapshot time to `xy` (N, 2) projected samples,
    `labels` (N,) in {-1, +1}, `edges` (2, E) / `edge_w` (E,) for the graph
    edges joining the two clusters, `cut`, their total weight, and
    `n_edges` / `weight` for the whole graph, so the cut can be reported as a
    fraction of either.

    Cached as `proj2d.jbl` next to the other pipeline outputs, since building it
    reads both `history.jbl` and the large `Ws.jbl` once.
    """
    cache = Path(run_dir) / PROJ_FILE
    if cache.exists():
        stored = joblib.load(cache)
        if (stored["params"].get("version") == PROJ_VERSION
                and np.allclose(stored["params"]["basis"], basis)):
            return stored["proj"]

    hist = joblib.load(Path(run_dir) / "history.jbl")["history"]
    ts = list(hist.keys())
    xys = [np.asarray(hist[t], dtype=float) @ basis for t in ts]
    del hist

    graphs = joblib.load(Path(run_dir) / "Ws.jbl")
    assert np.allclose(graphs["ts"], ts), "graph / history snapshot times differ"

    proj = {}
    for t, xy, W in zip(ts, xys, graphs["Ws"]):
        labels = np.where(xy[:, 0] >= 0, 1, -1).astype(np.int8)
        i, j = np.triu(W, 1).nonzero()
        w = W[i, j]
        between = labels[i] != labels[j]
        proj[t] = {
            "xy": xy.astype(np.float32),
            "labels": labels,
            "edges": np.stack([i[between], j[between]]).astype(np.int32),
            "edge_w": w[between].astype(np.float32),
            "cut": float(w[between].sum()),
            "n_edges": int(i.size),
            "weight": float(w.sum()),
        }
    del graphs

    joblib.dump(
        {"proj": proj, "params": {"ts": ts, "basis": basis, "version": PROJ_VERSION}},
        cache,
        compress=3,
    )
    return proj


def mixture_plane_basis(mu_star, seed=0):
    """(d, 2) orthonormal basis of the plane the bimodal mixture is drawn in.

    Column 1 is the unit vector along the mean difference mu_+ - mu_-, the only
    direction carrying cluster information; column 2 is an arbitrary orthogonal
    direction, along which both components have identical marginals.

    An orthogonal projection maps N(mu, Delta*I_d) onto N(B.T @ mu, Delta*I_2)
    exactly, so -- the mean difference lying inside the plane -- the drawn 2-D
    mixture has the same component overlap mass as the d-dimensional one, not
    an approximation of it. The basis is time-independent: `mixture_at_time`
    only rescales the means by exp(-t), it never rotates them.
    """
    mu = np.asarray(mu_star, dtype=float)
    u = mu / np.linalg.norm(mu)
    rng = np.random.default_rng(seed)
    v = rng.standard_normal(u.size)
    v -= (v @ u) * u
    v /= np.linalg.norm(v)
    return np.stack([u, v], axis=1)


def _draw_mixture_panel(ax, t, mu_star, std, basis, snap,
                        radii=(1.0, 2.0, 3.0), grid=401, max_edges=20):
    """Bimodal mixture at time t drawn in the mean-difference plane, in units of
    the (shared) component standard deviation sigma(t).

    Contour lines of the exact 2-D marginal sit `radii` standard deviations out
    from a component centre, so each blob keeps a constant apparent width while
    the centres travel inward as exp(-t); the window is rescaled per frame to
    +-(delta/2 + 4), like the histogram panels. On top go the projected samples,
    coloured by cluster, and the kNN edges that join the two clusters (at most
    `max_edges` of them are drawn, evenly spaced through the list, but the
    reported cut counts and weights are always the full sums, each shown against
    the whole graph's total). `snap` is one time slice of
    `get_projection_and_cut`.
    """
    means, variances, w = mixture_at_time(t, mu_star, std)
    assert np.allclose(variances, variances[0]), \
        "panel assumes one shared isotropic variance across components"
    sigma = float(np.sqrt(variances[0]))
    c = (means @ basis) / sigma
    lim = np.abs(c).max() + 4.0

    g = np.linspace(-lim, lim, grid)
    Z1, Z2 = np.meshgrid(g, g)
    dens = np.zeros_like(Z1)
    for (c1, c2), wk in zip(c, w):
        dens += wk * np.exp(-0.5 * ((Z1 - c1)**2 + (Z2 - c2)**2)) / (2 * np.pi)

    peak = w.max() / (2 * np.pi)
    levels = [peak * np.exp(-0.5 * r**2) for r in sorted(radii, reverse=True)]
    top = max(dens.max(), levels[-1]) * 1.01
    ax.contourf(Z1, Z2, dens, levels=levels + [top], cmap="Blues", alpha=0.55)
    ax.contour(Z1, Z2, dens, levels=levels, colors="steelblue", linewidths=1.0)

    z = snap["xy"] / sigma
    pos = snap["labels"] > 0
    ax.scatter(z[pos, 0], z[pos, 1], s=2, c="0.25", alpha=0.30, linewidths=0)
    ax.scatter(z[~pos, 0], z[~pos, 1], s=2, c="0.55", alpha=0.30, linewidths=0)

    i, j = snap["edges"]
    if i.size:
        step = max(1, int(np.ceil(i.size / max_edges)))
        segs = np.stack([z[i[::step]], z[j[::step]]], axis=1)
        alpha = float(np.clip(400 / len(segs), 0.06, 0.6))
        ax.add_collection(LineCollection(segs, colors="crimson", linewidths=0.6,
                                         alpha=alpha, zorder=3))
    ax.plot(c[:, 0], c[:, 1], "kx", ms=6, mew=1.5, zorder=4)

    delta = float(np.linalg.norm(c[0] - c[1]))
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_aspect("equal")
    ax.set_title("Gaussian mixture (mean-difference plane)", fontsize=11)
    ax.set_xlabel(r"along $(\mu_+-\mu_-)/\|\cdot\|$")
    ax.set_ylabel(r"orthogonal direction")
    ax.text(0.02, 0.98,
            f"$\\delta$ = {delta:.3g} $\\sigma$\n"
            f"cut edges  = {i.size:d} / {snap['n_edges']:d}"
            f"  ({i.size / snap['n_edges']:.1%})\n"
            f"cut weight = {snap['cut']:.4g} / {snap['weight']:.5g}"
            f"  ({snap['cut'] / snap['weight']:.1%})",
            transform=ax.transAxes, va="top", ha="left", fontsize=8)


def _panel_edges(dist, bins, logx, coverage=0.99, xlim=None):
    """Histogram bin edges for a *single* snapshot's distribution, spanning the
    central `coverage` fraction of its values (so ~99% of the observations fall
    inside the axis at that time step). `xlim` pins the range explicitly
    instead of deriving it from the data."""
    if xlim is not None:
        lo, hi = xlim
    else:
        tail = (1.0 - coverage) / 2
        lo, hi = np.quantile(dist, [tail, 1.0 - tail])
    if hi <= lo:
        hi = lo + (abs(lo) * 1e-6 or 1e-12)
    if logx:
        lo = max(lo, 1e-12)
        hi = max(hi, lo * (1 + 1e-9))
        return np.logspace(np.log10(lo), np.log10(hi), bins + 1)
    return np.linspace(lo, hi, bins + 1)


def animate_distributions(cell, run_dir, d, n, std=1.0, label=None, stride=2,
                          bins=200, coverage=0.99, t_range=(1.0, 3.0),
                          euclid_xlim=None, n_jobs=16, fps=6, save_path=None):
    """Animated Gaussian-mixture contours / normalized-CTD / Euclidean-distance
    panels for one completed run, one frame per snapshot time.

    `cell` is that run's loaded entry -- keys `ctds`, `tsagd`, `tstar`, `mu` --
    and `run_dir` its output directory; `d`, `n` and the optional `label` (e.g.
    the graph-construction mode) are used for the figure title only, so runs
    from different experiments can be animated with the same call.

    The left panel is the analytic bimodal mixture at that time
    (`ou_model.mixture_at_time`), projected onto the plane spanned by the mean
    difference and one orthogonal direction (see `mixture_plane_basis`) -- an
    exact 2-D marginal, so its component overlap is the d-dimensional one --
    overlaid with the run's own samples in the same projection and the kNN-graph
    edges that join the two clusters. The two right panels are the empirical
    distance distributions of the run.

    Returns an ``IPython.display.HTML`` player (frame slider + play/step
    controls) so the time step can be scrubbed manually or played through.
    `stride` sub-samples snapshots to keep the embedded animation small.

    Each panel is rescaled *per time step*: the x-range covers the central
    `coverage` fraction (default 99%) of that snapshot's values, and the y-range
    is set from that frame's own histogram -- so a single high-count snapshot no
    longer flattens the rest of the animation. The distributions shift by orders
    of magnitude along the trajectory, hence the per-frame limits; watch the
    tick labels, not the bar positions, when comparing frames.
    `euclid_xlim=(lo, hi)` pins the Euclidean panel to a fixed window instead
    (the useful window scales with `d`, since pairwise distances in d dimensions
    concentrate around sqrt(2*d)).

    `t_range=(t_lo, t_hi)` truncates the animation to snapshots with
    `t_lo <= t <= t_hi` (default (1.0, 3.0), the window around the transition);
    pass `t_range=None` to animate the whole trajectory. The per-frame axis
    limits are computed from the retained snapshots only.

    If `save_path` is given, the animation is also written to disk (parent dirs
    created as needed). The format follows the extension:
      * `.gif`          -> rendered frames via the pillow writer;
      * `.html`/`.htm`  -> the self-contained scrubbable jshtml player.
    (`.mp4` needs ffmpeg, which isn't installed here -- use `.gif`.)
    """
    ctds = cell["ctds"]["CTDs"]
    euclid = get_euclidean_quantiles(run_dir, n_jobs=n_jobs)
    snaps = list(ctds.keys())
    assert list(euclid.keys()) == snaps, "CTD / Euclidean snapshot times differ"
    if t_range is not None:
        t_lo, t_hi = t_range
        snaps = [t for t in snaps if t_lo <= t <= t_hi]
        if not snaps:
            raise ValueError(
                f"No snapshots inside t_range={t_range}; available times span "
                f"[{min(ctds):.3g}, {max(ctds):.3g}]"
            )
    tsagd, tstar = cell["tsagd"], cell["tstar"]

    sources = {"norm_ctds": lambda t: np.asarray(ctds[t]["norm_ctds"]),
               "euclid": lambda t: np.asarray(euclid[t])}
    mu_star = np.full(d, cell["mu"])
    basis = mixture_plane_basis(mu_star)
    proj = get_projection_and_cut(run_dir, basis)
    panel_xlim = {"euclid": euclid_xlim}
    panel_data, panel_edges, panel_ymax = [], [], []
    for _title, key, _c, logx in _PANELS:
        dists = [sources[key](t) for t in snaps]
        edges = [_panel_edges(a, bins, logx, coverage, panel_xlim.get(key))
                 for a in dists]
        ymax = [np.histogram(a, bins=e)[0].max() for a, e in zip(dists, edges)]
        if max(ymax) == 0:
            print(f"warning: no {key} values fall inside {panel_xlim.get(key)} -- "
                  f"panel will be empty (try a wider euclid_xlim for d={d})")
        panel_data.append(dists)
        panel_edges.append(edges)
        panel_ymax.append([max(m, 1) * 1.08 for m in ymax])

    frames = list(range(0, len(snaps), stride))
    if frames[-1] != len(snaps) - 1:
        frames.append(len(snaps) - 1)

    fig = plt.figure(figsize=(16, 5))
    gs = fig.add_gridspec(2, 3, height_ratios=[1, 9], hspace=0.55, wspace=0.28)
    ax_time = fig.add_subplot(gs[0, :])
    axes = [fig.add_subplot(gs[1, c]) for c in range(3)]

    T_max, T_min = max(snaps), min(snaps)
    pad = 0.02 * (T_max - T_min) or 1e-3
    ax_time.set_xlim(max(0.0, T_min - pad), T_max + pad)
    ax_time.set_ylim(0, 1)
    ax_time.set_yticks([])
    ax_time.set_xlabel("diffusion time  t  (T → 0)")
    ax_time.axvline(tsagd, color="darkcyan", lw=2, label=fr"$t_{{sagd}}$={tsagd:.2f}")
    ax_time.axvline(tstar, color="darkorange", lw=2, ls="--", label=fr"$t^*$={tstar:.2f}")
    ax_time.legend(loc="center left", bbox_to_anchor=(1.005, 0.5),
                   fontsize=8, frameon=False)
    time_marker = ax_time.axvline(T_max, color="red", lw=2.5)

    prefix = f"{label}   |   " if label else ""

    def draw(frame_idx):
        t = snaps[frame_idx]
        time_marker.set_xdata([t, t])
        axes[0].clear()
        _draw_mixture_panel(axes[0], t, mu_star, std, basis, proj[t])
        for ax, dists, edges, ymax, (title, _key, color, logx) in zip(
                axes[1:], panel_data, panel_edges, panel_ymax, _PANELS):
            e = edges[frame_idx]
            ax.clear()
            ax.hist(dists[frame_idx], bins=e, color=color, alpha=0.85)
            if logx:
                ax.set_xscale("log")
            ax.set_xlim(e[0], e[-1])
            ax.set_ylim(0, ymax[frame_idx])
            ax.set_title(title, fontsize=11)
            ax.set_xlabel("distance")
            ax.set_ylabel("count (quantile bins)")
        regime = ("noise" if t > tsagd else
                  "SAGD onset" if t > tstar else "resolved clusters")
        fig.suptitle(f"{prefix}D={d}, N={n}   |   t = {t:.2f}   |   regime: {regime}",
                     fontsize=13, y=1.0)

    anim = animation.FuncAnimation(
        fig, lambda k: draw(frames[k]), frames=len(frames), interval=1000 / fps
    )

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        suffix = save_path.suffix.lower()
        if suffix == ".gif":
            anim.save(save_path, writer="pillow", fps=fps)
        elif suffix in (".html", ".htm"):
            save_path.write_text(anim.to_jshtml(fps=fps))
        else:
            plt.close(fig)
            raise ValueError(
                f"Unsupported save extension '{suffix}'. Use '.gif' or '.html' "
                "('.mp4' requires ffmpeg, which isn't installed)."
            )
        print(f"Saved animation -> {save_path}")

    html = HTML(anim.to_jshtml(fps=fps))
    plt.close(fig)
    return html
