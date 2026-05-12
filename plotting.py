# Gemini wrote this for me.
import numpy as np
from matplotlib import cm
import torch
import seaborn as sns
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from mpl_toolkits.axes_grid1 import make_axes_locatable
# from SASNE.RRP import RRP
from matplotlib import pyplot as plt


def plot_data_distribution(
        step_idx,
        title,
        ax,
        path_history,
        times
):
    t = times[step_idx]
    d1_samples = path_history[step_idx, :, 0] 
    d2_samples = path_history[step_idx, :, 1] 
    
    # Calculate analytical density for the heatmap
    limit = 6
    res = 100
    _x = np.linspace(-limit, limit, res)
    _y = np.linspace(-limit, limit, res)
    xx, yy = np.meshgrid(_x, _y)
    
    # OU parameters at time t
    gamma_t = (1 - np.exp(-2*t)) + (std**2 * np.exp(-2*t))
    mu_t = mu_star.numpy() * np.exp(-t)
    
    # GMM Density: p_t(x) = 0.5 * [N(mu_t, gamma) + N(-mu_t, gamma)]
    d1 = (xx - mu_t[0])**2 + (yy - mu_t[1])**2
    d2 = (xx + mu_t[0])**2 + (yy + mu_t[1])**2
    density = (np.exp(-0.5 * d1 / gamma_t) + np.exp(-0.5 * d2 / gamma_t)) / (2 * np.pi * gamma_t)

    ax.contourf(xx, yy, density, levels=20, cmap="Reds", alpha=0.5)
    ax.scatter(d1_samples, d2_samples, s=80, color='black', marker='X', label='Data')
        
    ax.set_title(title, fontsize=14)
    ax.set_xlim(-limit, limit)
    ax.set_ylim(-limit, limit)
    ax.set_aspect('equal')
    ax.grid(alpha=0.1)


def plot_speciation_3d(
    path_history,
    times,
    mu_star,
    std,
    nsamples,
    t_s=None,
    save_fig_path=None,
    seed=123
):
    """
    Visualizes the 2D speciation process over time in a 3D plot with 
    transparent frames/panes at critical milestones.
    """
    torch.manual_seed(seed)
    t_max = times[0]
    plane_limit = 10 
    res = 50 
    
    fig = plt.figure(figsize=(15, 20))
    ax = fig.add_subplot(111, projection='3d')

    ax.set_ylabel("Diffusion Time (t)", fontsize=14, labelpad=25)
    ax.set_xlabel("$x_1$", fontsize=10)
    ax.set_zlabel("$x_2$", fontsize=10)
    
    ax.set_ylim(0, t_max)
    ax.set_xlim(-plane_limit, plane_limit)
    ax.set_zlim(-plane_limit, plane_limit)
    
    ax.invert_yaxis() # Time flows from t_max down to 0
    ax.set_box_aspect((1, 3, 1)) 
    ax.view_init(elev=20, azim=-120)

    _grid = np.linspace(-plane_limit, plane_limit, res) 
    xx, zz = np.meshgrid(_grid, _grid)
    
    def draw_milestone_pane(t_val, color, alpha_surf=0.05, label=None):
        # 1. Draw the actual Frame (the edges)
        ax.plot([-plane_limit, plane_limit, plane_limit, -plane_limit, -plane_limit],
                [t_val]*5,
                [-plane_limit, -plane_limit, plane_limit, plane_limit, -plane_limit], 
                color=color, lw=1.5, alpha=0.7, zorder=2)
        
        # 2. Draw the transparent surface (the 'glass')
        verts = [[(-plane_limit, t_val, -plane_limit), (plane_limit, t_val, -plane_limit), 
                  (plane_limit, t_val, plane_limit), (-plane_limit, t_val, plane_limit)]]
        pane = Poly3DCollection(verts, alpha=alpha_surf, facecolor=color, zorder=1)
        ax.add_collection3d(pane)
        
        if label:
            ax.text(-plane_limit - 2, t_val, -plane_limit, f" {label}", 
                    color=color, fontweight='bold', fontsize=12)

    # --- Start Layer (t = T) ---
    density_T = np.exp(-0.5 * (xx**2 + zz**2) / (1 + std**2))
    ax.contourf(xx, density_T, zz, zdir='y', offset=t_max, cmap='Blues', alpha=0.2)
    draw_milestone_pane(t_max, 'black', alpha_surf=0.02)

    # --- Speciation Layer (t = t_s) ---
    if t_s is not None:
        # Greenish density to show the 'stretched' Gaussian
        sig_ts = np.sqrt(np.exp(-t_s) + std**2) 
        density_ts = np.exp(-0.5 * (xx**2 + zz**2) / sig_ts**2)
        ax.contourf(xx, density_ts, zz, zdir='y', offset=t_s, cmap='Greens', alpha=0.15)
        draw_milestone_pane(t_s, 'darkcyan', alpha_surf=0.1, label="$t_s$")

    # --- Plotting Path Trajectories ---
    for i in range(nsamples):
        x1_path = path_history[:, i, 0]
        x2_path = path_history[:, i, 1]
        
        final_pos = torch.tensor(path_history[-1, i, :])
        class_color = 'gold' if final_pos[0] > 0 else 'royalblue'
        ax.plot(x2_path, times, x1_path, color=class_color, lw=1.5, alpha=0.6, zorder=10)

    # --- End Layer (t = 0) ---
    dist1 = (zz - mu_star[0].item())**2 + (xx - mu_star[1].item())**2
    dist2 = (zz + mu_star[0].item())**2 + (xx + mu_star[1].item())**2
    density_0 = np.exp(-0.5 * dist1 / std**2) + np.exp(-0.5 * dist2 / std**2)
    
    ax.contourf(xx, density_0, zz, zdir='y', offset=0, cmap='Reds', alpha=0.3)
    draw_milestone_pane(0, 'black', alpha_surf=0.05)

    # Initial and Final Scatter Points
    x1_start, x2_start = path_history[0, :, 0], path_history[0, :, 1]
    ax.scatter(x2_start, np.full_like(x2_start, t_max), x1_start,
               s=50, color='grey', alpha=0.4, depthshade=False)

    x1_final, x2_final = path_history[-1, :, 0], path_history[-1, :, 1]
    ax.scatter(x2_final, np.zeros_like(x2_final), x1_final, 
               s=100, color='black', marker='X', edgecolors='white', zorder=15)

    ax.set_xticks([])
    ax.set_zticks([])
    
    plt.tight_layout()
    if save_fig_path:
        plt.savefig(save_fig_path, dpi=300, bbox_inches='tight')
    plt.show()


def plot_sagd_heatmap_row(
        W_list,
        d_list,
        time_snaps_vector_list,
        ts_tuple_list,
        save_fig_path=None
):
    num_plots = len(W_list)
    max_cols = 6
    num_rows = (num_plots + max_cols - 1) // max_cols
    num_cols = min(num_plots, max_cols)

    fig, axes = plt.subplots(num_rows, num_cols, figsize=(5 * num_cols, 5 * num_rows))
    axes_flat = axes.flatten() if num_plots > 1 else [axes]

    for i, (W, d, ts_tuple, time_snaps_vector) in enumerate(zip(W_list, d_list, ts_tuple_list, time_snaps_vector_list)):
        ax_heat = axes_flat[i]
        ts, ts_idx = ts_tuple

        ## 1. Normalize current matrix to 0-1
        # W_min, W_max = W.min(), W.max()
        ## Handle cases where min might equal max to avoid division by zero
        # W_norm = (W - W_min) / (W_max - W_min) if W_max > W_min else W - W_min
        W_norm = np.clip(W / np.percentile(W, 95), 0, 1)

        n_samples = len(time_snaps_vector)
        heat_indices = np.linspace(0, n_samples - 1, 8, dtype=int)
        tick_labels = [f"{time_snaps_vector[idx]:.2f}" for idx in heat_indices]

        # 2. Plot with a fixed 0-1 colorbar for consistent relative comparison

        sns.heatmap(W_norm, cmap='viridis', ax=ax_heat, vmin=0, vmax=1, cbar_kws={'shrink': 0.8})
        # Heatmap Ticks
        ax_heat.set_xticks(heat_indices + 0.5)
        ax_heat.set_xticklabels(tick_labels, rotation=45)
        ax_heat.set_yticks(heat_indices + 0.5)
        ax_heat.set_yticklabels(tick_labels, rotation=0)
        ax_heat.set_xlabel("Time")

        # Red Speciation Lines
        ax_heat.axvline(x=ts_idx + 0.5, color='red', linestyle='--', alpha=0.8, label=f'$t_s = {ts:.2f}$')
        ax_heat.axhline(y=ts_idx + 0.5, color='red', linestyle='--', alpha=0.8)

        ax_heat.set_title(f"d={d}", fontsize=16)
        if i % max_cols == 0:
            ax_heat.set_ylabel("Time", fontsize=12)
        ax_heat.legend(loc='upper left')

    # Delete unused axes
    for j in range(i + 1, len(axes_flat)):
        fig.delaxes(axes_flat[j])

    plt.tight_layout()
    if save_fig_path:
        plt.savefig(save_fig_path, dpi=300)
    plt.show()


def plot_sagd_heatmap(
        W,
        time_vector,
        ts,
        ts_idx,
        d,
        save_fig_path=None
):
    plt.figure(figsize=(10, 8))
    ax = sns.heatmap(W, cmap='viridis', robust=True)
    n_samples = len(time_vector)
    indices = np.linspace(0, n_samples - 1, 8, dtype=int)

    ax.set_xticks(indices + 0.5)
    ax.set_xticklabels([f"{time_vector[i]:.2f}" for i in indices], rotation=45)
    ax.set_yticks(indices + 0.5)
    ax.set_yticklabels([f"{time_vector[i]:.2f}" for i in indices], rotation=0)

    # Draw vertical and horizontal lines at ts
    ts = round(ts, 2)
    ax.axvline(x=ts_idx + 0.5, color='red', linestyle='--', alpha=0.8, label=f'$t_s = {ts}$')
    ax.axhline(y=ts_idx + 0.5, color='red', linestyle='--', alpha=0.8)

    plt.title(f"SAGD Distance Matrix (d={d})", fontsize=14)
    plt.xlabel("Time", fontsize=12)
    plt.ylabel("Time", fontsize=12)
    plt.legend(loc='upper left')

    plt.tight_layout()
    if save_fig_path:
        plt.savefig(save_fig_path, dpi=300, bbox_inches='tight')
    plt.show()


# Three plotting functions below were generated with Claude
def _draw_sagd_heatmap_with_prob(
        ax_hm,
        W,
        time_vector,
        ts,
        ts_idx,
        d,
        mu,
        std,
        tsagd=None,
        tsagd_idx=None,
        show_ylabel=True,
        show_legend=True,
):
    """
    Draw a square SAGD heatmap into ``ax_hm`` and attach a same-cluster
    probability marginal on top and a colorbar on the right via an
    ``axes_grid1`` divider, so both track the heatmap's rendered size
    (jointplot-style).

    If ``tsagd`` and ``tsagd_idx`` are given, an additional green dashed
    marker is drawn at that time (heatmap v/h-line and prob v-line).
    """
    from ou_model import same_cluster_prob

    time_arr = np.asarray(time_vector)
    probs = same_cluster_prob(d, mu, std, time_arr)
    n_samples = len(time_arr)

    sns.heatmap(W, cmap='viridis', robust=True, ax=ax_hm, cbar=False, square=True)

    indices = np.linspace(0, n_samples - 1, 8, dtype=int)
    tick_labels = [f"{time_arr[i]:.2f}" for i in indices]
    ax_hm.set_xticks(indices + 0.5)
    ax_hm.set_xticklabels(tick_labels, rotation=45)
    ax_hm.set_yticks(indices + 0.5)
    ax_hm.set_yticklabels(tick_labels, rotation=0)

    ts_r = round(ts, 2)
    ax_hm.axvline(x=ts_idx + 0.5, color='red', linestyle='--', alpha=0.8, label=f'$t_s$')
    ax_hm.axhline(y=ts_idx + 0.5, color='red', linestyle='--', alpha=0.8)

    if tsagd is not None and tsagd_idx is not None:
        tsagd_r = round(tsagd, 2)
        ax_hm.axvline(x=tsagd_idx + 0.5, color='orange', linestyle='--', alpha=0.8,
                      label=f'$t_{{SAGD}}$')
        ax_hm.axhline(y=tsagd_idx + 0.5, color='orange', linestyle='--', alpha=0.8)

    ax_hm.set_xlabel("Time", fontsize=12)
    if show_ylabel:
        ax_hm.set_ylabel("Time", fontsize=12)
    if show_legend:
        ax_hm.legend(loc='upper left')

    divider = make_axes_locatable(ax_hm)
    ax_prob = divider.append_axes("top", size="22%", pad=0.2, sharex=ax_hm)
    cbar_ax = divider.append_axes("right", size="4%", pad=0.1)

    plt.colorbar(ax_hm.collections[0], cax=cbar_ax)

    x_positions = np.arange(n_samples) + 0.5
    ax_prob.plot(x_positions, probs, color='steelblue', lw=1.5)
    ax_prob.axvline(x=ts_idx + 0.5, color='red', linestyle='--', alpha=0.8)
    if tsagd is not None and tsagd_idx is not None:
        ax_prob.axvline(x=tsagd_idx + 0.5, color='orange', linestyle='--', alpha=0.8)
    ax_prob.set_ylim(0.48, 1.02)
    ax_prob.tick_params(axis='x', which='both', bottom=False, labelbottom=False)
    ax_prob.set_ylabel("φ(t)", fontsize=12)
    ax_prob.set_title(f"SAGD Distance Matrix (d={d})", fontsize=14)
    sns.despine(ax=ax_prob, top=True, right=True, bottom=True, left=False)


def plot_sagd_heatmap_with_prob(
        W,
        time_vector,
        ts,
        ts_idx,
        d,
        mu,
        std,
        tsagd=None,
        tsagd_idx=None,
        save_fig_path=None
):
    """
    SAGD distance matrix heatmap with a same-cluster probability curve
    (ou_model.same_cluster_prob) on top, sharing the time (x) axis like a
    seaborn jointplot.

    Parameters
    ----------
    mu : float
        Per-coordinate cluster amplitude (so the cluster center is mu*1_d).
    std : float
        Per-cluster internal standard deviation.
    tsagd, tsagd_idx : float, int, optional
        If both provided, a green dashed marker is drawn at this time on
        the heatmap and on the probability marginal.
    """
    fig, ax_hm = plt.subplots(figsize=(8, 9))
    _draw_sagd_heatmap_with_prob(
        ax_hm, W, time_vector, ts, ts_idx, d, mu, std,
        tsagd=tsagd, tsagd_idx=tsagd_idx,
        show_ylabel=True, show_legend=True,
    )
    if save_fig_path:
        plt.savefig(save_fig_path, dpi=300, bbox_inches='tight')
    plt.show()


def plot_sagd_heatmap_row_with_prob(
        W_list,
        d_list,
        time_snaps_vector_list,
        ts_tuple_list,
        mu,
        std,
        tsagd_tuple_list=None,
        save_fig_path=None,
):
    """
    Row of (probability curve on top, SAGD heatmap below) jointplot-style
    panels across dimensions. Each column's marginal and colorbar are pinned
    to the heatmap's rendered size via an ``axes_grid1`` divider.

    ``tsagd_tuple_list`` is an optional list aligned with ``W_list`` of
    ``(tsagd, tsagd_idx)`` pairs (use ``None`` for columns without one).
    Where provided, a green dashed marker is drawn alongside the red
    speciation lines.
    """
    num_plots = len(W_list)
    fig, axes = plt.subplots(
        1, num_plots,
        figsize=(6 * num_plots, 7.5),
        gridspec_kw={'wspace': 0.5},
    )
    if num_plots == 1:
        axes = [axes]

    if tsagd_tuple_list is None:
        tsagd_tuple_list = [None] * num_plots

    for i, (W, d, time_vec, ts_tuple, tsagd_tuple) in enumerate(
            zip(W_list, d_list, time_snaps_vector_list, ts_tuple_list, tsagd_tuple_list)):
        ts, ts_idx = ts_tuple
        if tsagd_tuple is None:
            tsagd, tsagd_idx = None, None
        else:
            tsagd, tsagd_idx = tsagd_tuple
        _draw_sagd_heatmap_with_prob(
            axes[i],
            W, time_vec, ts, ts_idx, d, mu, std,
            tsagd=tsagd, tsagd_idx=tsagd_idx,
            show_ylabel=(i == 0),
            show_legend=(i == 0),
        )

    if save_fig_path:
        plt.savefig(save_fig_path, dpi=300, bbox_inches='tight')
    plt.show()


def plot_breakpoint_and_speciation(
        d_list,
        ts_tuple_list,
        tsagd_tuple_list,
        mu,
        std,
        save_fig_path=None,
):
    """
    Plot ``1 - φ(t_s)`` and ``1 - φ(t_SAGD)`` across dimensions on a
    log-scale y-axis.
    """
    from ou_model import same_cluster_prob

    phi_ts = np.zeros(len(d_list))
    phi_tsagd = np.zeros(len(d_list))
    for i, (d, ts_tuple, tsagd_tuple) in enumerate(
            zip(d_list, ts_tuple_list, tsagd_tuple_list)):
        ts, _ = ts_tuple
        tsagd, _ = tsagd_tuple
        phi_ts[i] = same_cluster_prob(d, mu, std, np.array([ts]))[0]
        phi_tsagd[i] = same_cluster_prob(d, mu, std, np.array([tsagd]))[0]

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(d_list, 1 - phi_ts, marker='o', color='red',
            label=r'$1 - φ(t_s)$')
    ax.plot(d_list, 1 - phi_tsagd, marker='s', color='orange',
            label=r'$1 - φ(t_{SAGD})$')
    ax.set_yscale('log')
    ax.set_xscale('log', base=2)
    ax.set_xlim(min(d_list), max(d_list))
    ax.set_xticks(list(d_list))
    ax.set_xticklabels([str(d) for d in d_list])
    ax.minorticks_off()
    ax.set_xlabel('Dimension', fontsize=12)
    ax.set_ylabel(r'$1 - φ(t)$', fontsize=12)
    ax.set_title('Same-cluster probability at $t_s$ and $t_{SAGD}$',
                 fontsize=14)
    ax.legend()
    ax.grid(True, which='both', alpha=0.3)

    plt.tight_layout()
    if save_fig_path:
        plt.savefig(save_fig_path, dpi=300, bbox_inches='tight')
    plt.show()
