# Gemini wrote this for me.
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
import torch
import seaborn as sns
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib import pyplot as plt


def plot_data_distribution(
        step_idx,
        title,
        ax,
        path_history,
        times,
        mu_star,
        std

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


def create_cont_sasne_animation(
        embedding,
        time_snaps,
        d,
        ts_idx,
        save_path
):
    fig, ax = plt.subplots(figsize=(6, 6))
    
    ax.plot(embedding[:, 0], embedding[:, 1], c='gray', alpha=0.15, zorder=1)
    
    scatter_head = ax.scatter([], [], c='red', s=100, edgecolors='white', zorder=6)
    line_tail, = ax.plot([], [], c='blue', alpha=0.6, lw=2, zorder=5)
    
    ax.set_title(f"d={d}", fontsize=14)
    ax.set_xlabel("SASNE1")
    ax.set_ylabel("SASNE2")
    ax.set_xlim(embedding[:, 0].min() - 0.7, embedding[:, 0].max() + 0.7)
    ax.set_ylim(embedding[:, 1].min() - 0.7, embedding[:, 1].max() + 0.7)
    
    time_text = ax.text(0.05, 0.95, '', transform=ax.transAxes, 
                        fontweight='bold', fontsize=12,
                        bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))

    milestones = {}

    def init():
        scatter_head.set_offsets(np.empty((0, 2)))
        line_tail.set_data([], [])
        time_text.set_text('')
        return scatter_head, line_tail, time_text

    def update(frame):
        # Current position
        x, y = embedding[frame, 0], embedding[frame, 1]
        scatter_head.set_offsets([[x, y]])
        
        # Trailing tail
        start_tail = max(0, frame - 10)
        line_tail.set_data(embedding[start_tail:frame+1, 0], 
                           embedding[start_tail:frame+1, 1])
        
        # Time Label
        current_t = time_snaps[frame]
        time_text.set_text(f"t = {current_t:.2f}")

        # Speciation Pointer Logic
        if frame >= ts_idx and 'ts' not in milestones:
            milestones['ts'] = ax.annotate(
                f'$t_s$', 
                xy=(embedding[ts_idx, 0], embedding[ts_idx, 1]), 
                xytext=(40, 40), textcoords='offset points',
                arrowprops=dict(arrowstyle='->', color='darkcyan', lw=2),
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="darkcyan", alpha=0.8),
                fontsize=10, fontweight='bold', color='darkcyan', zorder=10
            )
            # Change head color at speciation for visual feedback
            scatter_head.set_color('darkcyan')

        # Collect artists for blitting
        return [scatter_head, line_tail, time_text] + list(milestones.values())

    # repeat=False ensures it stops at t=0
    ani = FuncAnimation(fig, update, frames=len(embedding),
                        init_func=init, blit=True, interval=150, repeat=False)
    ani.save(save_path, writer='pillow', fps=5, metadata={'loop': 1})
    plt.close()


def create_animated_embedding(
        embedding,
        d,
        ts_idx,
        time_snaps,
        save_path
):
    n_frames = len(embedding)
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.set_xlim(embedding[:, 0].min() - 1.5, embedding[:, 0].max() + 1.5)
    ax.set_ylim(embedding[:, 1].min() - 1.5, embedding[:, 1].max() + 1.5)
    ax.set_title(f"d={d}", fontsize=14)
    ax.set_xlabel("SASNE1")
    ax.set_ylabel("SASNE2")

    sc = ax.scatter([], [], c=[], cmap='viridis', s=25, alpha=0.8, vmin=0, vmax=n_frames)
    time_text = ax.text(0.05, 0.95, '', transform=ax.transAxes, fontweight='bold')

    milestones = {}

    def add_pointer_ani(
            ax,
            index,
            label,
            color,
            offset
    ):
        return ax.annotate(label, 
                    xy=(embedding[index, 0], embedding[index, 1]), 
                    xytext=offset, 
                    textcoords='offset points',
                    arrowprops=dict(arrowstyle='->', color=color, lw=2),
                    bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=color, alpha=0.8),
                    fontsize=10, fontweight='bold', color=color, zorder=10)

    def update(
            frame
    ):
        current_data = embedding[:frame+1]
        sc.set_offsets(current_data)
        sc.set_array(np.arange(frame + 1))
        current_t = time_snaps[frame]
        if frame == ts_idx:
            time_text.set_text(f"t = {current_t:.2f} * (Speciation)")
            time_text.set_color('darkcyan') # Match the milestone color
        else:
            time_text.set_text(f"t = {current_t:.2f}")
            time_text.set_color('black')
            
        # Annotation Logic
        if frame == 0 and 'start' not in milestones:
            milestones['start'] = add_pointer_ani(ax, 0, '$t=T$', 'red', (-80, 40))
        
        if frame == ts_idx and 'ts' not in milestones:
            milestones['ts'] = add_pointer_ani(ax, ts_idx, '$t=t_s$', 'darkcyan', (40, 40))
            
        if frame == n_frames - 1 and 'end' not in milestones:
            milestones['end'] = add_pointer_ani(ax, n_frames - 1, '$t=0$', 'orange', (20, -40))
            
        return [sc, time_text] + list(milestones.values())
    ani = FuncAnimation(fig, update, frames=n_frames, interval=200, blit=False, repeat=False)
    ani.save(save_path, writer='pillow', fps=5, savefig_kwargs={'facecolor':'white'}, metadata={'loop': 1})
    plt.close()


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
        W_min, W_max = W.min(), W.max()
        ## Handle cases where min might equal max to avoid division by zero
        W_norm = (W - W_min) / (W_max - W_min) if W_max > W_min else W - W_min
        # W_norm = np.clip(W / np.percentile(W, 95), 0, 1)

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


# The plotting functions below were generated with Claude
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

    W_min, W_max = W.min(), W.max()
    W_norm = (W - W_min) / (W_max - W_min)
    sns.heatmap(W_norm, cmap='viridis', robust=True, ax=ax_hm, cbar=False, square=True)

    heat_indices = np.linspace(0, n_samples - 1, 8, dtype=int)
    tick_labels = [f"{time_vector[idx]:.2f}" for idx in heat_indices]
    ax_hm.set_xticks(heat_indices + 0.5)
    ax_hm.set_xticklabels(tick_labels, rotation=45)
    ax_hm.set_yticks(heat_indices + 0.5)
    ax_hm.set_yticklabels(tick_labels, rotation=0)

    ts_r = round(ts, 2)
    ax_hm.axvline(x=ts_idx + 0.5, color='red', linestyle='--', alpha=0.8, label=f'$t_s = {ts_r:.2f}$')
    ax_hm.axhline(y=ts_idx + 0.5, color='red', linestyle='--', alpha=0.8)

    if tsagd is not None and tsagd_idx is not None:
        tsagd_r = round(tsagd, 2)
        ax_hm.axvline(x=tsagd_idx + 0.5, color='orange', linestyle='--', alpha=0.8, label=f'$t_{{SAGD}}={tsagd_r:.2f}$')
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
    fig, ax_hm = plt.subplots(figsize=(8, 9))
    _draw_sagd_heatmap_with_prob(
        ax_hm=ax_hm,
        W=W,
        time_vector=time_vector,
        ts=ts,
        ts_idx=ts_idx,
        d=d,
        mu=mu,
        std=std,
        tsagd=tsagd,
        tsagd_idx=tsagd_idx,
        show_ylabel=True,
        show_legend=True
    )
    if save_fig_path:
        plt.savefig(save_fig_path, dpi=300, bbox_inches='tight')
    plt.show()


def plot_sagd_heatmap_row_with_prob(
        W_list: list,
        d_list: list,
        time_snaps_vector_list: list,
        ts_tuple_list: list,
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
        nrows=1,
        ncols=num_plots,
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
            ax_hm=axes[i],
            W=W,
            time_vector=time_vec,
            ts=ts,
            ts_idx=ts_idx,
            d=d,
            mu=mu,
            std=std,
            tsagd=tsagd,
            tsagd_idx=tsagd_idx,
            show_ylabel=(i == 0),
            show_legend=True,
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


def extended_gray_coolwarm():
    """Coolwarm-style colormap with a flat neutral-gray plateau in [0.25, 0.75]."""
    return LinearSegmentedColormap.from_list(
        "extended_gray_coolwarm",
        [
            (0.00, (0.23, 0.30, 0.75)),
            (0.25, (0.78, 0.78, 0.78)),
            (0.75, (0.78, 0.78, 0.78)),
            (1.00, (0.71, 0.02, 0.15)),
        ],
    )


def plot_state_and_ctd_frame(
    step_idx,
    history_list,
    norm_ctds_list,
    time_snaps,
    mu,
    std,
    fig=None,
    axes=None,
    scatter_lim=None,
    ctd_bins=51,
    ctd_xlim=(0.0, 1.0),
    ctd_ylim=None,
    show=True,
    save_fig_path=None,
):
    """
    Render a single frame with:
      Left  - data distribution at t = time_snaps[step_idx], colored by
              ``pos_cluster_prob`` using an extended-gray coolwarm colormap.
              Gray in [0.25, 0.75]; blue for p < 0.25; red for p > 0.75.
      Right - normalized CTD distribution at the same t.

    Pass existing ``fig`` and ``axes`` (length-2) to reuse them across frames.
    """
    from ou_model import pos_cluster_prob

    data = np.asarray(history_list[step_idx])
    n, d = data.shape
    t = float(time_snaps[step_idx])

    y = torch.as_tensor(data, dtype=torch.float32).T
    probs = pos_cluster_prob(y, t=t, dim=d, mu=mu, std=std)
    if torch.is_tensor(probs):
        probs = probs.detach().cpu().numpy()
    probs = np.asarray(probs)

    if d == 2:
        xy = data
        xlabel, ylabel = "$x_1$", "$x_2$"
    else:
        raise ValueError("Only d = 2 is supported for this plot")

    created_fig = False
    if axes is None:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        created_fig = True
    ax_left, ax_right = axes
    ax_left.clear()
    ax_right.clear()

    cmap = extended_gray_coolwarm()
    sc = ax_left.scatter(
        xy[:, 0], xy[:, 1],
        c=probs, cmap=cmap, vmin=0.0, vmax=1.0,
        s=25, alpha=0.85, edgecolors='none',
    )
    if scatter_lim is not None:
        ax_left.set_xlim(*scatter_lim)
        ax_left.set_ylim(*scatter_lim)
    ax_left.set_xlabel(xlabel)
    ax_left.set_ylabel(ylabel)
    ax_left.set_title(f"Data at t = {t:.2f}  (d = {d})")
    ax_left.set_aspect('equal', adjustable='datalim')
    ax_left.grid(alpha=0.15)

    if created_fig:
        cb = fig.colorbar(sc, ax=ax_left, fraction=0.046, pad=0.04)
        cb.set_label(r"$P(+\mu \mid x_t)$")

    ctd_vals = np.asarray(norm_ctds_list[step_idx])
    bins = np.linspace(ctd_xlim[0], ctd_xlim[1], ctd_bins)
    ax_right.hist(
        ctd_vals, bins=bins, density=True,
        color='darkcyan', edgecolor='black', alpha=0.7,
    )
    ax_right.set_xlim(*ctd_xlim)
    if ctd_ylim is not None:
        ax_right.set_ylim(*ctd_ylim)
    ax_right.set_xlabel("Normalized CTD")
    ax_right.set_ylabel("Density")
    ax_right.set_title(f"CTD distribution at t = {t:.2f}")
    ax_right.grid(alpha=0.15)

    if save_fig_path:
        fig.savefig(save_fig_path, dpi=200, bbox_inches='tight')
    if show and created_fig:
        plt.show()
    return fig, (ax_left, ax_right)