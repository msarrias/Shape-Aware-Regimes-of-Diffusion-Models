# Gemini wrote this for me.
import numpy as np
from matplotlib import cm
import torch
import seaborn as sns
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from RRP import RRP
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


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


def plot_sasne_dimension_sweep(
        sasne_results,
        d_list,
        save_path
):
    """
    Plots SASNE embeddings and Rank Resilience Plots (RRP) for a sweep of dimensions.

    Parameters:
    - sasne_results: List of tuples (embedding, Z, D1, D2, idx)
    - d_list: List of dimension values corresponding to the results
    - save_path: Path to save the resulting figure
    """
    num_cols = len(d_list)
    fig, axes = plt.subplots(2, num_cols, figsize=(5 * num_cols, 10))

    # Ensure axes is 2D even if num_cols is 1
    if num_cols == 1:
        axes = np.atleast_2d(axes).T

    for i, (result, d) in enumerate(zip(sasne_results, d_list)):
        _, embedding, Z, D1, D2, _, ts_idx, _ = result

        # --- ROW 1: SASNE Projection ---
        ax_proj = axes[0, i]
        ax_proj.set_title(f'Dimension d = {d}', fontsize=14, fontweight='bold')

        # Main trajectory
        colors = np.arange(len(embedding))
        ax_proj.scatter(embedding[:, 0], embedding[:, 1], s=15, alpha=0.4,
                        c=colors, cmap='viridis')

        # Highlight specific points
        # t=T (Start of diffusion/Noise)
        ax_proj.scatter(embedding[0, 0], embedding[0, 1], c='red', s=80,
                        edgecolors='black', label='t=T', zorder=5)
        ax_proj.annotate('$t=T$', (embedding[0, 0], embedding[0, 1]),
                         xytext=(5, 5), textcoords='offset points')

        # t=0 (End of diffusion/Clean Data)
        ax_proj.scatter(embedding[-1, 0], embedding[-1, 1], c='blue', s=80,
                        edgecolors='black', label='t=0', zorder=5)
        ax_proj.annotate('$t=0$', (embedding[-1, 0], embedding[-1, 1]),
                         xytext=(5, -15), textcoords='offset points')

        # t_s (Speciation point)
        ax_proj.scatter(embedding[ts_idx, 0], embedding[ts_idx, 1], c='green', s=80,
                        edgecolors='black', zorder=6)
        ax_proj.annotate('$t_s$', (embedding[ts_idx, 0], embedding[ts_idx, 1]),
                         xytext=(-15, -15), textcoords='offset points')

        ax_proj.set_xlabel('SASNE1')
        if i == 0: ax_proj.set_ylabel('SASNE2')

        # Add a bit of padding to axis limits
        x_pad = (embedding[:, 0].max() - embedding[:, 0].min()) * 0.1
        y_pad = (embedding[:, 1].max() - embedding[:, 1].min()) * 0.1
        ax_proj.set_xlim(embedding[:, 0].min() - x_pad, embedding[:, 0].max() + x_pad)
        ax_proj.set_ylim(embedding[:, 1].min() - y_pad, embedding[:, 1].max() + y_pad)

        # --- ROW 2: Rank Resilience Plot (RRP) ---
        ax_rank = axes[1, i]
        plt.sca(ax_rank)  # Set current axes for RRP if it relies on plt calls

        # Assuming RRP is a pre-defined function that plots onto current axis
        RRP(D1, D2)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    

def plot_full_sasne_dashboard(
        sasne_results,
        d_list,
        time_snaps_vector,
        save_path=None
):
    """
    Plots a 3-row dashboard for each dimension:
    Row 1: SASNE Embedding Trajectory
    Row 2: Rank Resilience Plot (RRP)
    Row 3: SAGD Distance Heatmap
    """
    num_cols = len(d_list)
    # 3 Rows now: Projections, RRPs, and Heatmaps
    fig, axes = plt.subplots(3, num_cols, figsize=(5 * num_cols, 15))

    # Handle the single-column case
    if num_cols == 1:
        axes = axes.reshape(3, 1)

    # Prepare labels for the heatmaps (Row 3)
    n_samples = len(time_snaps_vector)
    heat_indices = np.linspace(0, n_samples - 1, 8, dtype=int)
    tick_labels = [f"{time_snaps_vector[i]:.2f}" for i in heat_indices]

    for i, (result, d) in enumerate(zip(sasne_results, d_list)):
        # result structure: (embedding, Z, D1, D2, W, ts_idx, ts_val)
        # Note: adjust the unpacking if your 'result' tuple order is different
        SAGD_dist_matrix, embedding, Z, D1, D2, time_snaps, ts_idx, t_s = result
        cbar = True if i == len(sasne_results) - 1 else False

        # --- ROW 1: SASNE Projection ---
        ax_proj = axes[0, i]
        ax_proj.set_title(f'Dimension d = {d}', fontsize=16, fontweight='bold')

        colors = np.arange(len(embedding))
        ax_proj.scatter(embedding[:, 0], embedding[:, 1], s=15, alpha=0.4,
                        c=colors, cmap='viridis')

        # Mark t=T, t=0, and ts
        ax_proj.scatter(embedding[0, 0], embedding[0, 1], c='red', s=80, edgecolors='black', zorder=5)
        ax_proj.scatter(embedding[-1, 0], embedding[-1, 1], c='blue', s=80, edgecolors='black', zorder=5)
        ax_proj.scatter(embedding[ts_idx, 0], embedding[ts_idx, 1], c='green', s=80, edgecolors='black', zorder=6)

        ax_proj.set_xlabel('SASNE1')
        if i == 0: ax_proj.set_ylabel('SASNE2')

        # --- ROW 2: Rank Resilience Plot (RRP) ---
        ax_rank = axes[1, i]
        plt.sca(ax_rank)  # Ensure RRP plots on the correct axis
        RRP(D1, D2)

        # --- ROW 3: SAGD Distance Heatmap ---
        ax_heat = axes[2, i]
        sns.heatmap(SAGD_dist_matrix, cmap='viridis', robust=True, ax=ax_heat, cbar=cbar, cbar_kws={'shrink': 0.8})

        # Add the Speciation Lines
        ax_heat.axvline(x=ts_idx + 0.5, color='red', linestyle='--', alpha=0.8, label=f'$t_s={round(t_s,2):.2f}$')
        ax_heat.axhline(y=ts_idx + 0.5, color='red', linestyle='--', alpha=0.8)

        # Heatmap Ticks
        ax_heat.set_xticks(heat_indices + 0.5)
        ax_heat.set_xticklabels(tick_labels, rotation=45)
        ax_heat.set_yticks(heat_indices + 0.5)
        ax_heat.set_yticklabels(tick_labels, rotation=0)
        ax_heat.set_xlabel("Time")
        if i == 0:
            ax_heat.set_ylabel("Time")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def plot_sagd_heatmap_row(
        W_list,
        d_list,
        time_snaps_vector_list,
        ts_tuple_list=None,
        save_fig_path=None
):
    num_plots = len(W_list)
    max_cols = 6
    num_rows = (num_plots + max_cols - 1) // max_cols
    num_cols = min(num_plots, max_cols)

    fig, axes = plt.subplots(num_rows, num_cols, figsize=(5 * num_cols, 5 * num_rows))
    axes_flat = axes.flatten() if num_plots > 1 else [axes]

    for i, (W, d, time_snaps_vector) in enumerate(zip(W_list, d_list, time_snaps_vector_list)):
        ax_heat = axes_flat[i]        
        # 1. Normalize current matrix to 0-1
        # W_min, W_max = W.min(), W.max()
        # W_norm = (W - W_min) / (W_max - W_min)
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
        if ts_tuple_list:
            ts, ts_idx = ts_tuple
            ax_heat.axvline(x=ts_idx + 0.5, color='red', linestyle='--', alpha=0.8, label=f'$t_s = {ts:.2f}$')
            ax_heat.axhline(y=ts_idx + 0.5, color='red', linestyle='--', alpha=0.8)
    
        ax_heat.set_title(f"d={d}", fontsize=16)
        if i % max_cols == 0: 
            ax_heat.set_ylabel("Time", fontsize=12)

    # Delete unused axes
    for j in range(i + 1, len(axes_flat)): 
        fig.delaxes(axes_flat[j])

    plt.tight_layout()
    if save_fig_path: 
        plt.savefig(save_fig_path, dpi=300)
    plt.show()