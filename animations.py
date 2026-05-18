# Gemini wrote this for me.
import numpy as np
from matplotlib import cm
import torch
import seaborn as sns
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from RRP import RRP
from matplotlib import pyplot as plt
import matplotlib.gridspec as gridspec
from mpl_toolkits.mplot3d import Axes3D


def create_synchronized_animation(
    history,             # List of [1000, 2] spatial snapshots
    time_snaps,          # Time values (t=T to t=0)
    ts,                  # Speciation time value
    ts_idx,              # Index where t approx ts
    SAGD_dist_matrix,    # The [100, 100] pre-computed matrix
    node_labels,
    save_path
):
    n_frames = len(history)
    fig = plt.figure(figsize=(16, 8))
    gs = gridspec.GridSpec(1, 2, width_ratios=[1, 1.2])
    n_samples = len(time_snaps)
    heat_indices = np.linspace(0, n_samples - 1, 8, dtype=int)
    tick_labels = [f"{time_snaps[i]:.2f}" for i in heat_indices]
    
    # --- Left: Spatial Embedding ---
    ax_emb = fig.add_subplot(gs[0])
    all_coords = np.concatenate(history)
    ax_emb.set_xlim(all_coords[:, 0].min() - 1, all_coords[:, 0].max() + 1)
    ax_emb.set_ylim(all_coords[:, 1].min() - 1, all_coords[:, 1].max() + 1)
    ax_emb.set_title(r"Data Progression")
    
    # Initialize points colored by the final classification
    sc = ax_emb.scatter(history[0][:, 0], history[0][:, 1], c=node_labels, s=25, alpha=0.6)

    # 1. Normalize current matrix to 0-1
    W_min, W_max = SAGD_dist_matrix.min(), SAGD_dist_matrix.max()
    W_norm = (SAGD_dist_matrix - W_min) / (W_max - W_min) if W_max > W_min else SAGD_dist_matrix - W_min

    # --- Right: Temporal Distance Matrix ---
    ax_heat = fig.add_subplot(gs[1])
    sns.heatmap(W_norm, cmap='viridis', ax=ax_heat, vmin=0, vmax=1, cbar_kws={'shrink': 0.8})

    ax_heat.axvline(x=ts_idx + 0.5, color='red', linestyle='--', alpha=0.8, label=f'$t_s={ts:.2f}$')
    ax_heat.axhline(y=ts_idx + 0.5, color='red', linestyle='--', alpha=0.8)
    
    current_time_line_v = ax_heat.axvline(x=0, color='white', linewidth=2)
    current_time_line_h = ax_heat.axhline(y=0, color='white', linewidth=2)

    # Heatmap Ticks
    ax_heat.set_xticks(heat_indices + 0.5)
    ax_heat.set_xticklabels(tick_labels, rotation=45)
    ax_heat.set_yticks(heat_indices + 0.5)
    ax_heat.set_yticklabels(tick_labels, rotation=0)
    ax_heat.set_xlabel("Time")
    ax_heat.set_ylabel("Time")
    ax_heat.set_title("SAGD Distance Matrix")
    ax_heat.legend(loc='upper right')

    # Status Text
    time_text = ax_emb.text(0.05, 0.95, '', transform=ax_emb.transAxes, fontweight='bold', fontsize=12)
    regime_text = ax_emb.text(0.05, 0.90, '', transform=ax_emb.transAxes, fontsize=10, style='italic')

    def update(frame):
        current_t = time_snaps[frame]
        
        # 1. Update Spatial Points
        sc.set_offsets(history[frame])

        current_time_line_v.set_xdata([frame + 0.5, frame + 0.5])
        current_time_line_h.set_ydata([frame + 0.5, frame + 0.5])

        if current_t > ts:
            label = "Regime I"
            color = 'gray'
        else:
            label = "Regime II"
            color = 'darkcyan'

        time_text.set_text(f"Time t = {current_t:.2f}")
        regime_text.set_text(label)
        regime_text.set_color(color)
            
        return sc, current_time_line_v, current_time_line_h, regime_text

    ani = FuncAnimation(fig, update, frames=n_frames, interval=300, blit=False)
    ani.save(save_path, writer='pillow', fps=3)
    plt.close()


def create_synchronized_3d_animation(
    history,             # List of [1000, 3] spatial snapshots
    time_snaps,          # Time values (t=T to t=0)
    ts,                  # Speciation time value
    ts_idx,              # Index where t approx ts
    SAGD_dist_matrix,    # The pre-computed temporal matrix
    node_labels,         # Your 'red'/'blue' color list
    save_path
):
    n_frames = len(history)
    fig = plt.figure(figsize=(18, 9))
    gs = gridspec.GridSpec(1, 2, width_ratios=[1, 1.2])
    
    n_samples = len(time_snaps)
    heat_indices = np.linspace(0, n_samples - 1, 8, dtype=int)
    tick_labels = [f"{time_snaps[i]:.2f}" for i in heat_indices]
    
    # --- Left: 3D Spatial Embedding ---
    ax_emb = fig.add_subplot(gs[0], projection='3d')
    all_coords = np.concatenate(history)
    
    ax_emb.set_xlim(all_coords[:, 0].min(), all_coords[:, 0].max())
    ax_emb.set_ylim(all_coords[:, 1].min(), all_coords[:, 1].max())
    ax_emb.set_zlim(all_coords[:, 2].min(), all_coords[:, 2].max())
    ax_emb.set_title("Data Progression")

    # In 3D, scatter returns a Path3DCollection
    sc = ax_emb.scatter(history[0][:, 0], history[0][:, 1], history[0][:, 2], c=node_labels, s=20, alpha=0.6)

    # 1. Normalize current matrix to 0-1
    W_min, W_max = SAGD_dist_matrix.min(), SAGD_dist_matrix.max()
    W_norm = (SAGD_dist_matrix - W_min) / (W_max - W_min) if W_max > W_min else SAGD_dist_matrix - W_min

    # --- Right: Temporal Distance Matrix ---
    ax_heat = fig.add_subplot(gs[1])
    sns.heatmap(W_norm, cmap='viridis', ax=ax_heat, vmin=0, vmax=1, cbar_kws={'shrink': 0.8})
    ax_heat.axvline(x=ts_idx + 0.5, color='red', linestyle='--', alpha=0.8, label=f'$t_s={ts:.2f}$')
    ax_heat.axhline(y=ts_idx + 0.5, color='red', linestyle='--', alpha=0.8)
    
    current_time_line_v = ax_heat.axvline(x=0, color='white', linewidth=2)
    current_time_line_h = ax_heat.axhline(y=0, color='white', linewidth=2)

    # Heatmap Ticks
    ax_heat.set_xticks(heat_indices + 0.5)
    ax_heat.set_xticklabels(tick_labels, rotation=45)
    ax_heat.set_yticks(heat_indices + 0.5)
    ax_heat.set_yticklabels(tick_labels, rotation=0)
    ax_heat.set_xlabel("Time")
    ax_heat.set_ylabel("Time")
    ax_heat.set_title("SAGD Distance Matrix")
    ax_heat.legend(loc='upper right')

    # Status Text
    time_text = fig.text(0.05, 0.93, '', fontweight='bold', fontsize=12)
    regime_text = fig.text(0.05, 0.89, '', fontsize=11, style='italic')

    def update(frame):
        current_t = time_snaps[frame]
        
        # 1. Update 3D Points using _offsets3d
        pos = history[frame]
        sc._offsets3d = (pos[:, 0], pos[:, 1], pos[:, 2])
        
        # 2. Update Heatmap Indicators
        current_time_line_v.set_xdata([frame + 0.5, frame + 0.5])
        current_time_line_h.set_ydata([frame + 0.5, frame + 0.5])
        if current_t > ts:
            label = "Regime I"
            color = 'gray'
        else:
            label = "Regime II"
            color = 'darkcyan'
            
        time_text.set_text(f"Time t = {current_t:.2f}")
        regime_text.set_text(label)
        regime_text.set_color(color)
            
        return sc, current_time_line_v, current_time_line_h, time_text, regime_text

    ani = FuncAnimation(fig, update, frames=n_frames, interval=300, blit=False)
    ani.save(save_path, writer='pillow', fps=3)
    plt.close()


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
            scatter_head.set_color('darkcyan')

        return [scatter_head, line_tail, time_text] + list(milestones.values())

    ani = FuncAnimation(fig, update, frames=len(embedding),
                        init_func=init, blit=True, interval=150, repeat=False)
    
    ani.save(save_path, writer='pillow', fps=5, metadata={'loop': 1})
    plt.close()


def create_animated_embedding(
        embedding,
        d,
        tsagd_idx,
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
        if frame == tsagd_idx:
            time_text.set_text(f"t = {current_t:.2f} * (Speciation)")
            time_text.set_color('darkcyan') # Match the milestone color
        else:
            time_text.set_text(f"t = {current_t:.2f}")
            time_text.set_color('black')
            
        # Annotation Logic
        if frame == 0 and 'start' not in milestones:
            milestones['start'] = add_pointer_ani(ax, 0, '$t=T$', 'red', (-80, 40))
        
        if frame == tsagd_idx and 'ts' not in milestones:
            milestones['ts'] = add_pointer_ani(ax, tsagd_idx, '$t=t_{\mathrm{SAGD}}$', 'darkcyan', (40, 40))
            
        if frame == n_frames - 1 and 'end' not in milestones:
            milestones['end'] = add_pointer_ani(ax, n_frames - 1, '$t=0$', 'orange', (20, -40))
            
        return [sc, time_text] + list(milestones.values())
    ani = FuncAnimation(fig, update, frames=n_frames, interval=200, blit=False, repeat=False)
    animation_path = save_path + ".gif"
    ani.save(animation_path, writer='pillow', fps=5, savefig_kwargs={'facecolor':'white'}, metadata={'loop': 1})
    frame_path = save_path + ".png"
    fig.savefig(frame_path)
    plt.close()


def create_ctd_synchronized_animation(
    d,
    ctds_list,            
    time_snaps,           
    ts,                   
    ts_idx,               
    SAGD_dist_matrix,     
    save_path
):
    n_frames = len(ctds_list)
    fig = plt.figure(figsize=(16, 8))
    gs = gridspec.GridSpec(1, 2, width_ratios=[1, 1.2])
    
    shared_bins = np.linspace(0, 1, 51) 
    n_samples = len(time_snaps)
    
    # --- Right: Setup Heatmap (Static Background) ---
    W_min, W_max = SAGD_dist_matrix.min(), SAGD_dist_matrix.max()
    W_norm = (SAGD_dist_matrix - W_min) / (W_max - W_min) if W_max > W_min else SAGD_dist_matrix
    
    ax_heat = fig.add_subplot(gs[1])
    sns.heatmap(W_norm, cmap='viridis', ax=ax_heat, vmin=0, vmax=1, cbar_kws={'shrink': 0.8})
    
    # Speciation lines
    ax_heat.axvline(x=ts_idx + 0.5, color='red', linestyle='--', alpha=0.8, label=f'$t_s={ts:.2f}$')
    ax_heat.axhline(y=ts_idx + 0.5, color='red', linestyle='--', alpha=0.8)
    
    # Dynamic indicator crosshair
    current_time_line_v = ax_heat.axvline(x=0, color='white', linewidth=2)
    current_time_line_h = ax_heat.axhline(y=0, color='white', linewidth=2)
    
    heat_indices = np.linspace(0, n_samples - 1, 8, dtype=int)
    tick_labels = [f"{time_snaps[i]:.2f}" for i in heat_indices]
    ax_heat.set_xticks(heat_indices + 0.5)
    ax_heat.set_xticklabels(tick_labels, rotation=45)
    ax_heat.set_yticks(heat_indices + 0.5)
    ax_heat.set_yticklabels(tick_labels, rotation=0)
    ax_heat.set_title("SAGD Distance Matrix")
    ax_heat.legend(loc='upper right')

    # --- Left: Spatial/CTD Distribution ---
    ax_emb = fig.add_subplot(gs[0])
    ax_emb.set_title(f"Normalized CTD Distribution d = {d}")
    
    _, _, bar_container = ax_emb.hist(ctds_list[0], bins=shared_bins, alpha=0.6, 
                                      density=True, color='darkcyan', edgecolor='black')

    time_text = ax_emb.text(0.05, 0.95, '', transform=ax_emb.transAxes, fontweight='bold', fontsize=12)
    regime_text = ax_emb.text(0.05, 0.90, '', transform=ax_emb.transAxes, fontsize=11, style='italic')

    def update(frame):
        current_t = time_snaps[frame]
        data = ctds_list[frame]
        
        counts, _ = np.histogram(data, bins=shared_bins, density=True)
        for count, rect in zip(counts, bar_container):
            rect.set_height(count)

        current_time_line_v.set_xdata([frame + 0.5, frame + 0.5])
        current_time_line_h.set_ydata([frame + 0.5, frame + 0.5])
        time_text.set_text(f"Time t = {current_t:.2f}")
            
        return bar_container.patches + [current_time_line_v, current_time_line_h, time_text, regime_text]

    ani = FuncAnimation(fig, update, frames=n_frames, interval=300, blit=True)
    ani.save(save_path, writer='pillow', fps=5)
    plt.close()
    