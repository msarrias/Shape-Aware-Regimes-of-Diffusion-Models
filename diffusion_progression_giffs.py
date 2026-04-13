# Gemini did this for me. Thank you Gemini
import matplotlib.pyplot as plt
from matplotlib import rc
from matplotlib.animation import FuncAnimation
import numpy as np
import torch
import os
import joblib
from scipy.spatial.distance import pdist, squareform

from ou_model import theoretical_ts
from SASNE import SASNE
from RRP import RRP


def create_animated_embedding(embedding, d, ts_idx, time_snaps, save_path):
    n_frames = len(embedding)
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.set_xlim(embedding[:, 0].min() - 1.5, embedding[:, 0].max() + 1.5)
    ax.set_ylim(embedding[:, 1].min() - 1.5, embedding[:, 1].max() + 1.5)
    ax.set_title(f"Backward Diffusion Progression (d={d})", fontsize=14)
    ax.set_xlabel("SASNE1")
    ax.set_ylabel("SASNE2")

    sc = ax.scatter([], [], c=[], cmap='viridis', s=25, alpha=0.8, vmin=0, vmax=n_frames)
    time_text = ax.text(0.05, 0.95, '', transform=ax.transAxes, fontweight='bold')

    milestones = {}

    def add_pointer_ani(ax, index, label, color, offset):
        return ax.annotate(label, 
                    xy=(embedding[index, 0], embedding[index, 1]), 
                    xytext=offset, 
                    textcoords='offset points',
                    arrowprops=dict(arrowstyle='->', color=color, lw=2),
                    bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=color, alpha=0.8),
                    fontsize=10, fontweight='bold', color=color, zorder=10)

    def update(frame):
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


def create_cont_sasne_animation(embedding, time_snaps, d, save_path):
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(embedding[:, 0], embedding[:, 1], c='gray', alpha=0.2, zorder=1)
    scatter_head = ax.scatter([], [], c='red', s=100, edgecolors='white', zorder=5)
    line_tail, = ax.plot([], [], c='blue', alpha=0.6, lw=2, zorder=4)
    ax.set_title(f"Diffusion Progression (d={d})", fontsize=14)
    ax.set_xlabel("SASNE1")
    ax.set_ylabel("SASNE2")
    ax.set_xlim(embedding[:, 0].min() - 0.5, embedding[:, 0].max() + 0.5)
    ax.set_ylim(embedding[:, 1].min() - 0.5, embedding[:, 1].max() + 0.5)
    
    time_text = ax.text(0.05, 0.95, '', transform=ax.transAxes, fontweight='bold')

    def init():
        scatter_head.set_offsets(np.empty((0, 2)))
        line_tail.set_data([], [])
        time_text.set_text('')
        return scatter_head, line_tail, time_text

    def update(frame):
        x, y = embedding[frame, 0], embedding[frame, 1]
        scatter_head.set_offsets([[x, y]])
        start_tail = max(0, frame - 10)
        line_tail.set_data(embedding[start_tail:frame+1, 0], 
                           embedding[start_tail:frame+1, 1])
        time_text.set_text(f"t = {time_snaps[frame]:.2f}")
        return scatter_head, line_tail, time_text

    ani = FuncAnimation(fig, update, frames=len(embedding),
                                  init_func=init, blit=True, interval=200)
    ani.save(save_path, writer='pillow', fps=5, metadata={'loop': 1})
    plt.close()
    

if __name__ == "__main__":
    sasne_embeddings_list = []
    d_list = [2, 256, 1024, 4096, 16384]
    for d in d_list:
        print('d=', d)
        mu_star, std = torch.ones(d) * 4, 1.0
        loaded_data = joblib.load(f"data/D{d}_N1000_100ts/D{d}_N1000.jbl")
        times = loaded_data['params']['times']
        time_snaps = loaded_data['params']['times_snapshots']
        t_s, ts_idx = theoretical_ts(mu_star, std, times)
        ts_idx = np.argmin(np.abs(time_snaps - t_s))
        SAGD_dist_matrix = joblib.load(f"data/D{d}_N1000_100ts/D{d}_N1000_SAGD.jbl")
        embedding, Z = SASNE(SAGD_dist_matrix)
        D1 = squareform(pdist(embedding)) 
        D2 = squareform(pdist(Z)) 
        sasne_embeddings_list.append((embedding, Z, D1, D2, ts_idx))
        print(' ')


    for result, d in zip(sasne_embeddings_list, d_list):
        embedding, _, _, _, ts_idx = result
        save_path = f"data/D{d}_N1000_100ts/diffusion_d{d}.gif"
        create_animated_embedding(embedding, d, ts_idx, time_snaps, save_path)
        save_path = f"data/D{d}_N1000_100ts/diffusion_d{d}_v2.gif"
        create_cont_sasne_animation(embedding, time_snaps, d, save_path)
        print(f"d={d}, Animation saved")


        
        