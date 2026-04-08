import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
import torch
import sys
from pyvis.network import Network
import networkx as nx

from ou_model import classify
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np


def plot_data_distribution(step_idx, title, ax, path_history, times, is_final=False):
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


def plot_speciation_3d(path_history, times, mu_star, std, nsamples, t_s=None, t_c=None, seed = 123):
    """
    Visualizes the 2D speciation process over time in a 3D plot.
    """
    torch.manual_seed(seed)
    T_max = times[0]
    plane_limit = 8
    res = 50 # Resolution for the density planes
    
    fig = plt.figure(figsize=(15, 20))
    ax = fig.add_subplot(111, projection='3d')

    ax.set_ylabel("Diffusion Time (t)", fontsize=14, labelpad=20)
    ax.set_xlabel("$x_1$", fontsize=10)
    ax.set_zlabel("$x_2$", fontsize=10)
    ax.set_ylim(max(0,-plane_limit), max(T_max,plane_limit))
    ax.set_xlim(-plane_limit, plane_limit)
    ax.set_zlim(-plane_limit, plane_limit)
    
    ax.invert_yaxis() # Invert Y so time flows downward from T_max to 0
    ax.set_box_aspect((1, 3, 1)) # Stretch the time axis
    ax.view_init(elev=30, azim=-120)

    # Grid for the density maps (t = T)
    _grid = np.linspace(-plane_limit, plane_limit, res) 
    xx, zz = np.meshgrid(_grid, _grid)
    
    density_T = np.exp(-0.5 * (xx**2 + zz**2) / (1 + std**2))
    ax.contourf(xx, density_T, zz, zdir='y', offset=T_max, cmap='Blues', alpha=0.3)

    if t_s:
        ax.plot([-plane_limit, plane_limit], [t_s, t_s], [-plane_limit, -plane_limit], 
                color='grey', lw=2, solid_capstyle='round', zorder=6,label=f'$t_s$')
        ax.text(-plane_limit-1.5, t_s, -plane_limit, f" $t_s$", color='black', fontweight='bold')

    if t_c:
        ax.plot([-plane_limit, plane_limit], [t_c, t_c], [-plane_limit, -plane_limit], 
                color='grey', lw=2, solid_capstyle='round', zorder=6,label=f'$t_c$')
        ax.text(-plane_limit-1.5, t_c, -plane_limit, f" $t_c$", color='black', fontweight='bold')

    x1_start = path_history[0, :, 0]
    x2_start = path_history[0, :, 1]
    ax.scatter(x2_start, np.full_like(x2_start, T_max), x1_start, 
               s=100, color='grey', alpha=0.6, label="Ancestral Noise")

    for i in range(nsamples):
        x1_path = path_history[:, i, 0]
        x2_path = path_history[:, i, 1]
        final_pos = torch.tensor(path_history[-1, i, :])
        class_val = classify(final_pos, mu_star)
        class_color = 'gold' if class_val > 0 else 'royalblue'
        ax.plot(x2_path, times, x1_path, color=class_color, lw=2, alpha=0.8)

    # Bimodal Gaussian Mixture density (t = 0)
    dist1 = (zz - mu_star[0].item())**2 + (xx - mu_star[1].item())**2
    dist2 = (zz + mu_star[0].item())**2 + (xx + mu_star[1].item())**2
    density_0 = np.exp(-0.5 * dist1 / std**2) + np.exp(-0.5 * dist2 / std**2)
    
    ax.contourf(xx, density_0, zz, zdir='y', offset=0, cmap='Reds', alpha=0.3)
    x1_final = path_history[-1, :, 0]
    x2_final = path_history[-1, :, 1]
    ax.scatter(x2_final, np.zeros_like(x2_final), x1_final, 
               s=150, color='black', marker='X', edgecolors='white', linewidth=1)

    ax.set_xticks([])
    ax.set_zticks([])
    plt.tight_layout()
    plt.show()


def visualize_interactive(W, colors=None):
    # 1. Convert NumPy weight matrix to NetworkX graph
    # (nx.from_numpy_array automatically handles the weights)
    G = nx.from_numpy_array(W)
    
    # 2. Initialize Pyvis Network
    # 'notebook=True' is essential for Jupyter
    net = Network(notebook=True, height="750px", width="100%", bgcolor="#222222", font_color="white", cdn_resources='remote')
    
    # 3. Import from NetworkX
    net.from_nx(G)
    
    # 4. Optional: Add color coding if you have your 'colors' list
    if colors is not None:
        for i, node in enumerate(net.nodes):
            node['color'] = colors[i]
            node['size'] = 10  # Make nodes visible
            
    # 5. Enable physics for Gephi-like movement
    # You can even show a UI to tweak physics in real-time
    net.show_buttons(filter_=['physics']) 
    
    return net.show("graph_vis.html")

def update(frame, ts_idx, embedding, sc, ax, milestones):
    # Ensure we are working with integers for frame comparison
    ts_idx_int = int(ts_idx)
    
    current_data = embedding[:frame+1]
    sc.set_offsets(current_data)
    sc.set_array(np.arange(frame + 1))
    
    # Start milestone
    if frame >= 0 and 'start' not in milestones:
        milestones['start'] = add_pointer_ani(embedding, ax, 0, 'Start (t=T)', 'red', (-60, 40))
    
    # Speciation milestone - Use >= to catch it even if the frame skip happens
    if frame >= ts_idx_int and 'ts' not in milestones:
        milestones['ts'] = add_pointer_ani(embedding, ax, ts_idx_int, 'Speciation ($t_s$)', 'darkcyan', (40, 40))
    
    # End milestone
    if frame >= len(embedding) - 1 and 'end' not in milestones:
        milestones['end'] = add_pointer_ani(embedding, ax, -1, 'End (T=0)', 'orange', (20, -40))
        
    return sc,

def add_pointer_ani(embedding, ax, index, label, color, offset=(40, 40)):
    return ax.annotate(label, 
                xy=(embedding[index, 0], embedding[index, 1]), 
                xytext=offset, 
                textcoords='offset points',
                arrowprops=dict(arrowstyle='->', color=color, lw=2),
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=color, alpha=0.8),
                fontsize=10, fontweight='bold', color=color)

    