import matplotlib.pyplot as plt
import numpy as np
import torch
import sys
import os
import joblib
from tqdm import tqdm
from pathlib import Path
import networkx as nx
from scipy.spatial.distance import pdist, squareform

from ou_model import forward, backward, score, classify, theoretical_ts
from plotting import plot_speciation_3d, plot_sagd_heatmap_row
from SASNE import SASNE
from SASNE.RRP import RRP

from SASNE.adaptive_knn import AdaptiveKNNGraph
from plotting import plot_sagd_heatmap_with_prob, plot_sagd_heatmap_row_with_prob
from animations import create_ctd_synchronized_animation

ds = [2, 16, 256, 1024]

path = Path('data/exp_01')

W_list = []
time_snaps_vector_list = []
ts_tuple_list = []
tsagd_tuple_list = []
std = 1.0
mu = None

for d_i in ds:
    ld = joblib.load(path / f"D{d_i}_N1000_T10/history.jbl")
    clustering = joblib.load(path / f"D{d_i}_N1000_T10/clusters.jbl")
    W_list.append(joblib.load(path / f"D{d_i}_N1000_T10/SAGD.jbl"))
    snaps = ld['params']['times_snapshots']
    t_s_i = ld['params']['ts_theoretical']
    mu = ld['params']['mu']
    time_snaps_vector_list.append(snaps)
    t_sagd = clustering["speciation"]
    t_sagd_idx = clustering["speciation_idx"]
    ts_tuple_list.append((t_s_i, int(np.argmin(np.abs(snaps - t_s_i)))))
    tsagd_tuple_list.append((t_sagd, t_sagd_idx))

plot_sagd_heatmap_row_with_prob(
    W_list=W_list,
    d_list=ds,
    time_snaps_vector_list=time_snaps_vector_list,
    ts_tuple_list=ts_tuple_list,
    mu=mu,
    std=std,
    tsagd_tuple_list=tsagd_tuple_list,
    save_fig_path='figures/sagd_heatmap_row_with_prob.png',
)

plot_sagd_heatmap_row(
    W_list=W_list,
    d_list=ds,
    time_snaps_vector=time_snaps_vector_list[0],
    ts_tuple_list=ts_tuple_list,
    save_fig_path='figures/sagd_heatmap_row.png',
)
