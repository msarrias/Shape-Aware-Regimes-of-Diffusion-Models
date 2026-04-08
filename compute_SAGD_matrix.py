import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
import torch
import sys
import os
import joblib
from tqdm import tqdm
import networkx as nx
from sklearn.neighbors import kneighbors_graph


from ou_model import forward, backward, score, classify
from plotting import plot_speciation_3d
from distance_matrix import construct_sagd_distance_matrix

from distances import CTD_matrix
from stats import normalize, Kruglov_distance
from SAGD import SAGD
from SASNE import SASNE
from construct_graph import construct_graph # this comes from the SASNE repo
from adaptive_knn import AdaptiveKNNGraph # comes from graph-theory repo

if __name__ == "__main__":
    torch.manual_seed(123)
    d = 2            # Dimension
    nsamples_small_n = 6     # Number of trajectories
    T_max = 10.0
    dt = 0.01
    mu_star, std = torch.tensor([4.0, 4.0]), 1.0
    times = np.arange(T_max, 0, -dt)
    Lambda = np.linalg.norm(mu_star)**2 + std**2
    t_s = np.log(Lambda) / 2
    t_s_idx = next(idx for idx, t in enumerate(times) if round(t, 2) == round(t_s, 2))
    print(t_s)
    time_indices = set(range(0, len(times), 10))
    time_indices.add(t_s_idx)
    diffusion_steps_idices = sorted(list(time_indices))
    
    loaded_data = joblib.load("data/D2_N1000.jbl")
    history = loaded_data["history"]
    dt = loaded_data["params"]["dt"]
    
    W_results = joblib.load("data/D2_N1000_Ws.jbl")
    CTDs_results = joblib.load("data/D2_N1000_CTDs.jbl")
    ctds_dict = CTDs_results["CTDs"]
    norm_ctds_list = [t_dict['norm_ctds'] for t_idx, t_dict in ctds_dict.items()]
    num_graphs = len(norm_ctds_list)
    sagd_dist_matrix = np.zeros((num_graphs, num_graphs))
    
    for i in tqdm(range(num_graphs), desc="Computing SAGD Matrix"):
        norm_i = norm_ctds_list[i]
        for j in range(i + 1, num_graphs):
            norm_j = norm_ctds_list[j]
            dist = Kruglov_distance(vi=norm_i, vj=norm_j)
            sagd_dist_matrix[i, j] = dist
            sagd_dist_matrix[j, i] = dist # symmetry
    joblib.dump(sagd_dist_matrix, "data/D2_N1000_SAGD.jbl", compress=3)