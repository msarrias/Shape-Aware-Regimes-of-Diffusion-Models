from matplotlib import rc
from matplotlib.animation import FuncAnimation
import numpy as np
import torch
import os
import joblib
from pathlib import Path
from scipy.spatial.distance import pdist, squareform

from plotting import plot_full_sasne_dashboard
from SASNE.SASNE import SASNE

if __name__ == "__main__":
    sasne_embeddings_list = []
    sagd_dist_matrices_list = []
    ts_list = []
    time_snaps = []
    d_list = [2, 50, 256, 1024, 4096, 16384]
    path = Path('data/exp_04')
    for d in d_list:
        print(f'd = {d}')
        loaded_data = joblib.load(path / f"D{d}_N1000_100ts/D{d}_N1000.jbl")
        times = loaded_data['params']['times']
        time_snaps = loaded_data['params']['times_snapshots']
        t_s = loaded_data['params']['ts']
        ts_idx = np.argmin(np.abs(time_snaps - t_s))

        SAGD_dist_matrix = joblib.load(path / f"D{d}_N1000_100ts/D{d}_N1000_SAGD.jbl")

        sasne_file = path / f"D{d}_N1000_100ts/D{d}_N1000_SASNE.jbl"
        if sasne_file.exists():
            sasne_embedding = joblib.load(sasne_file)
            embedding = sasne_embedding['embedding']
            Z = sasne_embedding['Z']
            D1 = sasne_embedding['D1']
            D2 = sasne_embedding['D2']
        else:
            embedding, Z = SASNE(SAGD_dist_matrix)
            D1 = squareform(pdist(embedding))
            D2 = squareform(pdist(Z))

        sasne_embeddings_list.append(
            (SAGD_dist_matrix, embedding, Z, D1, D2, time_snaps, ts_idx, t_s)
        )
    if time_snaps:
        plot_full_sasne_dashboard(
            sasne_results=sasne_embeddings_list,
            d_list=d_list,
            time_snaps_vector=time_snaps,
            save_path=Path(path / f"dimension_sweep_results.png"))

    # for result, d in zip(sasne_embeddings_list, d_list):
    #     SAGD_dist_matrix, embedding, Z, D1, D2, time_snaps, ts_idx, t_s = result
    #     plot_sagd_heatmap(
    #         W=SAGD_dist_matrix,
    #         time_vector=time_snaps,
    #         ts=t_s,
    #         ts_idx=ts_idx,
    #         d=d,
    #         save_fig_path=Path(path / f"D{d}_N1000_100ts/D{d}_heatmap_sagd_matrix.png")
    #     )
    #     save_path = Path(path / f"D{d}_N1000_100ts/diffusion_d{d}.gif")
    #     if not save_path.exists():
    #         create_cont_sasne_animation(
    #             embedding=embedding,
    #             time_snaps=time_snaps,
    #             d=d,
    #             ts_idx=ts_idx,
    #             save_path=save_path
    #         )
    #         print(f"d={d}, Animation saved")
