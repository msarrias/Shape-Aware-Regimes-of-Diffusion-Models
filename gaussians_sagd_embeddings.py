import numpy as np
import joblib
import torch
from joblib import Parallel, delayed
from tqdm import tqdm
from ou_model import forward, backward, score, classify, theoretical_ts
from plotting import plot_speciation_3d
from distance_matrix import construct_sagd_distance_matrix


from distances import CTD_matrix
from stats import normalize, Kruglov_distance
from SAGD import SAGD
from adaptive_knn import AdaptiveKNNGraph # comes from graph-theory repo
from pathlib import Path

if __name__ == "__main__":
    DEVICE = 'cpu'
    torch.manual_seed(232)
    np.random.seed(232)
    
    ds = [2, 256, 1024, 4096, 16384]
    T = 10
    nsteps = 1000
    nsamples = 1000
    dt = T / nsteps
    times = np.arange(0, T, dt)
    time_indices = list(range(0, len(times), 10)) 
    
    for d in ds:
        mu_star, std = torch.ones(d, device=DEVICE), 1
        t_s, ts_idx = theoretical_ts(mu_star, std, times)
        time_indices = sorted(list(set(time_indices + [ts_idx, len(times) - 1, 0])))
        path = Path(f"data/D{d}_N1000_100ts/")
        path.mkdir(parents=True, exist_ok=True)
        history_file = path / f"D{d}_N1000.jbl"
        
        if not history_file.exists():
            
            # sample for T
            x_current = torch.randn(d, nsamples) 
            history = {}
            # SDE Simulation
            for step in tqdm(reversed(range(nsteps)), total=nsteps, desc=f"SDE D={d}"):
                t = times[step]
                if step in time_indices:
                    history[t] = x_current.T.clone().numpy() 
                x_current, _ = backward(x_current, t, dt, mu_star, std)
            # history = np.array(history)
            time_snaps = list(history.keys())
            data_to_dump = {
                "history": history,
                "params": {
                    "dim": d,
                    "nsamples": nsamples,
                    "T_max": T,
                    "dt": dt,
                    "times": times,
                    "times_snapshots": time_snaps,
                    "mu_star": mu_star.numpy(),
                    "std": std,
                    "ts": t_s,
                    "ts_idx":ts_idx
                }
            }
            joblib.dump(data_to_dump, history_file, compress=3)
        else:
            print(f"[D={d}] History found. Loading...")
            history = joblib.load(history_file)["history"]

        ws_file = path / f"D{d}_N1000_Ws.jbl"
        if not ws_file.exists():
            # Graph Construction
            W_results = []
            k_results = []
            for _, graph in tqdm(history.items(), desc="KNN Progress"):
                model = AdaptiveKNNGraph(graph)
                W = model.compute_W()
                W_results.append(W)
                k_results.append(model.k)
            time_snaps = list(history.keys())
            joblib.dump({"Ws": W_results, "ks": k_results, 'ts': time_snaps}, ws_file, compress=3)
        else:
            print(f"[D={d}] Weights found. Loading...")
            load_file = joblib.load(ws_file)
            W_results = load_file["Ws"]
            time_snaps = load_file['ts']

        ctd_file = path / f"D{d}_N1000_CTDs.jbl"
        if not ctd_file.exists():
            # CTD Calculation
            ctds_dict = {}
            laplacian_type = "unnormalized"
            norm_type = "norm_wrt_avg_ctd"
            
            for W_i, t in tqdm(zip(W_results, time_snaps), total=len(time_snaps), desc="CTD Logic"):
                C_Gi = CTD_matrix(W=W_i, laplacian_type=laplacian_type)
                triu_i = C_Gi[np.triu_indices(W_i.shape[0], k=1)]
                
                norm_i = normalize(
                    list_values=triu_i,
                    norm_type=norm_type,
                    Vol=np.sum(W_i)
                )
                ctds_dict[t] = {'ctds': triu_i, 'norm_ctds': norm_i}
    
            ctd_dump = {
                "CTDs": ctds_dict,
                "params": {
                    "laplacian_type": laplacian_type,
                    "norm_type": norm_type,
                    "ts": list(ctds_dict.keys())
                }
            }
            joblib.dump(ctd_dump, ctd_file, compress=3)
        else:
            print(f"[D={d}] CTDs found. Loading...")
            ctds_dict = joblib.load(ctd_file)["CTDs"]
    
        # SAGD Distance Matrix
        sagd_file = path / f"D{d}_N1000_SAGD.jbl"
        if not sagd_file.exists():
            norm_ctds_list = [dict_['norm_ctds'] for k, dict_ in ctds_dict.items()]
            num_graphs = len(norm_ctds_list)
            sagd_dist_matrix = np.zeros((num_graphs, num_graphs))
            
            for i in tqdm(range(num_graphs), desc="Computing SAGD Matrix"):
                norm_i = norm_ctds_list[i]
                for j in range(i + 1, num_graphs):
                    norm_j = norm_ctds_list[j]
                    dist = Kruglov_distance(vi=norm_i, vj=norm_j)
                    sagd_dist_matrix[i, j] = dist
                    sagd_dist_matrix[j, i] = dist 
                    
            joblib.dump(sagd_dist_matrix, sagd_file, compress=3)
            