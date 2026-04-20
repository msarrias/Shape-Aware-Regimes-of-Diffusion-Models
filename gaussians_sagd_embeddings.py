import numpy as np
import joblib
import torch
from joblib import Parallel, delayed
from tqdm import tqdm
from ou_model import backward, theoretical_ts


from distances import CTD_matrix
from numpy import isclose
from stats import normalize, Kruglov_distance
from scipy.stats import wasserstein_distance
from adaptive_knn import AdaptiveKNNGraph # comes from graph-theory repo
from pathlib import Path
from SASNE.SASNE import SASNE
from scipy.spatial.distance import pdist, squareform


def ctd_job(
        w_result: np.ndarray,
        laplacian: str,
        normalization: str
):
    C_Gi = CTD_matrix(
        W=w_result,
        laplacian_type=laplacian
    )
    triu_i = C_Gi[np.triu_indices(w_result.shape[0], k=1)]
    
    norm_i = normalize(
        list_values=triu_i,
        norm_type=normalization,
        Vol=np.sum(w_result)
    )
    return {'ctds': triu_i, 'norm_ctds': norm_i}


def knn_job(
        data: np.ndarray,
        inject_edges: bool,
        kernel: str
):
    knn_obj = AdaptiveKNNGraph(
        data=data,
        inject_edges=inject_edges,
        kernel=kernel
    )
    w_matrix = knn_obj.compute_W()
    return knn_obj.k, w_matrix


if __name__ == "__main__":
    torch.manual_seed(9009)
    ds = [2, 50, 256, 1024, 4096, 16384]
    T = 10
    n_steps = 1000
    n_samples = 1000
    dt = T / n_steps
    times = np.arange(0, T, dt)

    n_threads = 10
    
    for d in ds:
        mu_star = torch.ones(d) * 4
        std = 1.0
        t_s, ts_idx = theoretical_ts(mu_star, std, times)
        time_indices = sorted(
            list(set(list(range(0, len(times), 10)) + [ts_idx, len(times) - 1])),
            reverse=True
        )
        path = Path(f"data/exp_05/D{d}_N1000_100ts/")
        path.mkdir(parents=True, exist_ok=True)
        history_file = path / f"D{d}_N1000.jbl"
        
        #  SDE Simulation
        if not history_file.exists():
            history = {}
            #  sample for T
            x_current = torch.randn(d, n_samples)  # d x N
            history[T] = x_current.T.clone().numpy()  # N x d
            #  T-1 --> 0
            
            for step in tqdm(reversed(range(n_steps)), total=n_steps, desc=f"SDE D={d}"):
                t = times[step]
                x_current, _ = backward(x_current, t, dt, mu_star, std)
                if step in time_indices:
                    history[t] = x_current.T.clone().numpy() 
    
            time_snaps = list(history.keys())
            data_to_dump = {
                "history": history,
                "params": {
                    "dim": d,
                    "n_samples": n_samples,
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
            time_snaps = list(history.keys())

        # Graph Construction
        ws_file = path / f"D{d}_N1000_Ws.jbl"
        
        if not ws_file.exists():
            w_results, k_results = [], []
            inject_edges_bool = True
            kernel_choice = "gaussian"

            knn_results = Parallel(n_jobs=n_threads)(
                delayed(knn_job)(
                    history[t], inject_edges_bool, kernel_choice)
                for t in tqdm(time_snaps, desc="KNN Progress")
            )
            
            for t, knn_obj in zip(time_snaps, knn_results):
                k, W = knn_obj
                w_results.append(W)
                k_results.append(k)
            
            joblib.dump(
                {"Ws": w_results,
                 "ks": k_results,
                 'ts': time_snaps
                 },
                ws_file, compress=3
            )
        else:
            print(f"[D={d}] Weights found. Loading...")
            load_file = joblib.load(ws_file)
            w_results = load_file["Ws"]
            time_snaps = load_file['ts']

        # CTD Calculation
        ctd_file = path / f"D{d}_N1000_CTDs.jbl"
        if not ctd_file.exists():
            laplacian_type = "unnormalized"
            norm_type = "norm_wrt_avg_ctd"

            ctds = Parallel(n_jobs=n_threads)(
                delayed(ctd_job)(W_i, laplacian_type, norm_type)
                for W_i in tqdm(w_results, total=len(w_results), desc="CTD Logic")
            )
            ctds_dict = {t: ctd for t, ctd in zip(time_snaps, ctds)}
            ctd_dump = {
                "CTDs": ctds_dict,
                "params": {
                    "laplacian_type": laplacian_type,
                    "norm_type": norm_type,
                    "ts": time_snaps
                }
            }
            
            joblib.dump(ctd_dump, ctd_file, compress=3)
        else:
            print(f"[D={d}] CTDs found. Loading...")
            ctds_dict = joblib.load(ctd_file)["CTDs"]
    
        # SAGD Distance Matrix
        sagd_file = path / f"D{d}_N1000_SAGD.jbl"
        
        if not sagd_file.exists():
            norm_ctds_list = [t_ctds_dict['norm_ctds'] for t, t_ctds_dict in ctds_dict.items()]
            num_graphs = len(norm_ctds_list)
            pairs = [(i, j) for i in range(num_graphs) for j in range(i + 1, num_graphs)]

            test_size = min(3, num_graphs)
            for i in range(test_size):
                for j in range(test_size):
                    kruglov_dist = Kruglov_distance(norm_ctds_list[i], norm_ctds_list[j])
                    wasserstein_dist = wasserstein_distance(norm_ctds_list[i], norm_ctds_list[j]) 
                    assert isclose(kruglov_dist, wasserstein_dist)

            distances = Parallel(n_jobs=n_threads)(
                delayed(wasserstein_distance)(norm_ctds_list[i], norm_ctds_list[j])
                for i, j in tqdm(pairs, desc="Computing SAGD Matrix")
            )

            sagd_dist_matrix = np.zeros((num_graphs, num_graphs))
            for (i, j), dist in zip(pairs, distances):
                sagd_dist_matrix[i, j] = dist
                sagd_dist_matrix[j, i] = dist
                    
            joblib.dump(sagd_dist_matrix, sagd_file, compress=3)
        else:
            print(f"[D={d}] SAGD matrix found. Loading...")
            sagd_dist_matrix = joblib.load(sagd_file)


        # SASNE embedding
        sasne_file = path / f"D{d}_N1000_SASNE.jbl"

        if not sasne_file.exists():
            embedding, Z = SASNE(sagd_dist_matrix)
            D1 = squareform(pdist(embedding))
            D2 = squareform(pdist(Z))

            sasne_dump = {
                'embedding': embedding,
                'Z': Z,
                'D1': D1,
                'D2': D2,
            }
            joblib.dump(sasne_dump, sasne_file, compress=3)
    print('Done!')
