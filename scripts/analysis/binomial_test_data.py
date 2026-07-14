import joblib
import numpy as np
import sys
import torch
from pathlib import Path
from tqdm import tqdm

from lib.ou_model import backward
from lib.stats import normalize
from scipy.stats import wasserstein_distance
from lib.utils import (theoretical_bimodal_gaussian_ts, centers, knn_job, ctd_job, fetch_pairs, get_snap_times)

def main(
        dims,
        mu,
        times,
        n_samples,
        T,
        n_steps,
        data_model,
        kernel,
        exp_path,
        dt,
        snap_time_indices,
        threads,
        laplacian,
        clip_perc,
        log_transform,
        clipping,
        norm_type,
):
    for d in dims:
        print(f"Running diffusion for D={d}")
        mu_star = torch.ones(d) * mu
        std = 1.0
        t_s, ts_idx = theoretical_bimodal_gaussian_ts(mu_star, std, times)
        path = exp_path / f"D{d}_N{n_samples}_T{int(T)}"
        path.mkdir(parents=True, exist_ok=True)
        history_file = path / "history.jbl"
        if not history_file.exists():
            history = {}
            x_current = torch.randn(d, n_samples)
            history[T] = x_current.T.clone().numpy()
            for step in tqdm(range(n_steps), total=n_steps, desc=f"[D={d}] SDE"):
                t = times[step]
                x_current, _ = backward(
                    x_t=x_current,
                    t=t,
                    dt=dt,
                    mu_star=mu_star,
                    std=std,
                    model=data_model,
                    weights=None,
                )
                if step in snap_time_indices:
                    history[t] = x_current.T.clone().numpy()
            data_to_dump = {
                # "history": history,
                "params": {
                    "dim": d,
                    "times_snapshots": list(history.keys()),
                    "times": times,
                    "ts_theoretical": (t_s, ts_idx),
                },
            }
            joblib.dump(data_to_dump, history_file, compress=3)
        else:
            history = joblib.load(history_file)["history"]
        edges_to_inject = np.random.permutation(n_samples)[:int(n_samples * 0.02)].reshape(-1, 2)
        # 1. Graph construction
        ws_file = path / "Ws.jbl"
        time_snaps = list(history.keys())
        if not ws_file.exists():
            knn_results = joblib.Parallel(n_jobs=threads, backend="threading")(
                joblib.delayed(knn_job)(history[t], edges_to_inject, kernel)
                for t in tqdm(time_snaps, desc="KNN Progress")
            )
            w_results = [w for *_, w in knn_results]
            k_results = [k for k, *_ in knn_results]

            results_dict = {
                "Ws": w_results,
                "ks": k_results,
                "ts": time_snaps,
                "kernel": kernel,
            }
            if kernel == "gaussian":
                results_dict["sigma"] = [sigma for _, sigma, _ in knn_results]

            joblib.dump(results_dict, ws_file, compress=3)
        else:
            w_results = joblib.load(ws_file)["Ws"]
        # 2. CTD matrix
        ctd_file = path / "CTDs.jbl"
        if not ctd_file.exists():
            ctds = joblib.Parallel(n_jobs=threads)(
                joblib.delayed(ctd_job)(W_i, laplacian)
                for W_i in tqdm(w_results, total=len(w_results), desc="CTD Logic")
            )
            ctds_dict = {t: triu_i for t, triu_i in zip(time_snaps, ctds)}
            joblib.dump(
                {
                    "CTDs": ctds_dict,
                    "params": {
                        "laplacian": laplacian,
                        "ts": time_snaps,
                    },
                },
                ctd_file,
                compress=3,
            )
        else:
            ctds_dict = joblib.load(ctd_file)["CTDs"]
        for transform in log_transform:
            for clip in clipping:
                for norm in norm_type:
                    is_clipped = 'clipped' if clip else ''
                    is_transformed = 'log_transformed' if transform else ''
                    sagd_file = path / f"SAGD_{norm}_{is_clipped}_{is_transformed}.jbl"
                    if not sagd_file.exists():
                        pairs = fetch_pairs(num_graphs=len(time_snaps))
                        ctds_list = [np.asarray(value) for value in ctds_dict.values()]
                        if transform:
                            ctds_list = [np.log1p(value) for value in ctds_list]
                        if clip:
                            ctds_list = [
                                np.clip(
                                    value, np.percentile(value, 100 - clip_perc), np.percentile(value, clip_perc)
                                )
                                for value in ctds_list
                            ]
                        norm_ctds = [
                            normalize(list_values=value, norm_type=norm)
                            for value in ctds_list
                        ]
                        num_graphs = len(norm_ctds)
                        distances = joblib.Parallel(n_jobs=threads)(
                            joblib.delayed(wasserstein_distance)(norm_ctds[i], norm_ctds[j])
                            for i, j in tqdm(pairs, desc="SAGD Matrix")
                        )
                        sagd_dist_matrix = np.zeros((num_graphs, num_graphs))
                        for (i, j), dist in zip(pairs, distances):
                            sagd_dist_matrix[i, j] = sagd_dist_matrix[j, i] = dist
                        joblib.dump(sagd_dist_matrix, sagd_file, compress=3)

if __name__ == "__main__":
    sys.setrecursionlimit(10000)
    normalization = ["shift_center", "mean", "volume"]
    exp_path = Path("/extra/shared/groups/marinaivan/data_marina/recurrence_matrices/test_data")
    ds = [2, 1024, 16384] # 50, 256,
    mu = 4
    n_samples_list = [1000]#, 3000, 6000, 10000]
    n_steps = 1000
    T = 10.0
    threads = 20
    laplacian = "unnormalized"
    kernel = "gaussian"
    data_model = 'bimodal_gaussian'
    norm = ['scale_and_shift', 'norm_wrt_avg_ctd']
    clip_perc = 95
    dt = T / n_steps
    times = np.linspace(T, dt, n_steps).tolist()
    snap_time_indices = get_snap_times(data_model=data_model, mu=mu, times=times, ds=ds)
    for n_samples in n_samples_list:
        main(
            dims=ds,
            mu=mu,
            times=times,
            n_samples=n_samples,
            T=T,
            n_steps=n_steps,
            data_model=data_model,
            kernel=kernel,
            exp_path=exp_path,
            dt=dt,
            snap_time_indices=snap_time_indices,
            threads=threads,
            laplacian=laplacian,
            clip_perc=95,
            log_transform=[True, False],
            clipping=[True, False],
            norm_type=norm
            )

