import joblib
from joblib import Parallel, delayed
import numpy as np
import sys
import torch
from pathlib import Path
from tqdm import tqdm
from lib.ou_model import backward
from utils import (theoretical_bimodal_gaussian_ts, centers, knn_job, ctd_job)

def main(
        ds,
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
        laplacian
):
    for d in ds:
        print(f"Running diffusion for D={d}")
        mu_star = torch.ones(d) * mu
        std = 1.0
        t_s, ts_idx = theoretical_bimodal_gaussian_ts(mu_star, std, times)
        path = exp_path / f"D{d}_N{n_samples}_T{int(T)}"
        path.mkdir(parents=True, exist_ok=True)
        history_file = path / "history.pkl"
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
                "history": history,
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
            knn_results = Parallel(n_jobs=threads, backend="threading")(
                delayed(knn_job)(history[t], edges_to_inject, kernel)
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
            ctds = Parallel(n_jobs=threads)(
                delayed(ctd_job)(W_i, laplacian)
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

if __name__ == "__main__":
    sys.setrecursionlimit(10000)
    exp_path = Path("/extra/shared/groups/marinaivan/data_marina/recurrence_matrices/test_data")
    ds = [2, 50, 256, 1024, 16384]
    mu = 4
    n_samples_list = [1000, 3000, 6000, 10000]
    n_steps = 1000
    T = 10.0
    threads = 20
    laplacian = "unnormalized"
    kernel = "gaussian"
    data_model = 'bimodal_gaussian'
    dt = T / n_steps
    times = list(np.arange(T, 0, -dt))
    ts_indices = []
    for n_samples in n_samples_list:
        print('Number of samples: {}'.format(n_samples))
        for d in ds:
            mu_star = torch.ones(d) * mu
            _, ts_idx = theoretical_bimodal_gaussian_ts(mu_star, 1.0, times)
            ts_indices.append(ts_idx)
        max_ts_idx = max(ts_indices)
        coarse = list(range(max_ts_idx, len(times), 10))
        dense = list(range(0, max_ts_idx, 3))
        snap_time_indices = sorted(set(coarse + dense + ts_indices + [len(times) - 1]), reverse=True)
        main(
            ds=ds,
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
            )
