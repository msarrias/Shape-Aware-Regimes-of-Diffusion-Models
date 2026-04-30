import numpy as np
import joblib
import torch
import sys
import argparse
from joblib import Parallel, delayed
from tqdm import tqdm
from pathlib import Path
from scipy.stats import wasserstein_distance
from scipy.spatial.distance import pdist, squareform
from numpy import isclose
import logging

from ou_model import backward, theoretical_ts
from distances import CTD_matrix
from stats import normalize, Kruglov_distance
from SASNE.adaptive_knn import AdaptiveKNNGraph
from SASNE.SASNE import SASNE


def setup_logging(exp_path, args):
    log_file = exp_path / "settings.log"

    logging.basicConfig(
        level=logging.INFO,
        format='%(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )
    logger = logging.getLogger()
    logger.info("=== Simulation Settings ===")
    for arg, value in vars(args).items():
        logger.info(f"{arg}: {value}")
    logger.info("===========================\n")
    return logger


def ctd_job(
        w_result: np.ndarray,
        laplacian: str,
        normalization: str,
        volume
):
    C_Gi = CTD_matrix(
        W=w_result,
        laplacian_type=laplacian,
    )
    triu_i = C_Gi[np.triu_indices(w_result.shape[0], k=1)]
    norm_i = normalize(
        list_values=triu_i,
        norm_type=normalization,
        Vol=volume
    )
    return {'ctds': triu_i, 'norm_ctds': norm_i}


def knn_job(
        data: np.ndarray,
        inject_edges: bool,
        kernel: str,
        sigma: float=None
):
    knn_obj = AdaptiveKNNGraph(
        data=data,
        inject_edges=inject_edges,
        kernel=kernel
    )
    w_matrix = knn_obj.compute_W(sigma=sigma)
    return knn_obj.k, (knn_obj.sigma if kernel == 'gaussian' else None), w_matrix


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--seed", type=int, default=123456)
    parser.add_argument("--n_samples", type=int, default=1000)
    parser.add_argument("--n_steps", type=int, default=1000)
    parser.add_argument("--T", type=float, default=10.0)
    parser.add_argument("--mu", type=float, default=4.0)

    parser.add_argument("--kernel", type=str, default="gaussian",
                        choices=["gaussian", "inverse_sq_euclidean_d"])
    parser.add_argument("--laplacian", type=str, default="unnormalized")
    parser.add_argument("--norm_type", type=str, default="norm_wrt_volume",
                        choices=["norm_wrt_volume", "norm_wrt_avg_ctd", "scale_and_shift", "rank_norm"])
    parser.add_argument("--inject_edges", action="store_true", default=True)

    parser.add_argument("--threads", type=int, default=20)
    parser.add_argument("--exp_name", type=str, default="exp_04")
    parser.add_argument("--ds", type=int, nargs="+", help="Explicit list of dimensions to run")

    args = parser.parse_args()

    sys.setrecursionlimit(2000)
    torch.manual_seed(args.seed)

    ds = args.ds
    dt = args.T / args.n_steps
    times = np.arange(0, args.T, dt)
    exp_path = Path(f'data/{args.exp_name}')
    exp_path.mkdir(parents=True, exist_ok=True)
    logger = setup_logging(exp_path, args)

    # We want every dimension to share the same snapshots,
    # including all theoretical ts points
    ts_indices = []
    for d in ds:
        mu_star = torch.ones(d) * args.mu
        _, ts_idx = theoretical_ts(mu_star, 1.0, times)
        ts_indices.append(ts_idx)

    snap_time_indices = sorted(
        list(set(list(range(0, len(times), 10)) + ts_indices + [len(times) - 1])),
        reverse=True
    )

    for d in ds:
        mu_star = torch.ones(d) * args.mu
        std = 1.0
        t_s, _ = theoretical_ts(mu_star, std, times)

        path = exp_path / f"D{d}_N{args.n_samples}_T{int(args.T)}/"
        path.mkdir(parents=True, exist_ok=True)
        history_file = path / f"history.jbl"

        # SDE Simulation
        if not history_file.exists():
            history = {}
            x_current = torch.randn(d, args.n_samples)  # d x N
            history[args.T] = x_current.T.clone().numpy()  # N x d

            for step in tqdm(reversed(range(args.n_steps)), total=args.n_steps, desc=f"[D={d}] SDE"):
                t = times[step]
                x_current, _ = backward(x_t=x_current, t=t, dt=dt, mu_star=mu_star, std=std)
                if step in snap_time_indices:
                    history[t] = x_current.T.clone().numpy()
            time_snaps = list(history.keys())
            data_to_dump = {
                "history": history,
                "params": {
                    **vars(args),
                    "d": d,
                    "ts_theoretical": t_s,
                    "times_snapshots": time_snaps,
                    "times": times
                }
            }
            joblib.dump(data_to_dump, history_file, compress=3)
        else:
            logger.info(f"[D={d}] History found. Loading...")
            history = joblib.load(history_file)["history"]
            time_snaps = list(history.keys())

        # Graph Construction
        ws_file = path / "Ws.jbl"
        if not ws_file.exists():
            knn_results = Parallel(n_jobs=args.threads, backend="threading")(
                delayed(knn_job)(history[t], args.inject_edges, args.kernel)
                for t in tqdm(time_snaps, desc=f"[D={d}] KNN Progress")
            )

            w_results = [w for *_, w in knn_results]
            k_results = [k for k, *_ in knn_results]

            results_dict = {
                "Ws": w_results,
                "ks": k_results,
                "ts": time_snaps,
                "kernel": args.kernel
            }

            if kernel == "gaussian":
                sigma_results = [sigma for _, sigma, _ in knn_results]
                results_dict["sigma"] = sigma_results

            joblib.dump(results_dict, ws_file, compress=3)
        else:
            logger.info(f"[D={d}] Ws found. Loading...")
            load_ws = joblib.load(ws_file)
            w_results = load_ws["Ws"]
            time_snaps = load_ws['ts']

        # CTD Calculation
        ctd_file = path / "CTDs.jbl"
        if not ctd_file.exists():
            if args.norm_type == "norm_wrt_volume":
                vol = np.sum(w_results[-1])
            else:
                vol = None

            ctds = Parallel(n_jobs=args.threads)(
                delayed(ctd_job)(W_i, args.laplacian, args.norm_type, vol)
                for W_i in tqdm(w_results, total=len(w_results), desc=f"[D={d}] CTD Logic")
            )
            ctds_dict = {t: ctd for t, ctd in zip(time_snaps, ctds)}
            joblib.dump({
                "CTDs": ctds_dict,
                "params": {
                    "laplacian": args.laplacian,
                    "normalization": args.norm_type,
                    "ts": time_snaps
                }
            }, ctd_file, compress=3)
        else:
            logger.info(f"[D={d}] CTDs found. Loading...")
            ctds_dict = joblib.load(ctd_file)["CTDs"]

        # SAGD Distance Matrix
        sagd_file = path / "SAGD.jbl"
        if not sagd_file.exists():
            norm_ctds = [ctds_dict[t]['norm_ctds'] for t in time_snaps]
            num_graphs = len(norm_ctds)
            pairs = [(i, j) for i in range(num_graphs) for j in range(i + 1, num_graphs)]

            distances = Parallel(n_jobs=args.threads)(
                delayed(wasserstein_distance)(norm_ctds[i], norm_ctds[j])
                for i, j in tqdm(pairs, desc=f"[D={d}] SAGD Matrix")
            )

            sagd_dist_matrix = np.zeros((num_graphs, num_graphs))
            for (i, j), dist in zip(pairs, distances):
                sagd_dist_matrix[i, j] = sagd_dist_matrix[j, i] = dist
            joblib.dump(sagd_dist_matrix, sagd_file, compress=3)
        else:
            sagd_dist_matrix = joblib.load(sagd_file)

        # SASNE embedding
        sasne_file = path / "SASNE.jbl"
        if not sasne_file.exists():
            embedding, Z = SASNE(data=sagd_dist_matrix)
            joblib.dump({
                "embedding": embedding,
                "Z": Z,
                "D1": squareform(pdist(embedding)),
                "D2": squareform(pdist(Z))
            }, sasne_file, compress=3)
    logger.info('Done!')

if __name__ == "__main__":
    main()