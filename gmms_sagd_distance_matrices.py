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
from clustering import cluster_distance_matrix
from distances import CTD_matrix
from stats import normalize, Kruglov_distance
from adaptive_knn import AdaptiveKNNGraph


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
        laplacian: str
):
    C_Gi = CTD_matrix(
        W=w_result,
        laplacian_type=laplacian
    )
    triu_i = C_Gi[np.triu_indices(w_result.shape[0], k=1)]

    return triu_i

def knn_job(
        data: np.ndarray,
        edges_to_inject: list,
        kernel: str,
        sigma: float=None
):
    knn_obj = AdaptiveKNNGraph(
        data=data,
        edges_to_inject=edges_to_inject,
        kernel=kernel
    )
    w_matrix = knn_obj.compute_W(sigma=sigma)
    return knn_obj.k, (knn_obj.sigma if kernel == 'gaussian' else None), w_matrix

def diffuse_job(
        history_file: Path,
        dim: int,
        args: argparse.Namespace,
        times: list,
        dt: float,
        snap_time_indices: list,
        mu_star: float,
        std: float,
        ts_theoretical: float,
        logger: logging.Logger,
):
    if not history_file.exists():
        history = {}
        x_current = torch.randn(dim, args.n_samples)  # d x N
        history[args.T] = x_current.T.clone().numpy()  # N x d

        for step in tqdm(reversed(range(args.n_steps)), total=args.n_steps, desc=f"[D={dim}] SDE"):
            t = times[step]
            x_current, _ = backward(x_t=x_current, t=t, dt=dt, mu_star=mu_star, std=std)
            if step in snap_time_indices:
                history[t] = x_current.T.clone().numpy()
        data_to_dump = {
            "history": history,
            "params": {
                **vars(args),
                "dim": dim,
                "ts_theoretical": ts_theoretical,
                "times_snapshots": list(history.keys()),
                "times": times
            }
        }
        joblib.dump(data_to_dump, history_file, compress=3)
    else:
        logger.info(f"[D={dim}] History found. Loading...")
        history = joblib.load(history_file)["history"]

    return history

def construct_graph_job(
        dim:int,
        ws_file: Path,
        args: argparse.Namespace,
        history: dict,
        edges_to_inject: list,
        logger: logging.Logger,
):
    time_snaps = list(history.keys())
    if not ws_file.exists():
        knn_results = Parallel(n_jobs=args.threads, backend="threading")(
            delayed(knn_job)(history[t], edges_to_inject, args.kernel)
            for t in tqdm(time_snaps, desc=f"[D={dim}] KNN Progress")
        )
        w_results = [w for *_, w in knn_results]
        k_results = [k for k, *_ in knn_results]

        results_dict = {
            "Ws": w_results,
            "ks": k_results,
            "ts": time_snaps,
            "kernel": args.kernel
        }

        if args.kernel == "gaussian":
            sigma_results = [sigma for _, sigma, _ in knn_results]
            results_dict["sigma"] = sigma_results

        joblib.dump(results_dict, ws_file, compress=3)
    else:
        logger.info(f"[D={dim}] Ws found. Loading...")
        load_ws = joblib.load(ws_file)
        w_results = load_ws["Ws"]
    return w_results


def ctds_job(
        ctd_file: Path,
        args: argparse.Namespace,
        w_results: dict,
        time_snaps: list,
        dim: int,
        logger: logging.Logger,
):
    if not ctd_file.exists():
        ctds = Parallel(n_jobs=args.threads)(
            delayed(ctd_job)(W_i, args.laplacian)
            for W_i in tqdm(w_results, total=len(w_results), desc=f"[D={dim}] CTD Logic")
        )

        if args.norm_type == "log_global_scale_and_shift":

            all_log_ctds = np.concatenate([np.log1p(triu_i) for triu_i in ctds])
            g_min = np.percentile(all_log_ctds, 1)
            g_max = np.percentile(all_log_ctds, 95)
            range_ = g_max - g_min

            ctds_dict = {
                t: {
                    'ctds': triu_i,
                    'norm_ctds': (np.clip(np.log1p(np.array(triu_i)), g_min, g_max) - g_min) / range_
                }
                for t, triu_i in zip(time_snaps, ctds)
            }
        else:
            ctds_dict = {
                t: {'ctds': triu_i, 'norm_ctds': normalize(triu_i, args.norm_type)}
                for t, triu_i in zip(time_snaps, ctds)
            }

        joblib.dump({
            "CTDs": ctds_dict,
            "params": {
                "laplacian": args.laplacian,
                "normalization": args.norm_type,
                "ts": time_snaps
            }
        }, ctd_file, compress=3)

    else:
        logger.info(f"[D={dim}] CTDs found. Loading...")
        ctds_dict = joblib.load(ctd_file)["CTDs"]
    return ctds_dict


def sagd_job(
        sagd_file:Path,
        ctds_dict: dict,
        time_snaps: list,
        dim: int,
        args: argparse.Namespace,
        logger: logging.Logger
):
    sagd_dist_matrix = None
    if not sagd_file.exists():
        norm_ctds = [ctds_dict[t]['norm_ctds'] for t in time_snaps]
        num_graphs = len(norm_ctds)
        pairs = [(i, j) for i in range(num_graphs) for j in range(i + 1, num_graphs)]

        distances = Parallel(n_jobs=args.threads)(
            delayed(wasserstein_distance)(norm_ctds[i], norm_ctds[j])
            for i, j in tqdm(pairs, desc=f"[D={dim}] SAGD Matrix")
        )

        sagd_dist_matrix = np.zeros((num_graphs, num_graphs))
        for (i, j), dist in zip(pairs, distances):
            sagd_dist_matrix[i, j] = sagd_dist_matrix[j, i] = dist
        joblib.dump(sagd_dist_matrix, sagd_file, compress=3)

    else: 
        logger.info(f"[D={dim}] SAGD found. Loading...")
        sagd_dist_matrix = joblib.load(sagd_file)
    
    return sagd_dist_matrix

def clustering_job(sagd_dist_matrix: np.ndarray,
                   dim: int, 
                   output_file: Path, 
                   args: argparse.Namespace,
                   logger: logging.Logger):
    time_intervals = None
    if not output_file.exists():
        logger.info(f"[D={dim}] Clustering SAGD matrix")
        time_intervals = cluster_distance_matrix(distances=sagd_dist_matrix, 
                                                 method="dp", 
                                                 weight_exp=2.0,
                                                 penalty_coeff=0.2)
        logger.info(time_intervals)
    else:
        logger.info(f"[D={dim}] Clusters found. Loading...")
        time_intervals = joblib.load(output_file)

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--seed", type=int, default=123456)
    parser.add_argument("--n_samples", type=int, default=1000)
    parser.add_argument("--n_steps", type=int, default=1000)
    parser.add_argument("--T", type=float, default=10.0)
    parser.add_argument("--mu", type=float, default=4.0)

    parser.add_argument("--kernel", type=str, default="gaussian",
                        choices=["gaussian", "inverse_sq_euclidean_d"])
    parser.add_argument("--data-model", type=str, default="bimodal",
                        choices=["bimodal", "hierarchical"])
    parser.add_argument("--laplacian", type=str, default="unnormalized")
    parser.add_argument("--norm_type", type=str, default="norm_wrt_volume",
                        choices=["norm_wrt_volume", "norm_wrt_avg_ctd", "scale_and_shift",
                                 "log_scale_and_shift", "log_global_scale_and_shift"])
    parser.add_argument("--inject_edges", action="store_true", default=True)

    parser.add_argument("--threads", type=int, default=20)
    parser.add_argument("--exp_name", type=str, default="exp_04")
    parser.add_argument("--ds", type=int, nargs="+", help="Explicit list of dimensions to run")

    args = parser.parse_args()
    return args


def main():

    args = parse_args()
    sys.setrecursionlimit(2000)
    torch.manual_seed(args.seed)

    exp_path = Path(f'data/{args.exp_name}')
    exp_path.mkdir(parents=True, exist_ok=True)
    logger = setup_logging(exp_path, args)

    ds = args.ds
    dt = args.T / args.n_steps
    times = np.arange(0, args.T, dt)

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
        ws_file = path / "Ws.jbl"
        ctd_file = path / "CTDs.jbl"
        sagd_file = path / "SAGD.jbl"
        cluster_file = path / "clusters.jbl"

        # 1. SDE Simulation
        history = diffuse_job(
            history_file=history_file,
            dim=d,
            args=args,
            times=times,
            dt=dt,
            snap_time_indices=snap_time_indices,
            mu_star=mu_star,
            std=std,
            ts_theoretical=t_s,
            logger=logger
        )
        time_snaps = list(history.keys())

        # 2. Graph Construction
        edges_to_inject = []
        if args.inject_edges:
            num_nodes = int(args.n_samples * 0.02)
            edges_to_inject = np.random.permutation(args.n_samples)[:num_nodes].reshape(-1, 2)

        w_results_list = construct_graph_job(
            dim=d,
            ws_file=ws_file,
            args=args,
            history=history,
            edges_to_inject=edges_to_inject,
            logger=logger
        )

        # 3. CTD Calculation
        ctds_dict = ctds_job(
            ctd_file=ctd_file,
            args=args,
            w_results=w_results_list,
            time_snaps=time_snaps,
            dim=d,
            logger=logger
        )

        # 4. SAGD Distance Matrix
        sagd_dist_matrix = sagd_job(
            sagd_file=sagd_file,
            args=args,
            ctds_dict=ctds_dict,
            time_snaps=time_snaps,
            dim=d,
            logger=logger
        )

        # 5. Clustering
        clustering_job(sagd_dist_matrix=sagd_dist_matrix,
                       dim=d, 
                       output_file=cluster_file,
                       args=args,
                       logger=logger)

    logger.info('Done!')

if __name__ == "__main__":
    main()