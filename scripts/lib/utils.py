from joblib import Parallel, delayed
from pathlib import Path
import argparse
import logging
import joblib

import numpy as np
import torch
from scipy.spatial.distance import pdist, squareform
from scipy.stats import wasserstein_distance

from tqdm import tqdm
from typing import Any

from lib.adaptive_knn import AdaptiveKNNGraph
from lib.clustering import cluster_distance_matrix
from lib.ou_model import backward, theoretical_bimodal_gaussian_ts, centers
from lib.sgd import compute_sgd, eigen_decompose_job
from lib.stats import normalize
from lib.distances import ctd_matrix

def ctd_job(w_result: np.ndarray, laplacian: str) -> np.ndarray:
    C_Gi = ctd_matrix(W=w_result, laplacian_type=laplacian)
    return C_Gi[np.triu_indices(w_result.shape[0], k=1)]


def knn_job(data: np.ndarray, edges_to_inject: list, kernel: str, sigma: Any = None) -> tuple:
    knn_obj = AdaptiveKNNGraph(data=data, edges_to_inject=edges_to_inject, kernel=kernel)
    w_matrix = knn_obj.compute_W(sigma=sigma)
    return knn_obj.k, (knn_obj.sigma if kernel == "gaussian" else None), w_matrix

def diffuse_job(
    history_file: Path,
    dim: int,
    args: argparse.Namespace,
    times: list,
    dt: float,
    snap_time_indices: list,
    mu_star: float,
    std: float,
    ts_theoretical: float | None,
    logger: logging.Logger,
) -> dict:
    weights = fetch_weights(args=args)
    if not history_file.exists():
        history = {}
        x_current = torch.randn(dim, args.n_samples)
        history[args.T] = x_current.T.clone().numpy()

        for step in tqdm(range(args.n_steps), total=args.n_steps, desc=f"[D={dim}] SDE"):
            t = times[step]
            x_current, _ = backward(
                x_t=x_current,
                t=t,
                dt=dt,
                mu_star=mu_star,
                std=std,
                model=args.data_model,
                weights=weights,
            )
            if step in snap_time_indices:
                history[t] = x_current.T.clone().numpy()
        data_to_dump = {
            "history": history,
            "params": {
                **vars(args),
                "dim": dim,
                "times_snapshots": list(history.keys()),
                "times": times,
            },
        }
        if args.data_model == "bimodal_gaussian":
            data_to_dump["params"]["ts_theoretical"] = ts_theoretical
        joblib.dump(data_to_dump, history_file, compress=3)
    else:
        logger.info(f"[D={dim}] History found. Loading...")
        history = joblib.load(history_file)["history"]

    return history


def construct_graph_job(
    ws_file: Path,
    args: argparse.Namespace,
    history: dict,
    edges_to_inject: np.ndarray | None,
    logger: logging.Logger,
):
    time_snaps = list(history.keys())
    if not ws_file.exists():
        knn_results = Parallel(n_jobs=args.threads, backend="threading")(
            delayed(knn_job)(history[t], edges_to_inject, args.kernel)
            for t in tqdm(time_snaps, desc="KNN Progress")
        )
        w_results = [w for *_, w in knn_results]
        k_results = [k for k, *_ in knn_results]

        results_dict = {
            "Ws": w_results,
            "ks": k_results,
            "ts": time_snaps,
            "kernel": args.kernel,
        }
        if args.kernel == "gaussian":
            results_dict["sigma"] = [sigma for _, sigma, _ in knn_results]

        joblib.dump(results_dict, ws_file, compress=3)
    else:
        logger.info("Ws found. Loading...")
        w_results = joblib.load(ws_file)["Ws"]
    return w_results


def ctds_job(
    ctd_file: Path,
    args: argparse.Namespace,
    w_results: list,
    time_snaps: list,
    logger: logging.Logger,
) -> dict:
    if not ctd_file.exists():
        ctds = Parallel(n_jobs=args.threads)(
            delayed(ctd_job)(W_i, args.laplacian)
            for W_i in tqdm(w_results, total=len(w_results), desc="CTD Logic")
        )
        all_log_ctds = None
        if args.norm_type == "log_global_scale_and_shift":
            all_log_ctds = np.concatenate([np.log1p(triu_i) for triu_i in ctds])
        ctds_dict = {
            t: {
                "ctds":triu_i,
                "norm_ctds": normalize(
                    list_values=triu_i,
                    norm_type=args.norm_type,
                    global_list_values=all_log_ctds,
                    clipping=args.clipping,
                )
            }
            for t, triu_i in zip(time_snaps, ctds)
        }
        joblib.dump(
            {
                "CTDs": ctds_dict,
                "params": {
                    "laplacian": args.laplacian,
                    "normalization": args.norm_type,
                    "ts": time_snaps,
                },
            },
            ctd_file,
            compress=3,
        )
    else:
        logger.info("CTDs found. Loading...")
        ctds_dict = joblib.load(ctd_file)["CTDs"]
    return ctds_dict


def sagd_job(
    sagd_file: Path,
    ctds_dict: dict,
    pairs: list,
    args: argparse.Namespace,
    logger: logging.Logger,
) -> np.ndarray:
    if not sagd_file.exists():
        norm_ctds = [value["norm_ctds"] for value in ctds_dict.values()]
        num_graphs = len(norm_ctds)
        distances = Parallel(n_jobs=args.threads)(
            delayed(wasserstein_distance)(norm_ctds[i], norm_ctds[j])
            for i, j in tqdm(pairs, desc="SAGD Matrix")
        )
        sagd_dist_matrix = np.zeros((num_graphs, num_graphs))
        for (i, j), dist in zip(pairs, distances):
            sagd_dist_matrix[i, j] = sagd_dist_matrix[j, i] = dist
        joblib.dump(sagd_dist_matrix, sagd_file, compress=3)
    else:
        logger.info("SAGD found. Loading...")
        sagd_dist_matrix = joblib.load(sagd_file)
    return sagd_dist_matrix


def sgd_matrix_job(
    w_list: list,
    pairs: list,
    sgd_file: Path,
    args: argparse.Namespace,
    logger: logging.Logger,
) -> np.ndarray:
    if not sgd_file.exists():
        graphs_eigvec_list = Parallel(n_jobs=args.threads)(
            delayed(eigen_decompose_job)(W_i, args.laplacian, norm=True)
            for W_i in tqdm(w_list, total=len(w_list), desc="Eigen-decomposition SGD")
        )
        distances = Parallel(n_jobs=args.threads)(
            delayed(compute_sgd)(graphs_eigvec_list[i], graphs_eigvec_list[j])
            for i, j in tqdm(pairs, desc="SGD Matrix")
        )
        n = len(w_list)
        sgd_dist_matrix = np.zeros((n, n))
        for (i, j), dist in zip(pairs, distances):
            sgd_dist_matrix[i, j] = sgd_dist_matrix[j, i] = dist
        joblib.dump(sgd_dist_matrix, sgd_file, compress=3)
    else:
        logger.info("SGD found. Loading...")
        sgd_dist_matrix = joblib.load(sgd_file)
    return sgd_dist_matrix


def clustering_job(
    distance_matrix: np.ndarray,
    output_file: Path,
    logger: logging.Logger,
):
    if not output_file.exists():
        logger.info("Clustering distance matrix")
        breakpoints = cluster_distance_matrix(distances=distance_matrix, method="dp")
        joblib.dump(breakpoints, output_file, compress=3)
    else:
        logger.info("Breakpoints found. Loading...")
        breakpoints = joblib.load(output_file)
    return breakpoints


def sasne_job(
        sasne_file_path: Path,
        distance_matrix: np.ndarray,
        dim: int = 2,
):
        if not sasne_file_path.exists():
            from SASNE.SASNE import SASNE
            embedding, Z = SASNE(data=distance_matrix, n_components=dim)
            joblib.dump(
                {
                    "embedding": embedding,
                    "Z": Z,
                    "dim": dim,
                    "D1": squareform(pdist(embedding)),
                    "D2": squareform(pdist(Z))
                },
                sasne_file_path, compress=3
            )

def fetch_weights(args: argparse.Namespace) -> np.ndarray | None:
    if args.data_model == "hierarchical_gaussian" and args.hierarchical_weights:
        return args.hierarchical_clusters_size / np.sum(args.hierarchical_clusters_size)
    return None


def get_snap_times(data_model:str, mu:float, times: list, ds: list) -> list:
    if data_model == "bimodal_gaussian":
        assert all(times[i] > times[i + 1] for i in range(len(times) - 1)), (
            "`times` must be strictly descending (t=T -> t=0). "
        )
        ts_indices = []
        for d in ds:
            mu_star = torch.ones(d) * mu
            _, ts_idx = theoretical_bimodal_gaussian_ts(mu_star, 1.0, times)
            ts_indices.append(ts_idx)
        min_ts_idx = min(ts_indices)
        coarse = list(range(0, min_ts_idx, 10))
        dense = list(range(min_ts_idx, len(times), 3))
        return [int(i) for i in sorted(set(coarse + dense + ts_indices + [len(times) - 1]), reverse=True)]
    else:
        return sorted(
            set(list(range(0, len(times), 10)) + [len(times) - 1]),
            reverse=True,
        )


def flatten_mnist_history(history: dict) -> dict:
    """Flatten image snapshots from (N, C, H, W) to (N, C*H*W)."""
    return {
        t: arr.reshape(arr.shape[0], -1) if arr.ndim > 2 else arr
        for t, arr in history.items()
    }


def fetch_pairs(num_graphs: int) -> list:
    return [(i, j) for i in range(num_graphs) for j in range(i + 1, num_graphs)]


def build_edges_to_inject(args: argparse.Namespace) -> np.ndarray | None:
    if not args.inject_edges:
        return None
    num_nodes = int(args.n_samples * 0.02)
    return np.random.permutation(args.n_samples)[:num_nodes].reshape(-1, 2)
