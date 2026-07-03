import joblib
import numpy as np
import sys
import torch
from joblib import Parallel, delayed
from pathlib import Path

from scipy.spatial.distance import pdist, squareform
from scipy.stats import wasserstein_distance
from tqdm import tqdm
from typing import Any

from logger import setup_logging, parse_args
from lib.adaptive_knn import AdaptiveKNNGraph
from lib.clustering import cluster_distance_matrix
from lib.ou_model import backward, theoretical_bimodal_gaussian_ts, centers
from lib.sgd import compute_sgd, eigen_decompose_job
from lib.stats import normalize
from lib.distances import ctd_matrix



def ctd_job(w_result: np.ndarray, laplacian: str) -> np.ndarray:
    C_Gi = CTD_matrix(W=w_result, laplacian_type=laplacian)
    return C_Gi[np.triu_indices(w_result.shape[0], k=1)]


def knn_job(data: np.ndarray, edges_to_inject: list, kernel: str, sigma: Any = None) -> tuple:
    knn_obj = AdaptiveKNNGraph(data=data, edges_to_inject=edges_to_inject, kernel=kernel)
    w_matrix = knn_obj.compute_W(sigma=sigma)
    return knn_obj.k, (knn_obj.sigma if kernel == "gaussian" else None), w_matrix


def fetch_weights(args: argparse.Namespace) -> np.ndarray | None:
    if args.data_model == "hierarchical_gaussian" and args.hierarchical_weights:
        return args.hierarchical_clusters_size / np.sum(args.hierarchical_clusters_size)
    return None


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
            delayed(_eigen_decompose_job)(W_i, args.laplacian, norm=True)
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


def get_snap_times(args: argparse.Namespace, times: list, ds: list) -> list:
    if args.data_model == "bimodal_gaussian":
        ts_indices = []
        for d in ds:
            mu_star = torch.ones(d) * args.mu
            _, ts_idx = theoretical_bimodal_gaussian_ts(mu_star, 1.0, times)
            ts_indices.append(ts_idx)
        max_ts_idx = max(ts_indices)
        coarse = list(range(max_ts_idx, len(times), 10))
        dense = list(range(0, max_ts_idx, 3))
        return sorted(set(coarse + dense + ts_indices + [len(times) - 1]), reverse=True)
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


def run_pipeline(
    path: Path,
    history: dict,
    args: argparse.Namespace,
    logger: logging.Logger,
) -> None:
    """Graph construction → distance matrix → clustering → (optional) SA-SNE."""
    time_snaps = list(history.keys())
    pairs = fetch_pairs(num_graphs=len(time_snaps))
    edges_to_inject = build_edges_to_inject(args)

    # 1. Graph construction
    w_results = construct_graph_job(
        ws_file=path / "Ws.jbl",
        args=args,
        history=history,
        edges_to_inject=edges_to_inject,
        logger=logger,
    )

    # 2. Distance matrix
    if args.distance == "SAGD":
        ctds_dict = ctds_job(
            ctd_file=path / "CTDs.jbl",
            args=args,
            w_results=w_results,
            time_snaps=time_snaps,
            logger=logger,
        )
        distance_matrix = sagd_job(
            sagd_file=path / "SAGD.jbl",
            ctds_dict=ctds_dict,
            pairs=pairs,
            args=args,
            logger=logger,
        )
    elif args.distance == "SGD":
        distance_matrix = sgd_matrix_job(
            w_list=w_results,
            pairs=pairs,
            sgd_file=path / "SGD.jbl",
            args=args,
            logger=logger,
        )
    else:
        raise NotImplementedError(f"Unknown distance: {args.distance}")

    # 3. Clustering
    _ = clustering_job(
        distance_matrix=distance_matrix,
        output_file=path / "clusters.jbl",
        logger=logger,
    )

    # 4. SASNE embedding (optional)
    if args.generate_sasne_embedding:
        sasne_job(
            sasne_file_path=path / "SASNE.jbl",
            distance_matrix=distance_matrix,
            dim=args.sasne_dimension
        )


def run_mnist(exp_path: Path, args: argparse.Namespace, logger: logging.Logger) -> None:
    for subdir in (d for d in exp_path.iterdir() if d.is_dir()):
        logger.info(f"Running diffusion for MNIST: {subdir.name}")
        history = joblib.load(subdir / "history.jbl")
        history = flatten_mnist_history(history)
        run_pipeline(path=subdir, history=history, args=args, logger=logger)    

def run_synthetic(exp_path: Path, args: argparse.Namespace, logger: logging.Logger) -> None:
    dt = args.T / args.n_steps
    times = list(np.arange(args.T, 0, -dt))
    snap_time_indices = get_snap_times(args, times, args.ds)

    if args.data_model == "hierarchical_gaussian":
        assert args.n_samples == sum(args.hierarchical_clusters_size)

    for d in args.ds:
        logger.info(f"Running diffusion for D={d}")
        if args.data_model == "bimodal_gaussian":
            mu_star = torch.ones(d) * args.mu
            std = 1.0
            t_s, _ = theoretical_bimodal_gaussian_ts(mu_star, std, times)
        elif args.data_model == "hierarchical_gaussian":
            mu_star = centers(d=d, mu_macro=args.mu_macro, mu_micro=args.mu_micro)
            std = args.hierarchical_sigma
            t_s = None
        else:
            raise NotImplementedError(f"Unknown data model: {args.data_model}")

        path = exp_path / f"D{d}_N{args.n_samples}_T{int(args.T)}"
        path.mkdir(parents=True, exist_ok=True)

        history = diffuse_job(
            history_file=path / "history.jbl",
            dim=d,
            args=args,
            times=times,
            dt=dt,
            snap_time_indices=snap_time_indices,
            mu_star=mu_star,
            std=std,
            ts_theoretical=t_s,
            logger=logger,
        )
        run_pipeline(path=path, history=history, args=args, logger=logger)


def main() -> None:
    args = parse_args()
    sys.setrecursionlimit(2000)
    torch.manual_seed(args.seed)

    exp_path = Path(args.save_path) if args.save_path else Path(f"data/{args.exp_name}")
    exp_path.mkdir(parents=True, exist_ok=True)

    logger = setup_logging(exp_path, args)

    if args.data_model == "mnist_unet_diffusion":
        run_mnist(exp_path, args, logger)
    else:
        run_synthetic(exp_path, args, logger)

    logger.info("Done!")


if __name__ == "__main__":
    main()