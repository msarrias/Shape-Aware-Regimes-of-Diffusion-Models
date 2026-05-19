import argparse
import joblib
import logging
import numpy as np
import sys
import torch
from joblib import Parallel, delayed
from pathlib import Path
from scipy.spatial.distance import pdist, squareform
from scipy.stats import wasserstein_distance
from tqdm import tqdm

from adaptive_knn import AdaptiveKNNGraph
from clustering import cluster_distance_matrix
from ou_model import backward, theoretical_ts, centers
from sgd import compute_sgd, _eigen_decompose_job
from stats import normalize
from distances import CTD_matrix


def setup_logging(exp_path, args) -> logging.Logger:
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument("--seed", type=int, default=123456)
    parser.add_argument("--n_samples", type=int, default=1000)
    parser.add_argument("--n_steps", type=int, default=1000)
    parser.add_argument("--T", type=float, default=10.0)
    parser.add_argument("--mu", type=float, default=4.0)

    parser.add_argument("--kernel", type=str, default="gaussian",
                        choices=["gaussian", "inverse_sq_euclidean_d"])
    parser.add_argument("--data_model", type=str, default="bimodal",
                        choices=["bimodal", "hierarchical"]
                        )
    parser.add_argument("--laplacian", type=str, default="unnormalized")
    parser.add_argument("--norm_type", type=str, default="norm_wrt_volume",
                        choices=["norm_wrt_volume", "norm_wrt_avg_ctd", "scale_and_shift",
                                 "log_scale_and_shift", "log_global_scale_and_shift"]
                        )
    parser.add_argument("--distance", type=str, default="SAGD", choices=["SAGD", "SGD"])
    parser.add_argument("--inject_edges", action="store_true", default=False)
    parser.add_argument("--clipping", action="store_true", default=False)
    parser.add_argument("--generate_sasne_embedding", action="store_true", default=False)
    parser.add_argument("--hierarchical_weights", action="store_true", default=False)
    parser.add_argument("--hierarchical_sigma", default=[1, 1, 1, 1, 1, 1])
    parser.add_argument("--hierarchical_clusters_size", default=[400, 200, 100, 300, 150, 350])
    parser.add_argument("--mu_macro", type=float, default=8)
    parser.add_argument("--mu_micro", type=float, default=6)

    parser.add_argument("--threads", type=int, default=20)
    parser.add_argument("--exp_name", type=str, default="exp_04")
    parser.add_argument("--ds", type=int, nargs="+", help="Explicit list of dimensions to run")

    args = parser.parse_args()
    return args


def ctd_job(
        w_result: np.ndarray,
        laplacian: str
) -> np.ndarray:
    C_Gi = CTD_matrix(
        W=w_result,
        laplacian_type=laplacian
    )
    upper_tri = C_Gi[np.triu_indices(w_result.shape[0], k=1)]

    return upper_tri

def knn_job(
        data: np.ndarray,
        edges_to_inject: list,
        kernel: str,
        sigma: float=None
) -> np.ndarray:
    knn_obj = AdaptiveKNNGraph(
        data=data,
        edges_to_inject=edges_to_inject,
        kernel=kernel
    )
    w_matrix = knn_obj.compute_W(sigma=sigma)
    return knn_obj.k, (knn_obj.sigma if kernel == 'gaussian' else None), w_matrix


def fetch_weights(
        args: argparse.Namespace
):
    if args.data_model == 'hierarchical' and args.hierarchical_weights:
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
        ts_theoretical: float,
        logger: logging.Logger,
) -> np.ndarray:
    weights = fetch_weights(args=args)
    if not history_file.exists():
        history = {}
        x_current = torch.randn(dim, args.n_samples)  # d x N
        history[args.T] = x_current.T.clone().numpy()  # N x d

        for step in tqdm(reversed(range(args.n_steps)), total=args.n_steps, desc=f"[D={dim}] SDE"):
            t = times[step]
            x_current, _ = backward(
                x_t=x_current,
                t=t,
                dt=dt,
                mu_star=mu_star,
                std=std,
                model=args.data_model,
                weights=weights
            )
            if step in snap_time_indices:
                history[t] = x_current.T.clone().numpy()
        data_to_dump = {
            "history": history,
            "params": {
                **vars(args),
                "dim": dim,
                "times_snapshots": list(history.keys()),
                "times": times
            }
        }
        if args.data_model == 'bimodal':
            data_to_dump["params"]["ts_theoretical"] = ts_theoretical
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
) -> dict:
    if not ctd_file.exists():
        ctds = Parallel(n_jobs=args.threads)(
            delayed(ctd_job)(W_i, args.laplacian)
            for W_i in tqdm(w_results, total=len(w_results), desc=f"[D={dim}] CTD Logic")
        )
        all_log_ctds = []
        if args.norm_type == "log_global_scale_and_shift":
            all_log_ctds = np.concatenate([np.log1p(triu_i) for triu_i in ctds])
        ctds_dict = {
            t: {
                'ctds': triu_i,
                'norm_ctds': normalize(
                    list_values=triu_i,
                    norm_type=args.norm_type,
                    global_list_values=all_log_ctds,
                    clipping=args.clipping
                )}
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
    sagd_file: Path,
    ctds_dict: dict,
    dim: int,
    pairs: list,
    args: argparse.Namespace,
    logger: logging.Logger
) -> np.array:
    if not sagd_file.exists():
        norm_ctds = [value['norm_ctds'] for value in ctds_dict.values()]
        num_graphs = len(norm_ctds)
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


def sgd_matrix_job(
        dim,
        w_list: list,
        pairs: list,
        sgd_file: Path,
        args: argparse.Namespace,
        logger: logging.Logger
) -> np.ndarray:
    if not sgd_file.exists():
        # eigen decomposition for all graphs
        graphs_eigvec_list = Parallel(n_jobs=args.threads)(
            delayed(_eigen_decompose_job)(W_i, args.laplacian, norm=True)
            for W_i in tqdm(w_list, total=len(w_list),
                            desc=f"[D={dim}] Eigen-decomposition SGD")
        )

        # pairwise SGD distances
        distances = Parallel(n_jobs=args.threads)(
            delayed(compute_sgd)(graphs_eigvec_list[i], graphs_eigvec_list[j])
            for i, j in tqdm(pairs, desc=f"[D={dim}] SGD Matrix")
        )

        n = len(w_list)
        sgd_dist_matrix = np.zeros((n, n))
        for (i, j), dist in zip(pairs, distances):
            sgd_dist_matrix[i, j] = sgd_dist_matrix[j, i] = dist

        joblib.dump(sgd_dist_matrix, sgd_file, compress=3)

    else:
        logger.info(f"[D={dim}] SGD found. Loading...")
        sgd_dist_matrix = joblib.load(sgd_file)
    return sgd_dist_matrix

    
def clustering_job(
        distance_matrix: np.ndarray,
        dim: int,
        output_file: Path,
        logger: logging.Logger
):
    if not output_file.exists():
        logger.info(f"[D={dim}] Clustering SAGD matrix")
        breakpoints = cluster_distance_matrix(
            distances=distance_matrix,
            method="dp"
        )
        joblib.dump(breakpoints, output_file, compress=3)
    else:
        logger.info(f"[D={dim}] Breakpoints found. Loading...")
        breakpoints = joblib.load(output_file)
    return breakpoints


def sasne_job(
        sasne_file_path: Path,
        distance_matrix: np.ndarray,
):
        if not sasne_file_path.exists():
            from SASNE.SASNE import SASNE
            embedding, Z = SASNE(data=distance_matrix)
            joblib.dump({
                "embedding": embedding,
                "Z": Z,
                "D1": squareform(pdist(embedding)),
                "D2": squareform(pdist(Z))
            }, sasne_file_path, compress=3)


def get_snap_times(args, times, ds):
    if args.data_model=="bimodal":
        # We want every dimension to share the same snapshots,
        # including all theoretical ts points
        ts_indices = []
        for d in ds:
            mu_star = torch.ones(d) * args.mu
            _, ts_idx = theoretical_ts(mu_star, 1.0, times)
            ts_indices.append(ts_idx)

        return sorted(
            list(set(list(range(0, len(times), 10)) + ts_indices + [len(times) - 1])),
            reverse=True
        )
    else:
        return sorted(
            list(set(list(range(0, len(times), 10)) + [len(times) - 1])),
            reverse=True
        )


def fetch_pairs(
        num_graphs: int
):
    return [
        (i, j)
        for i in range(num_graphs)
        for j in range(i + 1, num_graphs)
    ]

    
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
    snap_time_indices = get_snap_times(args, times, ds)
    
    if args.data_model=="hierarchical":
        assert args.n_samples == sum(args.hierarchical_clusters_size)

    for d in ds:
        t_s = None
        if args.data_model=="bimodal":
            mu_star = torch.ones(d) * args.mu
            std = 1.0
            t_s, _ = theoretical_ts(mu_star, std, times)

        elif args.data_model=="hierarchical":
            mu_star = centers(d=d, mu_macro=args.mu_macro, mu_micro=args.mu_micro)
            std = args.hierarchical_sigma

        path = exp_path / f"D{d}_N{args.n_samples}_T{int(args.T)}/"
        path.mkdir(parents=True, exist_ok=True)
        history_file = path / f"history.jbl"
        ws_file = path / "Ws.jbl"
        ctd_file = path / "CTDs.jbl"
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
            ts_theoretical=t_s if args.data_model == "bimodal" else None,
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

        pairs = fetch_pairs(num_graphs=len(time_snaps))

        if args.distance == 'SAGD':
            sagd_file = path / "SAGD.jbl"
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
            distance_matrix = sagd_job(
                sagd_file=sagd_file,
                args=args,
                ctds_dict=ctds_dict,
                dim=d,
                pairs=pairs,
                logger=logger
            )
        elif args.distance == 'SGD':
            sgd_file = path / "SGD.jbl"
            distance_matrix = sgd_matrix_job(
                dim=d,
                w_list=w_results_list,
                pairs=pairs,
                sgd_file=sgd_file,
                args=args,
                logger=logger
            )
        else:
            raise NotImplementedError

        # 5. Clustering
        _ = clustering_job(
            distance_matrix=distance_matrix,
            dim=d,
            output_file=cluster_file,
            logger=logger
        )

        #. 6 SASNE embedding -- optional
        if args.generate_sasne_embedding:
            sasne_file= path / f"SASNE.jbl"
            sasne_job(
                sasne_file_path=sasne_file,
                distance_matrix=distance_matrix
            )

    logger.info('Done!')

if __name__ == "__main__":
    main()