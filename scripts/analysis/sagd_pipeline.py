import joblib
import numpy as np
import sys
import torch
import argparse
import logging
from pathlib import Path
from logger import setup_logging, parse_args
from lib.utils import (
    fetch_pairs,
    build_edges_to_inject,
    construct_graph_job,
    ctds_job, sagd_job, sgd_matrix_job, clustering_job,
    sasne_job, diffuse_job, flatten_mnist_history,
    get_snap_times, theoretical_bimodal_gaussian_ts, centers
)

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
    times = np.linspace(args.T, dt, args.n_steps).tolist()
    snap_time_indices = get_snap_times(data_model=args.data_model, mu=args.mu, times=times, ds=args.ds)

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

    exp_path = Path(args.save_path) / args.exp_name if args.save_path else Path(f"data/{args.exp_name}")
    exp_path.mkdir(parents=True, exist_ok=True)

    logger = setup_logging(exp_path, args)

    if args.data_model == "mnist_unet_diffusion":
        run_mnist(exp_path, args, logger)
    else:
        run_synthetic(exp_path, args, logger)

    logger.info("Done!")


if __name__ == "__main__":
    main()