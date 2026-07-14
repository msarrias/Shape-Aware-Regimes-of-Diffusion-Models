import argparse
import logging
import sys
from pathlib import Path


def setup_logging(exp_path: Path, args) -> logging.Logger:
    HIERARCHICAL_PARAMS = {
        "hierarchical_weights",
        "hierarchical_sigma",
        "hierarchical_clusters_size",
        "mu_macro",
        "mu_micro",
    }

    log_file = exp_path / "settings.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout),
        ],
    )
    logger = logging.getLogger()
    logger.info("=== Simulation Settings ===")
    for arg, value in vars(args).items():
        if args.data_model != "hierarchical_gaussian" and arg in HIERARCHICAL_PARAMS:
            continue
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
    parser.add_argument("--threads", type=int, default=20)
    parser.add_argument("--exp_name", type=str, default="exp_04")
    parser.add_argument("--save_path", type=str, default=None)
    parser.add_argument("--ds", type=int, nargs="+")

    parser.add_argument(
        "--kernel",
        type=str,
        default="gaussian",
        choices=["gaussian", "inverse_sq_euclidean_d"],
    )
    parser.add_argument(
        "--data_model",
        type=str,
        default="bimodal_gaussian",
        choices=["bimodal_gaussian", "hierarchical_gaussian", "mnist_unet_diffusion"],
    )
    parser.add_argument(
        "--norm_type",
        type=str,
        default="norm_wrt_volume",
        choices=[
            "norm_wrt_volume",
            "norm_wrt_avg_ctd",
            "scale_and_shift",
            "log_scale_and_shift"
        ],
    )
    parser.add_argument("--laplacian", type=str, default="unnormalized")
    parser.add_argument("--distance", type=str, default="SAGD", choices=["SAGD", "SGD"])

    parser.add_argument("--inject_edges", action="store_true", default=False)
    parser.add_argument("--clipping", action="store_true", default=False)
    parser.add_argument("--generate_sasne_embedding", action="store_true", default=False)
    parser.add_argument("--sasne_dimension", type=int, default=2)

    parser.add_argument("--hierarchical_weights", action="store_true", default=False)
    parser.add_argument(
        "--hierarchical_sigma",
        type=float,
        nargs="+",
        default=[1, 1, 1, 1, 1, 1]
    )
    parser.add_argument(
        "--hierarchical_clusters_size",
        type=int,
        nargs="+",
        default=[400, 200, 100, 300, 150, 350]
    )
    parser.add_argument("--mu_macro", type=float, default=8)
    parser.add_argument("--mu_micro", type=float, default=4)

    return parser.parse_args()