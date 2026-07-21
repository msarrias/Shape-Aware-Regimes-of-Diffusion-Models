import numpy as np
from collections import Counter
from scipy.stats import rankdata


def scale_and_shift(
        arr: np.ndarray,
):
    v_min, v_max = arr.min(), arr.max()
    return (arr - v_min) / (v_max - v_min)


def compress_equal_frequency(arr: np.ndarray, n_bins: int) -> np.ndarray:
    """Compress a 1-D distribution to `n_bins` equal-frequency quantiles."""
    arr = np.asarray(arr)
    if arr.size <= n_bins:
        return arr
    return np.quantile(arr, np.linspace(0.0, 1.0, n_bins))


def normalize(
        list_values: list,
        norm_type: str,
        clipping: bool = False,
        clip_percentile: float = 95
):
    arr = np.asarray(list_values)
    if clipping:
        upper = np.percentile(arr, clip_percentile)
        lower = np.percentile(arr, 100 - clip_percentile)
        arr = np.clip(arr, lower, upper)

    if norm_type == "scale_and_shift":
        return scale_and_shift(arr)

    elif norm_type == "log_scale_and_shift":
        log_ctd = np.log1p(arr)
        return scale_and_shift(log_ctd)

    elif norm_type == "norm_wrt_volume":
        return arr / np.sum(arr)

    elif norm_type == "norm_wrt_avg_ctd":
        return arr / np.mean(arr)

    raise ValueError("Unsupported norm_type")
