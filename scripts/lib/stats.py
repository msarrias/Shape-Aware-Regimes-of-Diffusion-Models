import numpy as np
from collections import Counter
from scipy.stats import rankdata


def scale_and_shift(
        arr: np.ndarray,
):
    v_min, v_max = arr.min(), arr.max()
    return (arr - v_min) / (v_max - v_min)


def normalize(
        list_values: list,
        norm_type: str,
        global_list_values: np.ndarray | None,
        clipping: bool = False,
        clip_percentile: float = 95
):
    arr = np.asarray(list_values)

    if norm_type == "log_global_scale_and_shift":
        log_arr = np.log1p(arr)

        if clipping:
            upper = np.percentile(global_list_values, clip_percentile)
            lower = np.percentile(global_list_values, 100 - clip_percentile)
            log_arr = np.clip(log_arr, lower, upper)
        else:
            lower = np.min(log_arr)
            upper = np.max(log_arr)

        return (log_arr - lower) / (upper - lower)

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
