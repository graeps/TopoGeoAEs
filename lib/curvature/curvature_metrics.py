from typing import Union

import numpy as np


def compute_curvature_error_mse(
        curv1,
        curv2,
        eps: float = 1e-12,
) -> Union[float, np.ndarray]:
    """Compute mean squared error (MSE) between two curvature arrays.

    Values are clipped from below by ``eps`` prior to computing the MSE, which
    can help avoid numerical issues if some values are extremely small.

    Args:
        curv1: First sequence/array of curvature values.
        curv2: Second sequence/array of curvature values.
        eps: Small positive floor applied elementwise to both arrays.

    Returns:
        Mean squared error as a float or NumPy scalar.
    """
    arr1 = np.asarray(curv1)
    arr2 = np.asarray(curv2)
    arr1 = np.clip(arr1, eps, None)
    arr2 = np.clip(arr2, eps, None)

    diff = (arr1 - arr2) ** 2
    mse = diff.sum() / len(arr1)
    return mse


def compute_curvature_error_smape(
        true_curv,
        approx_curv,
        eps: float = 1e-12,
) -> Union[float, np.ndarray]:
    """Compute the symmetric mean absolute percentage error (SMAPE).

    SMAPE is computed as:
        mean(|y_true - y_pred| / (|y_true| + |y_pred|))

    Args:
        true_curv: Ground-truth curvature values.
        approx_curv: Approximated curvature values.
        eps: Small positive value to avoid division by zero.

    Returns:
        SMAPE percentage in [0, 100] as a float or NumPy scalar.
    """
    true_arr = np.asarray(true_curv)
    approx_arr = np.asarray(approx_curv)

    denominator = np.clip(np.abs(true_arr) + np.abs(approx_arr), eps, None)
    smape = np.abs(true_arr - approx_arr) / denominator

    return 100 * np.mean(smape)
