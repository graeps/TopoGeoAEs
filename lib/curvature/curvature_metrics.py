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
    arr1 = _to_float_array(curv1)
    arr2 = _to_float_array(curv2)

    # Compute only where both arrays are finite
    valid = np.isfinite(arr1) & np.isfinite(arr2)
    if not np.any(valid):
        return float("nan")

    a1 = np.clip(arr1[valid], eps, None)
    a2 = np.clip(arr2[valid], eps, None)

    diff = (a1 - a2) ** 2
    mse = diff.mean()
    return float(mse)


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
    true_arr = _to_float_array(true_curv)
    approx_arr = _to_float_array(approx_curv)

    # Compute only where both arrays are finite
    valid = np.isfinite(true_arr) & np.isfinite(approx_arr)
    if not np.any(valid):
        return float("nan")

    t = true_arr[valid]
    a = approx_arr[valid]

    denominator = np.clip(np.abs(t) + np.abs(a), eps, None)
    smape = np.abs(t - a) / denominator

    return float(100 * np.mean(smape))


def _to_float_array(x) -> np.ndarray:
    """
    Convert input to a float NumPy array, handling None and object-dtype safely.
    - If x is None: return an empty float array of shape (0,).
    - If x has object dtype (e.g., contains None), replace None with NaN and cast to float.
    - Otherwise cast to float without copying if possible.
    """
    if x is None:
        return np.array([], dtype=float)

    arr = np.asarray(x)
    if arr.dtype == object:
        flat = arr.ravel()
        flat_conv = np.array([np.nan if (v is None) else v for v in flat], dtype=float)
        return flat_conv.reshape(arr.shape)
    return arr.astype(float, copy=False)


# TODO: add error metrics to curvature plots
def compute_all_error_metrics(curv1, curv2):
    """
    Calculates error metrics between two lists of curvature estimates.

    This function computes two error metrics: Symmetric Mean Absolute Percentage Error
    (SMAPE) and Mean Squared Error (MSE) between two input curvature sequences.

    Args:
        curv1: Input list of curvature estimates.
        curv2: Another input list of curvature estimates.

    Returns:
        Tuple containing:
            - mse (list): A list of MSE values for the curvature comparison.
            - smape (list): A list of SMAPE values for the curvature comparison.
    """
    smape = []
    mse = []
    try:
        sm = compute_curvature_error_smape(curv1, curv2)
        ms = compute_curvature_error_mse(curv1, curv2)
        smape.append(sm)
        mse.append(ms)
    except Exception as e:
        print(f"compute_all_error_metrics: Skipping metric computation due to error: {e}")
    return mse, smape
