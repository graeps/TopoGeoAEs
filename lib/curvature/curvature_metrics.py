import numpy as np
import torch

from .errors import InvalidConfigError


def _compute_curvature_error_s1(thetas, learned, true):
    learned = np.array(learned)
    true = np.array(true)
    diff = np.trapz((learned - true) ** 2, thetas)
    norm = np.trapz(learned ** 2 + true ** 2, thetas)
    return diff / norm


def _integrate_s2(thetas, phis, h):
    thetas = torch.unique(thetas)
    phis = torch.unique(phis)
    sum_phis = torch.zeros_like(thetas)
    for t, theta in enumerate(thetas):
        sum_phis[t] = torch.trapz(h[t * len(phis):(t + 1) * len(phis)], phis) * np.sin(theta)
    return torch.trapz(sum_phis, thetas)


def _compute_curvature_error_s2(thetas, phis, learned, true):
    diff = _integrate_s2(thetas, phis, (learned - true) ** 2)
    norm = _integrate_s2(thetas, phis, learned ** 2 + true ** 2)
    return diff / norm


def compute_curvature_error_linf(curv1, curv2):
    curv1 = np.array(curv1)
    curv2 = np.array(curv2)
    return np.max(np.abs(curv1 - curv2))


def compute_curvature_error_mse(curv1, curv2, eps=1e-12):
    curv1 = np.array(curv1)
    curv2 = np.array(curv2)
    curv1 = np.clip(curv1, eps, None)
    curv2 = np.clip(curv2, eps, None)

    diff = (curv1 - curv2) ** 2
    mean = diff.sum() / len(curv1)
    return mean


def compute_curvature_error_smape(true_curv, approx_curv, eps=1e-12):
    """
    Computes the symmetric mean absolute percentage error (SMAPE) between true and approximate curvatures.
    """
    true_curv = np.asarray(true_curv)
    approx_curv = np.asarray(approx_curv)

    # Ensure non-zero denominator
    denominator = np.clip(np.abs(true_curv) + np.abs(approx_curv), eps, None)
    smape = np.abs(true_curv - approx_curv) / denominator

    return 100 * np.mean(smape)


def compute_curvature_error(z_grid, learned, true, config):
    if config.dataset_name == "s1_low":
        return _compute_curvature_error_s1(z_grid, learned, true)
    elif config.dataset_name in ("s2_low", "t2_low"):
        return _compute_curvature_error_s2(z_grid[:, 0], z_grid[:, 1], learned, true)
    else:
        raise InvalidConfigError(f"Unknown XX dataset: {config.dataset_name}")
