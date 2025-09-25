"""
    Adapted from:
    Acosta, Francisco, et al. "Quantifying extrinsic curvature in neural manifolds."
    Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 2023.
    https://arxiv.org/abs/2212.10414
    https://github.com/geometric-intelligence/neurometry
"""

import os
from typing import Optional, Tuple

# Set the geomstats backend before any geomstats import
os.environ["GEOMSTATS_BACKEND"] = "pytorch"

import torch
from tqdm import tqdm
from geomstats.geometry.pullback_metric import PullbackMetric

from .utils import (
    NeuralManifoldIntrinsic,
    get_learned_immersion,
    get_true_immersion,
)
from lib.errors import InvalidConfigError
from .utils import get_z_grid
from ..datasets.lookup import get_manifold_dim, get_dataset_category
from ..models.lookup import is_non_euclidean_model


def _clip_by_quantile(
        values: torch.Tensor, low: float = 0.01, high: float = 0.99
) -> torch.Tensor:
    """
    Clip values into [low, high] quantiles computed on valid (non-NaN) entries,
    then ensure non-negativity. If all values are NaN, returns clamped zeros.

    Args:
        values: 1D tensor of values (may contain NaNs)
        low: lower quantile in [0, 1]
        high: upper quantile in [0, 1]

    Returns:
        Tensor of the same shape as values with quantile clipping and non-negativity enforced.
    """
    if values.ndim != 1:
        # Flatten while preserving device/dtype; final shape restored by view_as below
        flat = values.view(-1)
    else:
        flat = values

    valid_mask = ~torch.isnan(flat)
    if not torch.any(valid_mask):
        # No valid entries — return non-negative zeros
        result = torch.zeros_like(flat)
        return result.view_as(values)

    v = flat[valid_mask]
    # Quantiles computed on-device and dtype-consistent
    q_low, q_high = torch.quantile(
        v, torch.tensor((low, high), dtype=v.dtype, device=v.device)
    )
    clipped = torch.clamp(flat, min=q_low.item(), max=q_high.item())
    clipped = torch.clamp(clipped, min=0.0)
    return clipped.view_as(values)


def _compute_curvature(
        z_grid: torch.Tensor,
        immersion,
        dim: int,
        embedding_dim: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute mean curvature vectors and their norms from an immersion via a pullback metric.

    Args:
        z_grid: Parameter grid on the intrinsic manifold (N x d tensor-like).
        immersion: Callable immersion mapping from intrinsic coords to ambient space.
        dim: Intrinsic manifold dimension.
        embedding_dim: Ambient (embedding) space dimension.

    Returns:
        curv: Tensor of shape (N, embedding_dim) with mean curvature vectors (NaN where unavailable).
        curv_norm: Tensor of shape (N,) with clipped curvature norms (non-negative).
    """
    neural_manifold = NeuralManifoldIntrinsic(dim, embedding_dim, immersion, equip=False)
    neural_manifold.equip_with_metric(PullbackMetric)

    # Allocate on the same device as z_grid if possible
    device = z_grid.device if isinstance(z_grid, torch.Tensor) else None
    curv = torch.full(
        (len(z_grid), embedding_dim),
        torch.nan,
        device=device,
        dtype=z_grid.dtype if isinstance(z_grid, torch.Tensor) else None,
    )

    if dim == 1:
        # For 1D, mean_curvature_vector expects a batch; unsqueeze each sample
        for i, z in tqdm(
                enumerate(z_grid),
                desc="Computing curvature from immersion (Manifold Dim = 1)",
                total=len(z_grid),
                leave=False,
        ):
            try:
                z_ = torch.unsqueeze(z, dim=0)
                curv[i, :] = neural_manifold.metric.mean_curvature_vector(z_)
            except Exception as e:
                # Keep NaN on failure, but report the index for debugging
                print(f"An error occurred for i={i}: {e}")
    else:
        for i, z in tqdm(
                enumerate(z_grid),
                desc="Computing curvature from immersion (Manifold Dim > 1)",
                total=len(z_grid),
                leave=False,
        ):
            try:
                curv[i, :] = neural_manifold.metric.mean_curvature_vector(z)
            except Exception as e:
                print(f"An error occurred for i={i}: {e}")

    # Norm along embedding dimension; shape (N,)
    curv_norm = torch.linalg.norm(curv, dim=1)

    # Quantile-based clipping to reduce spikes and enforce non-negativity
    curv_norm = _clip_by_quantile(curv_norm, low=0.01, high=0.99)

    return curv, curv_norm


def compute_curvature_learned(
        config,
        model,
        n_grid_points: int = 2000,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], torch.Tensor]:
    """
    Compute curvature for a learned immersion.
    This function is only supported for non-Euclidean models.

    Args:
        config: Configuration object with fields: model_type, dataset_name, embedding_dim, verbose.
        model: Trained model from which to extract the learned immersion.
        n_grid_points: Number of grid points to sample for non-Euclidean latent manifolds.

    Returns:
        z_grid: Intrinsic coordinates used for curvature computation.
        curv: Mean curvature vectors at z_grid.
        curv_norm: Clipped norms of curvature vectors at z_grid.

    Raises:
        InvalidConfigError: If the model type is not non-Euclidean or the dataset category is unsupported.
    """
    if getattr(config, "verbose", False):
        print("Computing learned curvature...")

    if not is_non_euclidean_model(config.model_type):
        raise InvalidConfigError(
            f"compute_curvature_learned is only supported for non-Euclidean models; got: {config.model_type}"
        )

    z_grid = get_z_grid(config=config, n_grid_points=n_grid_points)

    immersion = get_learned_immersion(model=model, config=config)
    category = get_dataset_category(config.dataset_name)
    if category == "entity":
        raise InvalidConfigError(
            f"Learned curvature not supported for entity datasets: {config.dataset_name}"
        )

    manifold_dim = get_manifold_dim(config.dataset_name)
    curv, curv_norm = _compute_curvature(
        z_grid=z_grid,
        immersion=immersion,
        dim=manifold_dim,
        embedding_dim=config.embedding_dim,
    )
    return z_grid, curv, curv_norm


def compute_curvature_true(
        config,
        n_grid_points: int = 2000,
        cache_dir: str = "./true_curvature_cache",
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute curvature for the true (ground-truth) immersion, with caching.

    Not applicable to multi_entity datasets. For such datasets, this function raises InvalidConfigError.

    Args:
        config: Configuration with fields: dataset_name, embedding_dim, compute_emp_curv,
                n_points_pullback_curv, deformation_amp, verbose.
        n_grid_points: Number of grid points used if empirical labels are not used.
        cache_dir: Directory where computed curvature is cached.

    Returns:
        angles_or_grid: Intrinsic coordinates used for curvature computation.
        curv: Mean curvature vectors.
        curv_norm: Clipped norms of curvature vectors.

    Raises:
        InvalidConfigError: On unsupported datasets, or multi_entity datasets.
    """
    os.makedirs(cache_dir, exist_ok=True)
    deformation = config.deformation_amp
    name = f"{config.dataset_name}_{config.n_points_pullback_curv}_{deformation}.pt"
    cache_path = os.path.join(cache_dir, name)

    if os.path.exists(cache_path):
        if getattr(config, "verbose", False):
            print(f"Loading cached curvature from {cache_path}")
        return torch.load(cache_path)

    if getattr(config, "verbose", False):
        print("Computing true curvature...")

    angles = get_z_grid(config, n_grid_points)

    category = get_dataset_category(config.dataset_name)
    if category in {"1d", "2d"}:
        immersion = get_true_immersion(config)
        dim = get_manifold_dim(config.dataset_name)
        curv, curv_norm = _compute_curvature(
            angles, immersion, dim, config.embedding_dim
        )

    elif category == "multi_entity":
        raise InvalidConfigError(
            f"compute_curvature_true is not applicable to multi_entity datasets: {config.dataset_name}"
        )

    else:
        raise InvalidConfigError(f"Unknown dataset: {config.dataset_name}")

    torch.save((angles, curv, curv_norm), cache_path)
    return angles, curv, curv_norm
