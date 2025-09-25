import os
import json
import numpy as np
import torch

from .pullback_curvature import compute_curvature_true, compute_curvature_learned
from .quadric_curvature import compute_quadric_curvature
from .curvature_metrics import (
    compute_curvature_error_mse,
    compute_curvature_error_smape,
)

os.environ["GEOMSTATS_BACKEND"] = "pytorch"

from scipy.spatial import cKDTree
from ..models.lookup import is_euclidean_model


def compute_all_curvatures(config, model, recons, latents, inputs, labels):
    """
    Computes various curvature metrics and organizes them into a structured dictionary, with options
    to save the results to a file. This function calculates curvatures on
    inputs, latents, and reconstructions, using the best quadric fit method,
    along with true and learned curvature metrics, using the pullback method.
    It supports optional alignment for spherical VAE models.

    Args:
        config: Configuration object containing settings and parameters.
        model: The model object used to compute learned curvature.
        recons: Torch tensor representing reconstruction points.
        latents: Torch tensor representing latent points.
        inputs: Torch tensor representing input points.
        labels: Torch tensor representing labels of the points.

    Returns:
        A dictionary containing metadata, points, and computed curvatures.
    """
    total_count = len(labels)
    target = min(getattr(config, "n_points_pullback_curv", total_count), total_count)
    n_pullback_points = int(np.sqrt(target)) ** 2
    n_pullback_points = max(1, n_pullback_points)

    curv_quadric_inputs = None
    curv_quadric_latents = None
    curv_quadric_recons = None
    z_grid = None
    curv_true = None
    curv_learned = None
    curv_learned_transformed = None

    # Only compute quadric curvature for Euclidean models
    if is_euclidean_model(getattr(config, "model_type", None)):
        if getattr(config, "compute_quadric_curv_inputs", False):
            curv_quadric_inputs = compute_quadric_curvature(config=config, labels=labels, points=inputs, k=config.k)
        if getattr(config, "compute_quadric_curv_latents", False):
            curv_quadric_latents = compute_quadric_curvature(config=config, labels=labels, points=latents, k=config.k)
        if getattr(config, "compute_quadric_curv_recons", False):
            curv_quadric_recons = compute_quadric_curvature(config=config, labels=labels, points=recons, k=config.k)
    elif getattr(config, "verbose", False) and (
            getattr(config, "compute_quadric_curv_inputs", False) or
            getattr(config, "compute_quadric_curv_latents", False) or
            getattr(config, "compute_quadric_curv_recons", False)
    ):
        print(
            f"Skipping quadric curvature: model_type={getattr(config, 'model_type', None)} "
            f"is non-Euclidean (Euclidean required)."
        )

    if getattr(config, "compute_true_curv", False):
        z_grid, _, curv_true = compute_curvature_true(config=config, n_grid_points=n_pullback_points)
    if getattr(config, "compute_learned_curv", False):
        if not is_euclidean_model(getattr(config, "model_type", None)):
            z_grid, _, curv_learned = compute_curvature_learned(
                config=config, model=model, n_grid_points=n_pullback_points
            )
            # Optional alignment for spherical Manifold-VAEs
            if getattr(config, "model_type", None) in {"VMFSphericalVAE", "SphericalAE"} and curv_learned is not None:
                curv_learned_transformed = compute_curvature_transformed(
                    learned_curvature=curv_learned, latents=latents, labels=labels, z_grid=z_grid
                )
        else:
            if getattr(config, "verbose", False):
                print(
                    f"Skipping learned curvature: model_type={getattr(config, 'model_type', None)} "
                    f"is Euclidean (non-Euclidean required)."
                )

    # Compute error metrics if available
    metrics = {}
    # (1) Quadric estimate on latents vs quadric estimate on inputs
    if (curv_quadric_latents is not None) and (curv_quadric_inputs is not None):
        try:
            mse_li = compute_curvature_error_mse(curv_quadric_latents, curv_quadric_inputs)
            smape_li = compute_curvature_error_smape(curv_quadric_latents, curv_quadric_inputs)
            metrics["quadric_inputs_vs_latents"] = {"mse": float(mse_li), "smape": float(smape_li)}
        except Exception as e:
            if getattr(config, "verbose", False):
                print(f"Warning: failed to compute metrics (latents vs inputs): {e}")
    # (2) True vs learned_rotated_sub (if available)
    if (curv_true is not None) and (curv_learned_transformed is not None):
        try:
            mse_tl = compute_curvature_error_mse(curv_true, curv_learned_transformed)
            smape_tl = compute_curvature_error_smape(curv_true, curv_learned_transformed)
            metrics["true_vs_learned_rotated_sub"] = {"mse": float(mse_tl), "smape": float(smape_tl)}
        except Exception as e:
            if getattr(config, "verbose", False):
                print(f"Warning: failed to compute metrics (true vs learned_rotated_sub): {e}")

    print(metrics)
    results_dict = {
        "experiment": getattr(config, "experiment", "unknown"),
        "points": {
            "labels": labels,
            "inputs": inputs,
            "latents": latents,
            "recons": recons,
        },
        "curvatures": {
            "inputs": curv_quadric_inputs ,
            "latents": curv_quadric_latents,
            "recons": curv_quadric_recons,
            "true": curv_true,
            "learned": curv_learned,
            "learned_transformed": curv_learned_transformed,
            "z_grid": z_grid,
        },
        "metrics": metrics,
    }

    # Save results_dict if requested
    if config.log_dir is not None:
        os.makedirs(config.log_dir, exist_ok=True)
        filename = f"vectors_curvatures_{getattr(config, 'experiment', 'exp')}.json"
        save_path = os.path.join(config.log_dir, filename)

        def _to_serializable(obj):
            if obj is None:
                return None
            # Native JSON types
            if isinstance(obj, (bool, int, float, str)):
                return obj
            # NumPy scalars
            if isinstance(obj, np.floating):
                return float(obj)
            if isinstance(obj, np.integer):
                return int(obj)
            # NumPy arrays
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            # Torch tensors
            if isinstance(obj, torch.Tensor):
                return obj.detach().cpu().tolist()
            # Dicts
            if isinstance(obj, dict):
                return {k: _to_serializable(v) for k, v in obj.items()}
            # Lists/Tuples
            if isinstance(obj, (list, tuple)):
                return [_to_serializable(v) for v in obj]
            # Fallback to string
            return str(obj)

        results_json = _to_serializable(results_dict)
        with open(save_path, "w") as f:
            json.dump(results_json, f, indent=2)
        if getattr(config, "verbose", False):
            print(f"Saved curvatures (JSON) to {save_path}")

    return results_dict


def compute_curvature_transformed(learned_curvature, latents, labels, z_grid):
    """
    Aligns the learned curvature values on a spherical or circular latent space
    by rotating the evaluation grid to match the orientation of the ground-truth
    labels. This is useful when the latent manifold exhibits rotational invariance,
    such as in Spherical VAEs or circular latent spaces, ensuring that curvature
    comparisons are not biased by arbitrary coordinate system choices.

    The function performs the following steps:
        1. Converts latent coordinates and label coordinates from angular
           representations to Cartesian vectors (2D or 3D, depending on the latent dimension).
        2. Computes an optimal rotation matrix $R$ using the orthogonal Procrustes
           problem to align the latent points to the label points.
        3. Rotates the grid of evaluation points and identifies the nearest
           neighbors in the original (unrotated) grid.
        4. Assigns the learned curvature values from the nearest neighbors
           to produce a curvature field aligned to the label orientation.

    Args:
        learned_curvature (torch.Tensor):
            Tensor of shape $(N,)$ or $(N,d)$ containing the curvature values
            computed on the grid points $z_{grid}$.
        latents (torch.Tensor):
            Tensor of shape $(M,2)$ or $(M,3)$ representing the latent
            coordinates of the data points in angular coordinates (e.g.,
            angle for 2D circle or $(\theta,\phi)$ for 3D sphere).
        labels (torch.Tensor):
            Tensor of shape $(M,2)$ or $(M,3)$ providing the true angular
            coordinates of the data points.
        z_grid (torch.Tensor):
            Tensor of shape $(N,1)$ for 2D or $(N,2)$ for 3D containing
            the angular grid points on which the curvature was computed.

    Returns:
        torch.Tensor:
            Rotated curvature tensor of the same shape as `learned_curvature`,
            where each entry corresponds to the curvature at the rotated grid
            point matched to its nearest original grid neighbor.

    Raises:
        NotImplementedError:
            If the latent dimension is not 2 or 3.
    """
    latents = latents.to(torch.float32)
    labels = labels.to(torch.float32)
    z_grid = z_grid.to(torch.float32)

    def angle_to_cartesian(theta):
        x = torch.cos(theta)
        y = torch.sin(theta)
        return torch.stack([x, y], dim=1).to(torch.float32)

    def spherical_to_cartesian(theta, phi):
        x = torch.sin(theta) * torch.cos(phi)
        y = torch.sin(theta) * torch.sin(phi)
        z = torch.cos(theta)
        return torch.stack([x, y, z], dim=1).to(torch.float32)

    def compute_rotation_matrix(source, target):
        A = source.T @ target
        U, _, Vt = torch.svd(A)
        R = Vt @ U.T
        # # Ensure det(R)=+1
        # if torch.det(R) < 0:
        #     Vt[-1, :] *= -1
        #     R = Vt @ U.T
        return R

    # Compute rotation
    if latents.shape[1] == 2:
        theta_grid = z_grid[:, 0] if z_grid.ndim > 1 else z_grid
        vecs = angle_to_cartesian(theta_grid)

        label_angles = labels[:, 0] if labels.ndim > 1 else labels
        label_cartesian = angle_to_cartesian(label_angles)
    elif latents.shape[1] == 3:
        vecs = spherical_to_cartesian(z_grid[:, 0], z_grid[:, 1])
        label_cartesian = spherical_to_cartesian(labels[:, 0], labels[:, 1])
    else:
        raise NotImplementedError

    R = compute_rotation_matrix(latents, label_cartesian)
    vecs_rotated = vecs @ R
    tree = cKDTree(vecs.numpy())
    _, nn_indices = tree.query(vecs_rotated.numpy(), k=1)
    rotated_curvature = learned_curvature[nn_indices]

    return rotated_curvature
