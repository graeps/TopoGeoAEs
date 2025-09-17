import os
import numpy as np
import torch

from .pullback_curvature import compute_curvature_true, compute_curvature_learned
from .quadric_curvature import compute_quadric_curvature

os.environ["GEOMSTATS_BACKEND"] = "pytorch"

from scipy.spatial import cKDTree


def compute_all_curvatures(config, model, recons, latents, inputs, labels, save_dir="./curvatures"):
    """
    Compute a set of curvature quantities on full data and a curated subset,
    then return legacy tuple outputs and persist a structured bundle for easy access.

    Returns (unchanged for backward compatibility):
      points_sub: (inputs_sub, latents_sub, recons_sub)
      curvatures_sub: (labels_sub, curv_true, curv_learned, curv_learned_rotated, z_grid)
      curvatures_emp_full: (labels, curv_in, curv_lat, curv_rec)
      points: (inputs, latents, recons)

    Additionally, a structured 'bundle' with named keys is saved to disk.
    """
    # Determine subset size and indices (square number for grid-friendly plots)
    total_count = len(labels)
    target = min(getattr(config, "n_points_pullback_curv", total_count), total_count)
    sub_count = int(np.sqrt(target)) ** 2
    sub_count = max(1, sub_count)  # ensure at least one sample
    sampled_indices = np.random.choice(total_count, size=sub_count, replace=False)
    sampled_indices.sort()

    # Empirical (quadric) curvature on full data with minimal branching
    def _zeros_full():
        return np.zeros(total_count)

    compute_flags = {
        "inputs": getattr(config, "compute_quadric_curv_inputs", False),
        "latents": getattr(config, "compute_quadric_curv_latents", False),
        "recons": getattr(config, "compute_quadric_curv_recons", False),
    }
    curv_emp_full = {
        "inputs": compute_quadric_curvature(config=config, labels=labels, points=inputs, k=config.k)
        if compute_flags["inputs"] else _zeros_full(),
        "latents": compute_quadric_curvature(config=config, labels=labels, points=latents, k=config.k)
        if compute_flags["latents"] else _zeros_full(),
        "recons": compute_quadric_curvature(config=config, labels=labels, points=recons, k=config.k)
        if compute_flags["recons"] else _zeros_full(),
    }

    # True curvature and grid (defined even if not computed)
    z_grid = torch.zeros(sub_count)
    curv_true = np.zeros(sub_count)
    if getattr(config, "compute_true_curv", False):
        z_res = compute_curvature_true(config=config, n_grid_points=sub_count)
        # Accept either (z_grid, _, curv_true) or any compatible tuple length
        if isinstance(z_res, (list, tuple)):
            if len(z_res) >= 3:
                z_grid, _, curv_true = z_res[-3], z_res[-2], z_res[-1]
            elif len(z_res) == 2:
                z_grid, curv_true = z_res
            else:
                # Only z_grid returned, keep defaults
                z_grid = z_res[0]
        else:
            # Unexpected non-iterable result; keep defaults
            pass

    # Learned curvature (robust to varying return signatures)
    curv_learned = np.zeros(sub_count)
    curv_learned_rotated = np.zeros(sub_count)
    if getattr(config, "compute_learned_curv", False):
        learned_res = compute_curvature_learned(config=config, model=model, n_grid_points=sub_count)
        if isinstance(learned_res, (list, tuple)) and len(learned_res) > 0:
            curv_learned = learned_res[-1]
        else:
            curv_learned = learned_res  # assume it's already the curvature array

        # Optional alignment for spherical models
        if getattr(config, "model_type", None) in {"VMFSphericalVAE", "SphericalAE"}:
            curv_learned_rotated = compute_curvature_rotated(
                learned_curvature=curv_learned, latents=latents, z_grid=z_grid
            )
        else:
            curv_learned_rotated = curv_learned

    # Legacy tuple outputs (unchanged)
    curvatures_sub = (curv_true, curv_learned, curv_learned_rotated, z_grid)
    curvatures_emp_full = (labels, curv_emp_full["inputs"], curv_emp_full["latents"], curv_emp_full["recons"])
    points = (inputs, latents, recons)

    # Structured bundle for easy, named access after saving
    bundle = {
        "metadata": {
            "experiment": getattr(config, "experiment", "unknown"),
            "sampled_indices": sampled_indices,
            "total_count": int(total_count),
            "sub_count": int(sub_count),
        },
        "labels": {
            "full": labels,
            "sub": labels_sub,
        },
        "points": {
            "full": {"inputs": inputs, "latents": latents, "recons": recons},
            "sub": {"inputs": inputs_sub, "latents": latents_sub, "recons": recons_sub},
        },
        "curvatures": {
            "empirical_full": {
                "inputs": curv_emp_full["inputs"],
                "latents": curv_emp_full["latents"],
                "recons": curv_emp_full["recons"],
            },
            "true_sub": curv_true,
            "learned_sub": curv_learned,
            "learned_rotated_sub": curv_learned_rotated,
            "z_grid": z_grid,
        },
    }

    # Persist both legacy tuples and the structured bundle
    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
        filename = f"curvatures_{getattr(config, 'experiment', 'exp')}.pt"
        save_path = os.path.join(save_dir, filename)

        torch.save(
            {
                "bundle": bundle,  # new, structured and self-descriptive
                "curvatures_sub": curvatures_sub,  # legacy
                "curvatures_emp_full": curvatures_emp_full,  # legacy
                "points_sub": points_sub,  # legacy
                "points": points,  # legacy
            },
            save_path,
        )

        if getattr(config, "verbose", False):
            print(f"Saved curvatures to {save_path}")

    return points_sub, curvatures_sub, curvatures_emp_full, points


def compute_curvature_rotated(learned_curvature, latents, labels, z_grid):
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
