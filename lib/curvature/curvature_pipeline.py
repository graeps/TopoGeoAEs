import os
import numpy as np
import torch

from .pullback_curvature import compute_curvature_true, compute_curvature_learned
from .quadric_curvature import compute_empirical_curvature

os.environ["GEOMSTATS_BACKEND"] = "pytorch"

from scipy.spatial import cKDTree


def compute_all_curvatures(config, model, recons, latents, inputs, labels, save_dir="./curvatures"):
    # Compute pullback curvature on (ordered) subset of points to reduce computation time
    n_total = len(labels)
    n_points = int(np.sqrt(min(config.n_points_pullback_curv, n_total))) ** 2
    sampled_indices = np.random.choice(n_total, size=n_points, replace=False)
    sampled_indices.sort()

    # Subsample data to match subset of curvatures for heat map plotting
    labels_sub = labels[sampled_indices]
    inputs_sub = inputs[sampled_indices]
    latents_sub = latents[sampled_indices]
    recons_sub = recons[sampled_indices]

    # Compute empirical curvatures on full data
    if config.compute_emp_curv:
        curv_in, curv_lat, curv_rec, labels = compute_empirical_curvature(config=config, labels=labels, inputs=inputs,
                                                                          latents=latents, recons=recons, k=config.k
                                                                          )
        norm_factor = np.max(curv_in[sampled_indices]) / np.max(curv_lat[sampled_indices])
        curv_lat_norm = curv_lat * norm_factor
        curv_lat_norm_sub = curv_lat_norm[sampled_indices]
        # Subsample empirical curvature to match subset for error computation
        curv_in_sub = curv_in[sampled_indices]
        curv_rec_sub = curv_rec[sampled_indices]
        curv_lat_sub = curv_lat[sampled_indices]
    else:
        curv_in_sub = np.zeros(n_points)
        curv_rec_sub = np.zeros(n_points)
        curv_lat_sub = np.zeros(n_points)
        curv_lat_norm_sub = np.zeros(n_points)
        curv_lat_norm = np.zeros(len(labels))
        curv_in = np.zeros(len(labels))
        curv_lat = np.zeros(len(labels))
        curv_rec = np.zeros(len(labels))
    if config.compute_true_curv:
        z_grid, _, curv_true = compute_curvature_true(config=config, labels=labels_sub, n_grid_points=n_points)
    else:
        curv_true = np.zeros(n_points)
        z_grid = np.zeros(n_points)
    if config.compute_learned_curv:
        _, _, _, curv_learned = compute_curvature_learned(config=config, model=model, latents=latents_sub,
                                                          labels=labels_sub, n_grid_points=n_points)
        if config.model_type in {"VMFSphericalVAE", "SphericalAE"}:
            curv_learned_rotated = compute_curvature_rotated(learned_curvature=curv_learned, latents=latents_sub,
                                                             labels=labels_sub, z_grid=z_grid)
        else:
            curv_learned_rotated = curv_learned
    else:
        curv_learned = np.zeros(n_points)
        curv_learned_rotated = np.zeros(n_points)

    curvatures_sub = (labels_sub, curv_in_sub, curv_rec_sub, curv_lat_sub, curv_lat_norm_sub, curv_true,
                      curv_learned, curv_learned_rotated, z_grid)
    curvatures_emp_full = (labels, curv_in, curv_lat, curv_lat_norm, curv_rec)

    points_sub = (inputs_sub, latents_sub, recons_sub)

    points = (inputs, latents, recons)

    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
        filename = f"curvatures_{config.experiment}.pt"
        save_path = os.path.join(save_dir, filename)

        torch.save({
            'curvatures_sub': curvatures_sub,
            'curvatures_emp_full': curvatures_emp_full,
            'points_sub': points_sub,
            'points': points,
        }, save_path)

        if config.verbose:
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
