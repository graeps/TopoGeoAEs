import json
import os
import time

import matplotlib as mpl
from random import sample
import pandas as pd
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from scipy.interpolate import griddata, RBFInterpolator

from .eval_curvature import compute_curvature_learned, compute_curvature_error, \
    compute_curvature_true, estimate_curvature_1d_quadric, \
    compute_all_curvatures, get_vectors, compute_curvature_error_mse, compute_curvature_error_linf, \
    compute_curvature_error_smape
from .eval_topology import compare_persistent_homology

import plotly.graph_objs as go
from plotly.subplots import make_subplots


def show_training_history(config, history):
    loss_keys = [key.replace('train_', '') for key in history.keys() if key.startswith('train_')]
    unique_losses = sorted(set(loss_keys))

    # Plot setup
    n_losses = len(unique_losses)
    _, axs = plt.subplots(figsize=(4 * n_losses, 4), ncols=n_losses)

    if n_losses == 1:
        axs = [axs]

    for i, loss_name in enumerate(unique_losses):
        train_key = f'train_{loss_name}'
        test_key = f'test_{loss_name}'
        axs[i].plot(history[train_key], color='orange', label='train')
        axs[i].plot(history[test_key], color='blue', label='val')
        axs[i].set_xlabel('epoch')
        axs[i].set_ylabel('loss')
        axs[i].set_title(f'{loss_name.replace("_", " ").title()} History')
        axs[i].legend()

    if config.log_dir is not None:
        os.makedirs(config.log_dir, exist_ok=True)
        plt.savefig(os.path.join(config.log_dir, "training_history.png"))

    plt.show()


def plot_on_torus(latent_vars):
    # Torus parameters
    major_radius = 2
    minor_radius = 1

    # Extract the components from latentvars
    cos_theta = latent_vars[:, 0]
    sin_theta = latent_vars[:, 1]
    cos_phi = latent_vars[:, 2]
    sin_phi = latent_vars[:, 3]

    cos_theta, sin_theta = cos_theta / torch.sqrt(cos_theta ** 2 + sin_theta ** 2), sin_theta / torch.sqrt(
        cos_theta ** 2 + sin_theta ** 2)
    cos_phi, sin_phi = cos_phi / torch.sqrt(cos_phi ** 2 + sin_phi ** 2), sin_phi / torch.sqrt(
        cos_phi ** 2 + sin_phi ** 2)

    # Compute the coordinates on the torus
    x = (major_radius - minor_radius * cos_theta) * cos_phi
    y = (major_radius - minor_radius * cos_theta) * sin_phi
    z = minor_radius * sin_theta

    # Create a base torus for visualization
    u = np.linspace(0, 2 * np.pi, 100)
    v = np.linspace(0, 2 * np.pi, 100)
    u, v = np.meshgrid(u, v)
    base_x = (major_radius + minor_radius * np.cos(v)) * np.cos(u)
    base_y = (major_radius + minor_radius * np.cos(v)) * np.sin(u)
    base_z = minor_radius * np.sin(v)

    # Compute colors for the surface based on proximity to latent points
    colors = np.zeros_like(base_x)
    latent_points = torch.stack((x, y, z), dim=1).numpy()

    for i in range(base_x.shape[0]):
        for j in range(base_x.shape[1]):
            point = np.array([base_x[i, j], base_y[i, j], base_z[i, j]])
            distances = np.linalg.norm(latent_points - point, axis=1)
            colors[i, j] = np.min(distances)

    colors = 1 - colors / np.max(colors)  # Normalize colors to range [0, 1]

    # Plot the torus points and surface
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    surface = ax.plot_surface(base_x, base_y, base_z, facecolors=plt.cm.get_cmap("viridis")(colors), rstride=1,
                              cstride=1, alpha=0.8, edgecolor='none')

    # Add color bar
    m = plt.cm.ScalarMappable(cmap=plt.cm.get_cmap("viridis"))
    m.set_array(colors)
    fig.colorbar(m, ax=ax, shrink=0.5, aspect=10)

    # Set labels and aspect
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title("3D Torus with Colored Surface")
    plt.show()


def plot_test_latents_on_torus(model, test_loader, device="cpu"):
    """
    Encodes test dataset samples into the toroidal latent space and plots them.

    Args:
        model (torch.nn.Module): Trained VAE model.
        test_loader (torch.utils.data.DataLoader): DataLoader for the test dataset.
        device (torch.device): Device to run computations on ('cuda' or 'cpu').
    """
    model.eval()
    latent_vars = []

    with torch.no_grad():
        for x, _ in test_loader:
            x = x.to(device)
            posterior_params = model.encode(x)
            z = model.reparameterize(posterior_params)
            latent_vars.append(z)

    latent_vars = torch.cat(latent_vars, dim=0)
    plot_on_torus(latent_vars)


def plot_latents_ae_3d(model, test_loader, device="cpu"):
    model.eval()
    latent_vars = []

    with torch.no_grad():
        for x in test_loader:
            x = x[0].to(device)
            z, _, _, _ = model(x)
            if z.shape[1] != 4:
                raise ValueError("plotting in 3d only for latent space S^1xS^1")
            latent_vars.append(z)

    latent_vars = torch.cat(latent_vars, dim=0)
    plot_on_torus(latent_vars)


def plot_latent_projections(model, pointcloud, test_loader, device="cpu"):
    model.eval()
    latent_vars = []
    latent_angles = []
    x_reconstructions = []

    d = model.latent_dim
    d = d // 2 if model.type == "euclidean_ae" else d

    with torch.no_grad():
        for x in test_loader:
            x = x[0].to(device)
            if model.type == "euclidean_ae":
                z, x_recon = model(x)
            elif model.type == "shape_toroidal_ae":
                z, x_recon, _, theta = model(x)
                latent_angles.append(theta)
            else:
                raise ValueError(f"Unknown model type: {model.type}")
            # x_recon = torch.where(x_recon.abs() < 1e-10, torch.zeros_like(x_recon), x_recon)
            x_reconstructions.append(x_recon[:, :2 * d])
            latent_vars.append(z)

    latent_vars = torch.cat(latent_vars, dim=0)
    latent_angles = torch.cat(latent_angles, dim=0) if latent_angles else None
    x_reconstructions = torch.cat(x_reconstructions, dim=0) if x_reconstructions else None

    latent_x = latent_vars[:, :d]  # First d columns (cos components)
    latent_y = latent_vars[:, d:]  # Last d columns (sin components)

    x_recon_x = x_reconstructions[:, :d]
    x_recon_y = x_reconstructions[:, d:]

    pointcloud_x = pointcloud[:, :d]
    pointcloud_y = pointcloud[:, d:]

    if d == 1:
        fig, axes = plt.subplots(1, 4, figsize=(15, 4))

        # Pointcloud on the circle
        axes[0].scatter(pointcloud_x, pointcloud_y, s=1, color='orange')
        axes[0].set_title("Pointcloud, not embedded")
        axes[0].set_xlabel("cos(θ)")
        axes[0].set_ylabel("sin(θ)")
        axes[0].autoscale()
        axes[0].set_aspect('equal')
        axes[0].grid(True, linestyle='--', alpha=0.5)

        # Latent variables
        axes[1].scatter(latent_x, latent_y, s=1)
        axes[1].set_title("Latent Coords")
        axes[1].set_xlabel("cos(θ)")
        axes[1].set_ylabel("sin(θ)")
        # axes[1].set_xlim([-1.1, 1.1])
        # axes[1].set_ylim([-1.1, 1.1])
        axes[1].autoscale()
        axes[2].set_aspect('equal', adjustable='box')
        axes[1].grid(True, linestyle='--', alpha=0.5)

        # x reconstructed
        axes[2].scatter(x_recon_x, x_recon_y, s=1, color='orange')
        axes[2].set_title("x-recon - translation")
        axes[2].set_xlabel("x")
        axes[2].set_ylabel("y")
        axes[2].autoscale()
        axes[2].set_aspect('equal', adjustable='box')
        axes[2].grid(True, linestyle='--', alpha=0.5)

        # Latent angles on a line
        if latent_angles is not None:
            axes[3].scatter(latent_angles, np.zeros_like(latent_angles), s=1)
            axes[3].set_title("Latent Angles")
            axes[3].set_xlabel("θ")
            axes[3].set_yticks([])
            axes[3].set_xlim([-np.pi, np.pi])
            axes[3].grid(True, linestyle='--', alpha=0.5)

        plt.show()

    else:
        for i in range(d):
            fig, axes = plt.subplots(1, 3, figsize=(10, 4))

            # Pointcloud projection
            axes[0].scatter(pointcloud_x[:, i], pointcloud_y[:, i], s=1, color='orange')
            axes[0].set_title(f'Pointcloud Projection {i + 1}')
            axes[0].set_xlabel("x")
            axes[0].set_ylabel("y")
            axes[0].set_xlim([-10, 10])
            axes[0].set_ylim([-10, 10])
            axes[0].set_aspect('equal', adjustable='box')
            axes[0].grid(True, linestyle='--', alpha=0.5)

            # Latent projection
            axes[1].scatter(latent_x[:, i], latent_y[:, i], s=1)
            axes[1].set_title(f'Latent Projection {i + 1}')
            axes[1].set_xlabel("cos(θ)")
            axes[1].set_ylabel("sin(θ)")
            axes[1].set_aspect('equal', adjustable='datalim')
            axes[1].autoscale()
            axes[1].grid(True, linestyle='--', alpha=0.5)

            # x reconstructed
            axes[2].scatter(x_recon_x[:, i], x_recon_y[:, i], s=1, color='orange')
            axes[2].set_title("x-recon - translation")
            axes[2].set_xlabel("x")
            axes[2].set_ylabel("y")
            axes[2].autoscale()
            axes[2].set_aspect('equal', adjustable='box')
            axes[2].grid(True, linestyle='--', alpha=0.5)

            plt.show()

    # Latent angles for d=2
    if model.type == "shape_toroidal_ae" and d == 2 and latent_angles is not None:
        fig, ax = plt.subplots(figsize=(5, 5))
        theta1 = latent_angles[:, 0]
        theta2 = latent_angles[:, 1]
        ax.scatter(theta1, theta2, s=1)
        ax.set_title('Latent Angles')
        ax.set_xlabel("θ_1")
        ax.set_ylabel("θ_2")
        ax.set_aspect('equal', adjustable='datalim')
        ax.autoscale()
        ax.grid(True, linestyle='--', alpha=0.5)
        plt.show()


def _scatter_datapoints(ax, data, title, colors=None, cmap='hsv', pca_dim=3):
    d = data.shape[1]
    pca_applied = False
    dot_size = 2

    if d == 1:
        sc = ax.scatter(data[:, 0], np.zeros_like(data[:, 0]), c=colors, cmap=cmap, s=dot_size, alpha=0.7)
        ax.set_yticks([])
    elif d == 2:
        sc = ax.scatter(data[:, 0], data[:, 1], c=colors, cmap=cmap, s=dot_size, alpha=0.7)
    elif d == 3:
        sc = ax.scatter(data[:, 0], data[:, 1], data[:, 2], c=colors, cmap=cmap, s=dot_size, alpha=0.7)
    else:
        data = PCA(n_components=pca_dim).fit_transform(data)
        if pca_dim == 2:
            sc = ax.scatter(data[:, 0], data[:, 1], c=colors, cmap=cmap, s=dot_size, alpha=0.7)
        else:
            sc = ax.scatter(data[:, 0], data[:, 1], data[:, 2], c=colors, cmap=cmap, s=dot_size, alpha=0.7)
        pca_applied = True

    title_suffix = " (PCA)" if pca_applied else ""
    ax.set_title(f"{title}{title_suffix}")
    ax.set_aspect('equal', adjustable='datalim')

    return sc


def plot_data_latents_recon(config, model, data_loader):
    recons, latents, inputs, labels = get_vectors(config, model, data_loader, config.n_plot_points)
    if torch.isnan(latents).any():
        print("NaNs detected in the dataset!")

    if labels.ndim > 1 and labels.shape[1] == 2:
        colors = labels[:, 0]
        color_map = "hsv"
    elif config.dataset_name in {"nested_spheres", "nested_spheres_high_dim", "interlocked_tori", "interlocked_tubes"}:
        tab10_colors = plt.cm.get_cmap('tab10', 10)
        colors = np.array([tab10_colors(int(label)) for label in labels[:, 0]])
        color_map = None
    else:
        colors = labels.squeeze()
        color_map = "viridis"

    fig = plt.figure(figsize=(18, 5))

    if config.dataset_name == "s1_synthetic":
        pca_dim = 2
    else:
        pca_dim = 3
    # Dataset plot
    ax1 = fig.add_subplot(1, 3, 1, projection='3d' if inputs.shape[1] >= 3 and pca_dim == 3 else None)
    _scatter_datapoints(ax1, inputs, "Input Data", colors, cmap=color_map, pca_dim=pca_dim)

    # Latent space plot
    ax2 = fig.add_subplot(1, 3, 2, projection='3d' if inputs.shape[1] >= 3 and pca_dim == 3 else None)
    _scatter_datapoints(ax2, latents, "Latent Space", colors, cmap=color_map, pca_dim=pca_dim)

    # Reconstruction plot
    ax3 = fig.add_subplot(1, 3, 3, projection='3d' if inputs.shape[1] >= 3 and pca_dim == 3 else None)
    _scatter_datapoints(ax3, recons, "Reconstructed Data", colors, cmap=color_map, pca_dim=pca_dim)

    plt.tight_layout()

    if config.log_dir is not None:
        save_path = os.path.join(config.log_dir, "input_latents_recons_plot.png")
        plt.savefig(save_path)

    plt.show()


def curvature_compute_plot_vm(config, model, test_loader):
    """Compute and plot curvature results."""
    all_data = []
    all_labels = []

    for data, labels in test_loader:
        all_data.append(data)
        all_labels.append(labels)

    all_data = torch.cat(all_data)
    all_labels = torch.cat(all_labels)
    # Compute
    print("Computing learned curvature...")
    z_grid, _, _, curv_norms_learned = compute_curvature_learned(
        model=model,
        config=config,
        n_grid_points=config.n_grid_points,
    )

    curv_norm_learned_profile = pd.DataFrame(
        {
            "curv_norm_learned": curv_norms_learned,
        }
    )
    if config.dataset_name in (
            "s1_synthetic",
    ):
        curv_norm_learned_profile["z_grid"] = z_grid
    elif config.dataset_name in ("s2_synthetic", "t2_synthetic"):
        curv_norm_learned_profile["z_grid_theta"] = z_grid[:, 0]
        curv_norm_learned_profile["z_grid_phi"] = z_grid[:, 1]

    norm_val = None
    if config.dataset_name in ("s1_synthetic", "s2_synthetic", "t2_synthetic"):
        print("Computing true curvature for synthetic data...")
        z_grid, _, curv_norms_true = compute_curvature_true(config, n_grid_points=config.n_grid_points)
        print("Computing curvature error for synthetic data...")

        curvature_error = compute_curvature_error(
            z_grid, curv_norms_learned, curv_norms_true, config
        )
        norm_val = max(curv_norms_true)

        curv_norm_true_profile = pd.DataFrame(
            {
                "curv_norm_true": curv_norms_true,
            }
        )

        if config.dataset_name == "s1_synthetic":
            curv_norm_true_profile["z_grid"] = z_grid
        else:
            curv_norm_true_profile["z_grid_theta"] = z_grid[:, 0]
            curv_norm_true_profile["z_grid_phi"] = z_grid[:, 1]

    # Plot
    fig_curv_norms_learned = plot_curvature_norms(
        angles=z_grid,
        curvature_norms=curv_norms_learned,
        config=config,
        norm_val=norm_val,
        profile_type="learned",
    )
    if config.dataset_name in ("s1_synthetic", "s2_synthetic", "t2_synthetic"):
        fig_curv_norms_true = plot_curvature_norms(
            angles=z_grid,
            curvature_norms=curv_norms_true,
            config=config,
            norm_val=None,
            profile_type="true",
        )

    if config.dataset_name in ("s1_synthetic", "experimental", "three_place_cells_synthetic",):
        fig_neural_manifold_learned = plot_neural_manifold_learned(
            curv_norm_learned_profile=curv_norm_learned_profile,
            config=config,
            labels=all_labels,
        )


def curvature_compute_plot_euclidean(config, model):
    latent_vectors, labels, _, curv_norms_learned = compute_curvature_learned(
        model=model,
        config=config,
        n_grid_points=config.n_grid_points,
    )
    z_grid, _, curv_norms_true_grid = compute_curvature_true(config, n_grid_points=config.n_grid_points)
    labels, _, curv_norms_true_latents = compute_curvature_true(config, labels.squeeze())

    fig, axs = plt.subplots(2, 2, figsize=(10, 7))

    # Top-left: learned curvature
    sc1 = axs[0, 0].scatter(latent_vectors[:, 0], latent_vectors[:, 1], c=curv_norms_learned, cmap='hsv', s=2)
    fig.colorbar(sc1, ax=axs[0, 0], label='Curvature')
    axs[0, 0].set_title("Learned curvature heatmap")
    axs[0, 0].set_xlabel("z₁")
    axs[0, 0].set_ylabel("z₂")

    # Top-right: true curvature on latent points
    sc2 = axs[0, 1].scatter(latent_vectors[:, 0], latent_vectors[:, 1], c=curv_norms_true_latents, cmap='hsv', s=2)
    fig.colorbar(sc2, ax=axs[0, 1], label='Curvature')
    axs[0, 1].set_title("True curvature heatmap")
    axs[0, 1].set_xlabel("z₁")
    axs[0, 1].set_ylabel("z₂")

    # Bottom-left: curvature over angle
    axs[1, 0].plot(z_grid, curv_norms_true_grid, linewidth=2)
    axs[1, 0].set_xlabel("angle", fontsize=14)
    axs[1, 0].set_ylabel("mean curvature norm", fontsize=14)
    axs[1, 0].set_title("Curvature vs angle")

    # Bottom-right: polar plot
    polar_ax = fig.add_subplot(2, 2, 4, projection="polar")
    polar_ax.scatter(
        z_grid,
        np.ones_like(z_grid),
        c=curv_norms_true_grid,
        s=100,
        cmap="hsv",
        linewidths=0,
    )
    polar_ax.set_yticks([])
    polar_ax.set_title("Polar curvature")

    plt.tight_layout()
    plt.show()


def plot_curvature_norms(angles, curvature_norms, config, norm_val, profile_type):
    fig = plt.figure(figsize=(8, 4))  # smaller figure
    colormap = plt.get_cmap("hsv")

    if norm_val is not None:
        color_norm = mpl.colors.Normalize(0.0, norm_val)
    else:
        color_norm = mpl.colors.Normalize(0.0, max(curvature_norms))

    if config.dataset_name in {"s1_synthetic", "interlocking_rings_synthetic", "scrunchy_synthetic"}:
        ax1 = fig.add_subplot(121)
        ax1.plot(angles, curvature_norms, linewidth=2)
        ax1.set_xlabel("angle", fontsize=12)
        ax1.set_ylabel("mean curvature norm", fontsize=12)

        ax2 = fig.add_subplot(122, projection="polar")
        sc = ax2.scatter(
            angles,
            np.ones_like(angles),
            c=curvature_norms,
            s=50,  # smaller markers
            cmap=colormap,
            norm=color_norm,
            linewidths=0,
        )
        ax2.set_yticks([])
        ax2.set_xlabel("angle", fontsize=12)

        ax1.set_title(f"{profile_type} profile", fontsize=14)
        ax2.set_title(f"{profile_type} profile", fontsize=14)

    elif config.dataset_name in {"s2_synthetic", "t2_synthetic"}:
        ax = fig.add_subplot(111, projection="3d")

        if config.dataset_name == "s2_synthetic":
            x = config.radius * [np.sin(a[0]) * np.cos(a[1]) for a in angles]
            y = config.radius * [np.sin(a[0]) * np.sin(a[1]) for a in angles]
            z = config.radius * [np.cos(a[0]) for a in angles]
        else:  # t2_synthetic
            x = [(config.major_radius - config.minor_radius * np.cos(a[0])) * np.cos(a[1]) for a in angles]
            y = [(config.major_radius - config.minor_radius * np.cos(a[0])) * np.sin(a[1]) for a in angles]
            z = [config.minor_radius * np.sin(a[0]) for a in angles]

        sc = ax.scatter3D(x, y, z, s=50, c=curvature_norms, cmap="Spectral", norm=color_norm)
        plt.colorbar(sc, ax=ax, shrink=0.6)
        ax.set_title(f"{profile_type} profile", fontsize=14)

        if config.dataset_name == "t2_synthetic":
            r = config.major_radius + config.minor_radius
            ax.set_xlim(-r, r)
            ax.set_ylim(-r, r)
            ax.set_zlim(-r, r)
            plt.axis("off")

    plt.tight_layout()
    return fig


def scatter_curvature_heatmaps(config, points, points_sub, curv_true, curv_in, curv_rec, curv_lat,
                               curv_learned, entity=None):
    inputs, latents, recons = points
    inputs_sub, latents_sub = points_sub

    fig = plt.figure(figsize=(18, 10))
    color_map = 'rainbow'

    def plot_heatmap_group(heatmaps, suptitle, tag):
        num_heatmaps = len(heatmaps)
        num_cols = 3  # Maximum number of plots per row
        num_rows = (num_heatmaps + num_cols - 1) // num_cols  # Calculate number of rows required

        fig = plt.figure(figsize=(18, 6 * num_rows))
        for i, (title, (curv, points)) in enumerate(heatmaps.items(), 1):
            ax = fig.add_subplot(num_rows, num_cols, i,
                                 projection='3d' if points.shape[1] >= 3 else None)
            sc = _scatter_datapoints(ax=ax, data=points, title=title, colors=curv, cmap=color_map)
            fig.colorbar(sc, ax=ax, shrink=0.7)
        fig.suptitle(suptitle, fontsize=16)
        plt.tight_layout(rect=[0, 0, 1, 0.96])  # Adjust layout to make room for the suptitle

        if config.log_dir is not None:
            fname = f"heatmap_{tag}.png"
            plt.savefig(os.path.join(config.log_dir, fname))

        plt.show()

    heatmaps = {
        "Empirical Curvature on Inputs": (curv_in, inputs),
        "Empirical Curvature on Latents": (curv_lat, latents),
    }

    if config.compute_rec_curv:
        heatmaps["Empirical Curvature on Reconstructed Data"] = (curv_rec, recons)
    if config.compute_true_curv:
        heatmaps["True Curvature on Inputs"] = (curv_true, inputs_sub)
    if config.compute_learned_curv:
        heatmaps["Pullback Curvature on Latents"] = (curv_learned, latents_sub)

    if entity is not None:
        plot_heatmap_group(heatmaps, f'Curvature Heatmap Comparison - Connected Components {int(entity)}',
                           'curvature_heatmaps')
    else:
        plot_heatmap_group(heatmaps, 'Curvature Heatmap Comparison', 'curvature_heatmaps')


def plot_curvatures_1d(labels, curvature_true, curvature_inputs, curvature_recons,
                       curvature_latents, curvature_latents_normalized, curvature_learned, config):
    curve_groups = [
        [
            ('True Curvature', curvature_true, 'tab:green'),
            ('Input Curvature', curvature_inputs, 'tab:blue'),
            ('Reconstructed Curvature', curvature_recons, 'tab:red')
        ],
        [
            ('True Curvature', curvature_true, 'tab:green'),
            ('Learned Curvature', curvature_learned, 'tab:pink'),
            ('Latent Curvature', curvature_latents, 'tab:orange')
        ],
        [
            ('True Curvature', curvature_true, 'tab:green'),
            ('Input Curvature', curvature_inputs, 'tab:blue'),
            ('Normalized Learned Curvature', curvature_latents_normalized, 'tab:orange')
        ]
    ]
    titles = ['True, Input, Reconstructed', 'True, Learned, Latent', 'True, Normalized Learned, Latent + Learned']

    fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharex=True)
    for ax, curves, title in zip(axes, curve_groups, titles):
        for label, curve, color in curves:
            ax.plot(labels, curve, label=label, color=color, linewidth=1.5, alpha=0.8)
        ax.set_title(title)
        ax.set_xlabel('Angle (radians)')
        ax.grid(True, linestyle='--', alpha=0.5)
        ax.legend(fontsize=8)
    axes[0].set_ylabel('Curvature')
    plt.tight_layout()
    if config.log_dir is not None:
        plt.savefig(os.path.join(config.log_dir, "curvature_grid_plot.png"))
    plt.show()


def plot_curvatures_2d(labels, labels_sub, curv_true, curv_in, curv_rec, curv_lat, curv_learned, config, entity=None):
    grid_res = 100
    if config.dataset_name in {"sphere", "nested_spheres", "nested_spheres_high_dim"}:
        grid_x, grid_y = np.meshgrid(np.linspace(0, np.pi, int(grid_res // 2)), np.linspace(0, 2 * np.pi, grid_res))
    else:
        grid_x, grid_y = np.meshgrid(np.linspace(0, 2 * np.pi, grid_res), np.linspace(0, 2 * np.pi, grid_res))

    def interpolate(lbls, values):
        return griddata(lbls, values, (grid_x, grid_y), method="cubic")

    def plot_surface_group(surfaces, suptitle, tag):
        num_surfaces = len(surfaces)
        num_cols = 3  # Maximum number of plots per row
        num_rows = (num_surfaces + num_cols - 1) // num_cols  # Calculate number of rows required

        fig = plt.figure(figsize=(18, 6 * num_rows))  # Adjust the figure height based on the number of rows

        # Iterate over the surfaces and create subplots
        for i, (title, surface) in enumerate(surfaces.items(), 1):
            ax = fig.add_subplot(num_rows, num_cols, i, projection='3d')
            surf = ax.plot_surface(grid_x, grid_y, surface, cmap='viridis')
            ax.set_title(title)
            ax.set_xlabel(r'$\theta_1$')
            ax.set_ylabel(r'$\theta_2$')
            ax.set_zlabel('Curvature')
            fig.colorbar(surf, ax=ax, shrink=0.6, aspect=10)

        fig.suptitle(suptitle, fontsize=16)
        plt.tight_layout(rect=[0, 0, 1, 0.96])  # Adjust layout to make room for the suptitle

        if config.log_dir is not None:
            fname = f"surface_{tag}.png"
            plt.savefig(os.path.join(config.log_dir, fname))

        plt.show()

    surfaces = {
        "Empirical Curvature on Input Data": interpolate(labels, curv_in),
        "Empirical Curvature on Latent Representation": interpolate(labels, curv_lat),

    }
    if config.compute_true_curv:
        surfaces["True Curvature on Input Data"] = interpolate(labels_sub, curv_true)
    if config.compute_learned_curv:
        surfaces["Learned Curvature (pullback) on Latent Representation"] = interpolate(labels_sub, curv_learned)
    if config.compute_rec_curv:
        surfaces["Empirical Curvature on Reconstructed Data"] = interpolate(labels, curv_rec)

    if entity is not None:
        plot_surface_group(surfaces, f'Curvature Comparison - Connected Component {int(entity)}',
                           'curvature_plots')
    else:
        plot_surface_group(surfaces, 'Input Data Curvature Comparison', 'true_input_reconstructed')


def compute_all_error_metrics(pairs):
    names = [name for name, _, _ in pairs]
    smape = [compute_curvature_error_smape(a, b) for _, a, b in pairs]
    mse = [compute_curvature_error_mse(a, b) for _, a, b in pairs]
    linf = [compute_curvature_error_linf(a, b) for _, a, b in pairs]
    return names, mse, smape, linf


def compute_stds(*arrays):
    stds = [np.std(np.asarray(arr)) for arr in arrays]
    labels = [name for name in ['True', 'Input', 'Latent', 'Reconstructed', 'Learned'][:len(stds)]]
    return stds, labels


def plot_error_bars(ax, names, mse, linf, bar_width=0.2):
    x = np.arange(len(['MSE', '$L^\infty$']))
    for i, name in enumerate(names):
        offsets = x + (i - len(names) / 2) * bar_width
        heights = [mse[i], linf[i]]
        ax.bar(offsets, heights, bar_width, label=name)
    ax.set_xticks(x)
    ax.set_xticklabels(['MSE', '$L^\infty$'], rotation=15)
    ax.set_title('Error Metrics')
    ax.legend(fontsize=8)
    ax.grid(True, linestyle='--', alpha=0.5)


def plot_smape(ax, names, smape):
    ax.bar(np.arange(len(smape)), smape, color='tab:cyan', width=0.4)
    ax.set_xticks(np.arange(len(smape)))
    ax.set_xticklabels(names, rotation=15)
    ax.set_title('SMAPE')
    ax.grid(True, linestyle='--', alpha=0.5)


def plot_std(ax, labels, stds):
    ax.bar(np.arange(len(stds)), stds, color='tab:gray', width=0.4)
    ax.set_xticks(np.arange(len(stds)))
    ax.set_xticklabels(labels, rotation=15)
    ax.set_title('Curvature STD')
    ax.grid(True, linestyle='--', alpha=0.5)


def plot_curvature_errors_and_stats(curv_true, curv_in, curv_rec,
                                    curv_lat, curv_lat_norm, curv_learned, config):
    pairs = [
        ("Empirical Inputs vs Latents", curv_in, curv_lat),
        ("Empirical Inputs vs Latents Normalized", curv_in, curv_lat_norm),
    ]
    std_arrays = [curv_in, curv_lat, curv_rec]
    std_labels = ['Input', 'Latent', 'Reconstructed']

    if config.compute_true_curv:
        pairs += [
            ("True vs Empirical on Inputs", curv_true, curv_in),
            ("True vs Empirical on Latent", curv_true, curv_lat),
            ("True vs Normalized Empirical on Latent", curv_true, curv_lat_norm),
        ]
        std_arrays.append(curv_true)
        std_labels.append('True')

    if config.compute_learned_curv:
        pairs += [
            ("Learned vs Empirical on Inputs", curv_learned, curv_in),
        ]
        std_arrays.append(curv_learned)
        std_labels.append('Learned')

    if config.compute_true_curv and config.compute_learned_curv:
        pairs += [
            ("True vs Learned", curv_true, curv_learned),
        ]

    names, mse, smape, linf = compute_all_error_metrics(pairs)
    stds, _ = compute_stds(*std_arrays)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    plot_error_bars(axes[0], names, mse, linf, bar_width=0.1)
    plot_smape(axes[1], names, smape)
    plot_std(axes[2], std_labels, stds)

    if config.log_dir is not None:
        plt.savefig(os.path.join(config.log_dir, "curvature_errors_combined.png"))
    plt.tight_layout()
    plt.show()

    results = {
        "error_comparisons": names,
        "errors": {"MSE": mse, "SMAPE_percent": smape, "L_inf": linf},
        "curvature_std": {"labels": std_labels, "values": stds}
    }

    if config.log_dir is not None:
        with open(os.path.join(config.log_dir, "curvature_errors_stats.json"), "w") as f:
            json.dump(results, f, indent=4)


def _plot_all_curvatures_from_vectors(config, model, recons, latents, inputs, labels):
    points_sub, curvatures_sub, curvatures_emp_full = compute_all_curvatures(config, model, recons, latents, inputs,
                                                                             labels)
    points = (inputs, latents, recons)
    inputs_sub, latents_sub, recons_sub = points_sub
    labels_sub, curv_in_sub, curv_rec_sub, curv_lat_sub, curv_lat_norm_sub, curv_true, curv_learned = curvatures_sub
    labels, curv_in, curv_lat, curv_lat_norm, curv_rec = curvatures_emp_full

    # Plot curvatures over angles
    if labels.ndim == 1:
        points_sub = (inputs_sub, latents_sub)
        plot_curvatures_1d(labels_sub, curv_true, curv_in_sub, curv_rec_sub, curv_lat_sub, curv_lat_norm_sub,
                           curv_learned, config)
        scatter_curvature_heatmaps(config, points=points, points_sub=points_sub, curv_true=curv_true,
                                   curv_in=curv_in, curv_rec=curv_rec, curv_learned=curv_learned, curv_lat=curv_lat)
    elif labels.ndim == 2 and labels.shape[1] == 2:
        points_sub = (inputs_sub, latents_sub)
        plot_curvatures_2d(labels=labels, labels_sub=labels_sub, curv_true=curv_true, curv_in=curv_in,
                           curv_rec=curv_rec, curv_lat=curv_lat, curv_learned=curv_learned,
                           config=config)
        scatter_curvature_heatmaps(config, points=points, points_sub=points_sub, curv_true=curv_true,
                                   curv_in=curv_in, curv_rec=curv_rec, curv_learned=curv_learned, curv_lat=curv_lat)
    elif labels.ndim == 2 and labels.shape[1] == 3:
        entity_indices = labels[:, 0]
        entity_indices_sub = labels_sub[:, 0]
        unique_entities = entity_indices.unique(sorted=True)
        unique_entities = unique_entities[unique_entities != 100]
        for entity in unique_entities:
            mask = (entity_indices == entity)
            mask_sub = (entity_indices_sub == entity)
            angels = labels[mask][:, 1:]
            angels_sub = labels_sub[mask_sub][:, 1:]

            inputs_entity = inputs[mask]
            latents_entity = latents[mask]
            recons_entity = recons[mask]
            points_entity = (inputs_entity, latents_entity, recons_entity)

            inputs_sub_entity = inputs_sub[mask_sub]
            latents_sub_entity = latents_sub[mask_sub]
            points_sub_entity = (inputs_sub_entity, latents_sub_entity)

            curv_true_entity = curv_true[mask_sub]
            curv_in_entity = curv_in[mask]
            curv_rec_entity = curv_rec[mask]
            curv_lat_entity = curv_lat[mask]
            curv_learned_entity = curv_learned[mask_sub]
            plot_curvatures_2d(labels=angels, labels_sub=angels_sub, curv_true=curv_true_entity,
                               curv_in=curv_in_entity, curv_rec=curv_rec_entity, curv_lat=curv_lat_entity,
                               curv_learned=curv_learned_entity, config=config, entity=entity)
            scatter_curvature_heatmaps(config, points=points_entity, points_sub=points_sub_entity,
                                       curv_true=curv_true_entity,
                                       curv_in=curv_in_entity, curv_rec=curv_rec_entity,
                                       curv_learned=curv_learned_entity, curv_lat=curv_lat_entity, entity=entity)
    else:
        raise NotImplementedError("Label dimension not supported for curvature plotting.")

    # Plot curvature heat maps

    plot_curvature_errors_and_stats(curv_true=curv_true, curv_in=curv_in_sub, curv_rec=curv_rec_sub,
                                    curv_lat=curv_lat_sub, curv_lat_norm=curv_lat_norm_sub, curv_learned=curv_learned,
                                    config=config)


def plot_all_curvatures(config, model, data_loader):
    recons, latents, inputs, labels = get_vectors(config, model, data_loader, config.n_points_emp_curv)
    _plot_all_curvatures_from_vectors(config=config, model=model, recons=recons, latents=latents, inputs=inputs,
                                      labels=labels)


def plot_persistence_diagrams(config, suptitle, diagrams, homology_dimensions):
    """Plot two persistence diagrams side-by-side and print bottleneck distance matrix."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 6), )
    fig.suptitle(suptitle, fontsize=16)
    titles = ["Persistence Diagram for Input Data", "Persistence Diagram for Latent Representation"]
    colors = {0: 'red', 1: 'blue', 2: 'green'}

    for i, ax in enumerate(axes):
        diagram = diagrams[i]
        diagram = diagram[diagram[:, 0] != diagram[:, 1]]
        bd = diagram[:, :2]

        posinf_mask = np.isposinf(bd)
        neginf_mask = np.isneginf(bd)

        if bd.size:
            max_val = np.max(np.where(posinf_mask, -np.inf, bd))
            min_val = np.min(np.where(neginf_mask, np.inf, bd))
        else:
            max_val, min_val = 1.0, 0.0

        value_range = max_val - min_val
        buffer = 0.05 * value_range
        min_val_display = min_val - buffer
        max_val_display = max_val + buffer

        ax.plot([min_val_display, max_val_display], [min_val_display, max_val_display],
                linestyle='--', color='black', linewidth=1)

        for dim in homology_dimensions:
            subdiagram = diagram[diagram[:, 2] == dim]
            births = subdiagram[:, 0]
            deaths = subdiagram[:, 1]
            deaths = np.where(np.isposinf(deaths), max_val_display + 0.05 * value_range, deaths)

            label = f"$H_{dim}$"
            ax.scatter(births, deaths, label=label, color=colors.get(dim, 'gray'), s=20)

        ax.set_title(titles[i])
        ax.set_xlabel("Birth")
        ax.set_ylabel("Death")
        ax.set_xlim([min_val_display, max_val_display])
        ax.set_ylim([min_val_display, max_val_display])
        ax.set_aspect("equal")
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.legend()

    plt.tight_layout()
    if config.log_dir is not None:
        fname = "persistence_diagrams.png"
        plt.savefig(os.path.join(config.log_dir, fname))
    plt.show()


def plot_betti_curves(config, suptitle, betti_curves, homology_dimensions=None):
    betti_numbers, samplings = betti_curves
    n_plots = len(betti_numbers)
    titles = ["Normalized Betti Curves Input Data", "Normalized Betti Curves Latent Space"]
    fig, axes = plt.subplots(nrows=1, ncols=n_plots, figsize=(6 * n_plots, 5), squeeze=False)
    fig.suptitle(suptitle, fontsize=16)

    for i, betti_numbers in enumerate(betti_numbers):
        if homology_dimensions is None:
            dims = list(range(betti_numbers.shape[0]))
        else:
            dims = homology_dimensions

        ax = axes[0, i]
        for dim in dims:
            # Normalize each Betti number curve
            normalized_betti = betti_numbers[dim] / np.max(betti_numbers[dim])  # Normalize to [0, 1]
            ax.plot(samplings[dim], normalized_betti, label=f"H{dim}", linewidth=1.5)

        ax.set_title(titles[i])
        ax.set_xlabel("Filtration parameter")
        ax.set_ylabel("Normalized Betti number")
        ax.grid(True, linestyle="--", alpha=0.6)
        ax.legend()
        ax.ticklabel_format(axis='both', style='sci', scilimits=(0, 0))

    plt.tight_layout()
    if config.log_dir is not None:
        fname = "betti_curves.png"
        plt.savefig(os.path.join(config.log_dir, fname))
    plt.show()


def plot_curvature_persistence(config, model, data_loader):
    recons, latents, inputs, labels = get_vectors(config, model, data_loader, config.n_points_emp_curv)
    _plot_all_curvatures_from_vectors(config=config, model=model, recons=recons, latents=latents, inputs=inputs,
                                      labels=labels)

    # Subsample before persistent homology

    start_time = time.time()

    if config.dataset_name in {"nested_spheres", "nested_spheres_high_dim", "interlocked_tori"}:
        n = config.n_points_pers_hom
        idx = np.random.choice(inputs.shape[0], n, replace=False)
        inputs_subsampled = inputs[idx]
        latents_subsampled = latents[idx]
        diagrams, betti_curves, distances = compare_persistent_homology((inputs_subsampled, latents_subsampled),
                                                                        config.homology_dimensions, scale=config.scale)
        title_diagram = f"Persistence Diagram Comparison - Full Dataset"
        title_betti_curve = f"Betti Curve Comparison - Full Dataset"
        plot_persistence_diagrams(config, title_diagram, diagrams, config.homology_dimensions)
        plot_betti_curves(config, title_betti_curve, betti_curves)

        entity_indices = labels[:, 0]
        unique_entities = entity_indices.unique(sorted=True)
        for entity in unique_entities:
            mask = (entity_indices == entity)
            inputs_for_entity = inputs[mask]
            latents_for_entity = latents[mask]
            n = config.n_points_pers_hom
            idx = np.random.choice(inputs_for_entity.shape[0], n, replace=False)
            inputs_subsampled = inputs_for_entity[idx]
            latents_subsampled = latents_for_entity[idx]
            diagrams, betti_curves, distances = compare_persistent_homology((inputs_subsampled, latents_subsampled),
                                                                            config.homology_dimensions,
                                                                            scale=config.scale)
            title_diagram = f"Persistence Diagram Comparison - Connected Component {entity}"
            title_betti_curve = f"Betti Curve Comparison - Connected Component {entity}"
            plot_persistence_diagrams(config, title_diagram, diagrams, config.homology_dimensions)
            plot_betti_curves(config, title_betti_curve, betti_curves)
        print("Bottleneck Distances", distances)
    else:
        n = config.n_points_pers_hom
        idx = np.random.choice(inputs.shape[0], n, replace=False)
        inputs_subsampled = inputs[idx]
        latents_subsampled = latents[idx]
        diagrams, betti_curves, distances = compare_persistent_homology((inputs_subsampled, latents_subsampled),
                                                                        config.homology_dimensions, scale=config.scale)
        title_diagram = f"Persistence Diagram Comparison"
        title_betti_curve = f"Betti Curve Comparison"
        plot_persistence_diagrams(config, title_diagram, diagrams, config.homology_dimensions)
        plot_betti_curves(config, title_betti_curve, betti_curves)
        print("Bottleneck Distances", distances)
    end_time = time.time()
    print(f"Execution time persistent homology: {end_time - start_time:.4f} seconds")

    results = {
        "bottleneck distances": distances.tolist()
    }

    if config.log_dir is not None:
        with open(os.path.join(config.log_dir, "distances.json"), "w") as f:
            json.dump(results, f, indent=4)


def normalize(arr):
    return (arr - np.min(arr)) / (np.max(arr) - np.min(arr) + 1e-8)


def plot_neural_manifold_learned(curv_norm_learned_profile, config, labels):
    if config.dataset_name == "experimental":
        stats = [
            "mean_velocities",
            "median_velocities",
            "std_velocities",
            "min_velocities",
            "max_velocities",
        ]
        cmaps = ["viridis", "viridis", "magma", "Blues", "Reds"]

        fig, axes = plt.subplots(
            nrows=1,
            ncols=len(stats),
            figsize=(10, 7),
            subplot_kw={"projection": "polar"},
        )
        for i_stat, stat_velocities in enumerate(stats):
            ax = axes[i_stat]
            ax.scatter(
                # Note: using the geodesic distance makes the plot
                # reparameterization invariant.
                # However, the computation is extremely slow, thus
                # we recommend using z_grid for the main pipeline
                # and computing geodesic_dist in the notebook 07
                # after having selected a run.
                curv_norm_learned_profile["z_grid"],
                1 / curv_norm_learned_profile["curv_norm_learned"],
                c=curv_norm_learned_profile[stat_velocities],
                cmap=cmaps[i_stat],
            )
            ax.plot(
                curv_norm_learned_profile["z_grid"],
                1 / curv_norm_learned_profile["curv_norm_learned"],
                c="black",
            )
            ax.set_rlabel_position(-22.5)  # Move radial labels away from plotted line
            ax.grid(True)
            ax.set_title("Color: " + stat_velocities, va="bottom")
            fig.tight_layout()
    else:
        fig, ax = plt.subplots(
            nrows=1, ncols=1, figsize=(20, 4), subplot_kw={"projection": "polar"}
        )

        ax.scatter(
            # Note: using the geodesic distance makes the plot
            # reparameterization invariant.
            # However, the computation is extremely slow, thus
            # we recommend using z_grid for the main pipeline
            # and computing geodesic_dist in the notebook 07
            # after having selected a run.
            curv_norm_learned_profile["z_grid"],
            1 / curv_norm_learned_profile["curv_norm_learned"],
        )
        ax.plot(
            curv_norm_learned_profile["z_grid"],
            1 / curv_norm_learned_profile["curv_norm_learned"],
            c="black",
        )
        ax.set_rlabel_position(-22.5)  # Move radial labels away from plotted line
        ax.grid(True)
        fig.tight_layout()

    return fig


def show_recon_mnist(model, loader, device="cpu"):
    model.eval()
    with torch.no_grad():
        x, _ = next(iter(loader))  # Get a batch of images
        x = x.to(device)  # Original inputs (flattened 784)

        # Forward pass through the model
        _, x_recon, _ = model(x)  # Extract only the reconstructed images

    # Randomly select 10 indices
    indices = sample(range(x.size(0)), 10)

    fig, axes = plt.subplots(2, 10, figsize=(15, 4))
    for i, idx in enumerate(indices):
        # Reshape and plot the original input images
        axes[0, i].imshow(x[idx].view(28, 28).cpu().numpy(), cmap="gray")
        axes[0, i].axis("off")
        # Reshape and plot the reconstructed images
        axes[1, i].imshow(x_recon[idx].view(28, 28).cpu().numpy(), cmap="gray")
        axes[1, i].axis("off")

    axes[0, 0].set_title("Original Images")
    axes[1, 0].set_title("Reconstructed Images")
    plt.show()


def show_recon_mnist_ae(model, loader, device="cpu"):
    model.eval()
    with torch.no_grad():
        x, _ = next(iter(loader))  # Get a batch of images
        x = x.to(device)  # Original inputs (flattened 784)

        # Forward pass through the model
        _, x_recon, _, _ = model(x)  # Extract only the reconstructed images

    # Randomly select 10 indices
    indices = sample(range(x.size(0)), 10)

    fig, axes = plt.subplots(2, 10, figsize=(15, 4))
    for i, idx in enumerate(indices):
        # Reshape and plot the original input images
        axes[0, i].imshow(x[idx].view(28, 28).cpu().numpy(), cmap="gray")
        axes[0, i].axis("off")
        # Reshape and plot the reconstructed images
        axes[1, i].imshow(x_recon[idx].view(28, 28).cpu().numpy(), cmap="gray")
        axes[1, i].axis("off")

    axes[0, 0].set_title("Original Images")
    axes[1, 0].set_title("Reconstructed Images")
    plt.show()
