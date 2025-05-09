import matplotlib as mpl
from random import sample
import pandas as pd
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

from .evaluation import compute_curvature_learned, compute_curvature_true, compute_curvature_error, \
    compute_curvature_true_latents, estimate_curvature_1d_quadric, estimate_curvature_2d_quadric, \
    compute_empiric_curvature
from ..datasets.synthetic_sphere_like import get_s1_synthetic_immersion, get_scrunchy_immersion, \
    get_interlocking_rings_immersion


def show_training_history(history: dict) -> None:
    """
    Displays the training history of the Variational Autoencoder (VAE) model.

    Args:
        history: A dictionary containing the training history.
    """

    _, axs = plt.subplots(figsize=(14, 4), ncols=4)

    axs[0].plot(history['train_loss'], color='orange', label='train')
    axs[0].plot(history['test_loss'], color='blue', label='val')
    axs[0].set_xlabel('epoch')
    axs[0].set_ylabel('loss')
    axs[0].set_title('Loss History')
    axs[0].legend()

    axs[1].plot(history['train_recon_loss'], color='orange', label='train')
    axs[1].plot(history['test_recon_loss'], color='blue', label='val')
    axs[1].set_xlabel('epoch')
    axs[1].set_ylabel('loss')
    axs[1].set_title('Recon Loss History')
    axs[1].legend()

    axs[2].plot(history['train_kl_loss'], color='orange', label='train')
    axs[2].plot(history['test_kl_loss'], color='blue', label='val')
    axs[2].set_xlabel('epoch')
    axs[2].set_ylabel('loss')
    axs[2].set_title('KL Loss History')
    axs[2].legend()

    axs[3].plot(history['train_topo_loss'], color='orange', label='train')
    axs[3].plot(history['test_topo_loss'], color='blue', label='val')
    axs[3].set_xlabel('epoch')
    axs[3].set_ylabel('loss')
    axs[3].set_title('Topo Loss History')
    axs[3].legend()

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


def plot_euclidean_latent_space(model, test_loader, device='cpu', n_samples=200):
    """
    Plots the latent space of a Variational Autoencoder (VAE).

    Args:
        model: The trained VAE model.
        test_loader: DataLoader for the test dataset.
        device: The device to run computations on ('cpu' or 'cuda').
        n_samples: Number of random samples to plot.
    """
    model.eval()
    model.to(device)

    latent_vectors = []
    labels = []

    # Collect latent space representations and labels
    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(device)
            z, _, _ = model.forward(x)
            latent_vectors.append(z.cpu())
            labels.append(y)

    latent_vectors = torch.cat(latent_vectors, dim=0)
    labels = torch.cat(labels, dim=0)

    latent_vectors = latent_vectors.numpy()
    labels = labels.numpy()

    assert latent_vectors.shape[0] == labels.shape[0], \
        f"Mismatch: {latent_vectors.shape[0]} latent vectors vs {labels.shape[0]} labels"

    n_total = latent_vectors.shape[0]
    if n_samples > n_total:
        print(f"Warning: n_samples ({n_samples}) > total samples ({n_total}). Using all samples.")
        n_samples = n_total

    indices = np.random.choice(n_total, size=n_samples, replace=False)
    latent_vectors = latent_vectors[indices]
    labels = labels[indices]

    if labels.ndim > 1 and labels.shape[1] == 2:
        colors = (labels[:, 0] + labels[:, 1]) % 360
    else:
        colors = labels.squeeze()

    dim = latent_vectors.shape[1]

    if dim == 1:
        # 1D scatter plot
        plt.figure(figsize=(10, 2))
        plt.scatter(latent_vectors[:, 0], np.zeros_like(latent_vectors[:, 0]),
                    c=colors, cmap='hsv', alpha=0.7)
        plt.xlabel("Latent Dimension 1")
        plt.title("1D Latent Space Visualization")
        plt.yticks([])
        plt.colorbar(label="Class Label")
        plt.show()
    elif dim == 2:
        plt.figure(figsize=(10, 7))
        scatter = plt.scatter(
            latent_vectors[:, 0], latent_vectors[:, 1], c=colors, cmap='hsv', alpha=0.7
        )
        plt.colorbar(scatter, label="Class Label")
        plt.title("2D Latent Space Visualization")
        plt.xlabel("Dimension 1")
        plt.ylabel("Dimension 2")
        plt.show()
    elif dim == 3:
        fig = plt.figure(figsize=(10, 7))
        ax = fig.add_subplot(111, projection='3d')
        scatter = ax.scatter(
            latent_vectors[:, 0], latent_vectors[:, 1], latent_vectors[:, 2],
            c=colors, cmap='hsv', alpha=0.7
        )
        ax.set_title("3D Latent Space Visualization")
        ax.set_xlabel("Dimension 1")
        ax.set_ylabel("Dimension 2")
        ax.set_zlabel("Dimension 3")
        fig.colorbar(scatter, label="Class Label")
        plt.show()
    else:
        # PCA to 2D
        pca = PCA(n_components=2)
        reduced = pca.fit_transform(latent_vectors)
        plt.figure(figsize=(10, 7))
        scatter = plt.scatter(
            reduced[:, 0], reduced[:, 1], c=colors, cmap='hsv', alpha=0.7
        )
        plt.colorbar(scatter, label="Class Label")
        plt.title("PCA Projection of Latent Space")
        plt.xlabel("PC 1")
        plt.ylabel("PC 2")
        plt.show()


def plot_recon_manifold(model, test_loader, device='cpu', n_samples=200):
    model.eval()
    model.to(device)

    recon_dataset = []
    labels = []

    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(device)
            _, x_recon, _ = model(x)
            recon_dataset.append(x_recon.cpu())
            labels.extend(y.numpy())

    recon_dataset = np.concatenate(recon_dataset)
    labels = np.array(labels)

    if len(recon_dataset) > n_samples:
        indices = sample(range(len(recon_dataset)), n_samples)
        recon_dataset = recon_dataset[indices]
        labels = labels[indices]

    if labels.shape[1] == 2:
        colors = (labels[:, 0] + labels[:, 1]) % 360
    else:
        colors = labels

    if recon_dataset.shape[1] == 2:
        plt.scatter(recon_dataset[:, 0], recon_dataset[:, 1], c=colors, cmap='hsv')
        plt.axis('equal')
        plt.title("Reconstructed Manifold ℝ²")
        plt.colorbar(label='Angle [0, 2π]')
        plt.show()
    elif recon_dataset.shape[1] == 3:
        fig = plt.figure(figsize=(10, 7))
        ax = fig.add_subplot(111, projection='3d')
        p = ax.scatter(recon_dataset[:, 0], recon_dataset[:, 1], recon_dataset[:, 2],
                       c=colors, cmap='hsv')
        fig.colorbar(p, ax=ax, label='Angle [0, 2π]')
        ax.set_title("Reconstructed Manifold in ℝ³")
        plt.show()
    else:
        proj = PCA(n_components=2).fit_transform(recon_dataset)
        plt.scatter(proj[:, 0], proj[:, 1], c=colors, cmap='hsv')
        plt.axis('equal')
        plt.title("Reconstructed Manifold projected to ℝ² via PCA")
        plt.colorbar(label='Angle [0, 2π]')
        plt.show()


def plot_dataset(test_loader, device='cpu'):
    dataset = []
    for x, _ in test_loader:
        dataset.append(x.to(device))
    dataset = torch.cat(dataset, dim=0)

    if dataset.shape[1] == 2:
        plt.scatter(dataset[:, 0].cpu(), dataset[:, 1].cpu(), s=1)
        plt.axis('equal')
        plt.title("Noisy S¹ in ℝ²")
        plt.show()
    elif dataset.shape[1] == 3:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(dataset[:, 0].cpu(), dataset[:, 1].cpu(), dataset[:, 2].cpu(), s=1)
        ax.set_title("Noisy S¹ in ℝ³")
        plt.show()
    else:
        proj = PCA(n_components=2).fit_transform(dataset.cpu().numpy())
        plt.scatter(proj[:, 0], proj[:, 1], s=1)
        plt.axis('equal')
        plt.title("Noisy S¹ projected to ℝ² via PCA")
        plt.show()


def plot_data_latents_recon(model, test_loader, device='cpu', n_samples=200):
    model.eval()
    model.to(device)

    # Gather original data, latent vectors, reconstructions, and labels
    dataset = []
    latent_vectors = []
    recon_dataset = []
    labels = []

    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(device)
            z, x_recon, _ = model(x)

            dataset.append(x.cpu())
            latent_vectors.append(z.cpu())
            recon_dataset.append(x_recon.cpu())
            labels.append(y)

    dataset = torch.cat(dataset, dim=0).numpy()
    latent_vectors = torch.cat(latent_vectors, dim=0).numpy()
    recon_dataset = torch.cat(recon_dataset, dim=0).numpy()
    labels = torch.cat(labels, dim=0).numpy()

    # Downsample
    n_total = latent_vectors.shape[0]
    if n_samples > n_total:
        n_samples = n_total
    indices = np.random.choice(n_total, size=n_samples, replace=False)

    dataset = dataset[indices]
    latent_vectors = latent_vectors[indices]
    recon_dataset = recon_dataset[indices]
    labels = labels[indices]

    if labels.ndim > 1 and labels.shape[1] == 2:
        colors = (labels[:, 0] + labels[:, 1]) % 360
    else:
        colors = labels.squeeze()

    fig = plt.figure(figsize=(18, 5))

    # Dataset plot
    ax1 = fig.add_subplot(1, 3, 1, projection='3d' if dataset.shape[1] == 3 else None)
    if dataset.shape[1] == 2:
        ax1.scatter(dataset[:, 0], dataset[:, 1], s=1)
        ax1.set_title("Noisy S¹ in ℝ²")
    elif dataset.shape[1] == 3:
        ax1.scatter(dataset[:, 0], dataset[:, 1], dataset[:, 2], s=1)
        ax1.set_title("Noisy S¹ in ℝ³")
    else:
        proj = PCA(n_components=2).fit_transform(dataset)
        ax1.scatter(proj[:, 0], proj[:, 1], s=1)
        ax1.set_title("Noisy S¹ projected to ℝ² via PCA")
    ax1.axis('equal')

    # Latent space plot
    ax2 = fig.add_subplot(1, 3, 2, projection='3d' if latent_vectors.shape[1] == 3 else None)
    if latent_vectors.shape[1] == 1:
        ax2 = fig.add_subplot(1, 3, 2)
        ax2.scatter(latent_vectors[:, 0], np.zeros_like(latent_vectors[:, 0]), c=colors, cmap='hsv', alpha=0.7, s=1)
        ax2.set_title("1D Latent Space")
        ax2.set_yticks([])
    elif latent_vectors.shape[1] == 2:
        ax2.scatter(latent_vectors[:, 0], latent_vectors[:, 1], c=colors, cmap='hsv', alpha=0.7, s=1)
        ax2.set_title("2D Latent Space")
    elif latent_vectors.shape[1] == 3:
        ax2.scatter(latent_vectors[:, 0], latent_vectors[:, 1], latent_vectors[:, 2], c=colors, cmap='hsv', alpha=0.7,
                    s=1)
        ax2.set_title("3D Latent Space")
    else:
        reduced = PCA(n_components=2).fit_transform(latent_vectors)
        ax2 = fig.add_subplot(1, 3, 2)
        ax2.scatter(reduced[:, 0], reduced[:, 1], c=colors, cmap='hsv', alpha=0.7, s=1)
        ax2.set_title("Latent Space (PCA)")
    if hasattr(ax2, 'axis'):
        ax2.axis('equal')

    # Recon manifold plot
    ax3 = fig.add_subplot(1, 3, 3, projection='3d' if recon_dataset.shape[1] == 3 else None)
    if recon_dataset.shape[1] == 2:
        ax3.scatter(recon_dataset[:, 0], recon_dataset[:, 1], c=colors, cmap='hsv', s=1)
        ax3.set_title("Reconstructed Manifold ℝ²")
    elif recon_dataset.shape[1] == 3:
        ax3.scatter(recon_dataset[:, 0], recon_dataset[:, 1], recon_dataset[:, 2], c=colors, cmap='hsv', s=1)
        ax3.set_title("Reconstructed Manifold ℝ³")
    else:
        proj = PCA(n_components=2).fit_transform(recon_dataset)
        ax3 = fig.add_subplot(1, 3, 3)
        ax3.scatter(proj[:, 0], proj[:, 1], c=colors, cmap='hsv', s=1)
        ax3.set_title("Reconstructed Manifold (PCA)")
    if hasattr(ax3, 'axis'):
        ax3.axis('equal')

    plt.tight_layout()
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
        test_loader=test_loader,
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
    labels, _, curv_norms_true_latents = compute_curvature_true_latents(config, labels.squeeze())

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


def scatter_empiric_curvature(config, model, test_loader, n_samples=2000):
    model.eval()
    model.to(config.device)

    # Gather original data, latent vectors, reconstructions, and labels
    dataset = []
    latent_vectors = []
    recon_dataset = []

    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(config.device)
            z, x_recon, _ = model(x)

            dataset.append(x.cpu())
            latent_vectors.append(z.cpu())
            recon_dataset.append(x_recon.cpu())

    dataset = torch.cat(dataset, dim=0).numpy()
    latent_vectors = torch.cat(latent_vectors, dim=0).numpy()
    recon_dataset = torch.cat(recon_dataset, dim=0).numpy()

    # Downsample
    n_total = latent_vectors.shape[0]
    if n_samples > n_total:
        n_samples = n_total
    indices = np.random.choice(n_total, size=n_samples, replace=False)

    dataset = dataset[indices]
    latent_vectors = latent_vectors[indices]
    recon_dataset = recon_dataset[indices]

    true_curvature = estimate_curvature_1d_quadric(dataset)
    latent_curvature = estimate_curvature_1d_quadric(latent_vectors)
    recon_curvature = estimate_curvature_1d_quadric(recon_dataset)

    fig = plt.figure(figsize=(18, 5))

    # Dataset plot
    ax1 = fig.add_subplot(1, 3, 1, projection='3d' if dataset.shape[1] == 3 else None)
    if dataset.shape[1] == 2:
        sc1 = ax1.scatter(dataset[:, 0], dataset[:, 1], c=true_curvature, cmap='hsv', s=1)
        ax1.set_title("Noisy S¹ in ℝ²")
    elif dataset.shape[1] == 3:
        sc1 = ax1.scatter(dataset[:, 0], dataset[:, 1], dataset[:, 2], c=true_curvature, cmap='hsv', s=1)
        ax1.set_title("Noisy S¹ in ℝ³")
    else:
        proj = PCA(n_components=2).fit_transform(dataset)
        sc1 = ax1.scatter(proj[:, 0], proj[:, 1], c=true_curvature, cmap='hsv', s=1)
        ax1.set_title("Noisy S¹ projected to ℝ² via PCA")
    ax1.axis('equal')
    fig.colorbar(sc1, ax=ax1, shrink=0.7)

    # Latent space plot
    ax2 = fig.add_subplot(1, 3, 2, projection='3d' if latent_vectors.shape[1] == 3 else None)
    if latent_vectors.shape[1] == 1:
        ax2 = fig.add_subplot(1, 3, 2)
        sc2 = ax2.scatter(latent_vectors[:, 0], np.zeros_like(latent_vectors[:, 0]), c=latent_curvature, cmap='hsv',
                          alpha=0.7, s=1)
        ax2.set_title("1D Latent Space")
        ax2.set_yticks([])
    elif latent_vectors.shape[1] == 2:
        sc2 = ax2.scatter(latent_vectors[:, 0], latent_vectors[:, 1], c=latent_curvature, cmap='hsv', alpha=0.7, s=1)
        ax2.set_title("2D Latent Space")
    elif latent_vectors.shape[1] == 3:
        sc2 = ax2.scatter(latent_vectors[:, 0], latent_vectors[:, 1], latent_vectors[:, 2], c=latent_curvature,
                          cmap='hsv', alpha=0.7)
        ax2.set_title("3D Latent Space")
    else:
        reduced = PCA(n_components=2).fit_transform(latent_vectors)
        ax2 = fig.add_subplot(1, 3, 2)
        sc2 = ax2.scatter(reduced[:, 0], reduced[:, 1], c=latent_curvature, cmap='hsv', alpha=0.7, s=1)
        ax2.set_title("Latent Space (PCA)")
    if hasattr(ax2, 'axis'):
        ax2.axis('equal')
    fig.colorbar(sc2, ax=ax2, shrink=0.7)

    # Recon manifold plot
    ax3 = fig.add_subplot(1, 3, 3, projection='3d' if recon_dataset.shape[1] == 3 else None)
    if recon_dataset.shape[1] == 2:
        sc3 = ax3.scatter(recon_dataset[:, 0], recon_dataset[:, 1], c=recon_curvature, cmap='hsv', s=1)
        ax3.set_title("Reconstructed Manifold ℝ²")
    elif recon_dataset.shape[1] == 3:
        sc3 = ax3.scatter(recon_dataset[:, 0], recon_dataset[:, 1], recon_dataset[:, 2], c=recon_curvature, cmap='hsv',
                          s=1)
        ax3.set_title("Reconstructed Manifold ℝ³")
    else:
        proj = PCA(n_components=2).fit_transform(recon_dataset)
        ax3 = fig.add_subplot(1, 3, 3)
        sc3 = ax3.scatter(proj[:, 0], proj[:, 1], c=recon_curvature, cmap='hsv', s=1)
        ax3.set_title("Reconstructed Manifold (PCA)")
    if hasattr(ax3, 'axis'):
        ax3.axis('equal')
    fig.colorbar(sc3, ax=ax3, shrink=0.7)

    plt.tight_layout()
    plt.show()


def plot_empiric_curvature(config, model):
    # Compute empiric curvature
    curvature_inputs, curvature_latents, curvature_recons, labels = compute_empiric_curvature(config, model)
    angles = labels["angles"]
    # Compute true curvature
    _, _, curvature_true = compute_curvature_true_latents(config, angles)

    plt.figure(figsize=(10, 6))

    plt.plot(angles, curvature_inputs, label='Input Curvature', color='tab:blue', linewidth=1.5, alpha=0.7)
    plt.plot(angles, curvature_true, label='True Curvature', color='tab:green', linewidth=1.5, alpha=0.5)
    #plt.plot(angles, curvature_latents, label='Latent Curvature', color='tab:orange', linewidth=1, alpha=0.7)
    #plt.plot(angles, curvature_recons, label='Reconstructed Curvature', color='tab:red', linewidth=1, alpha=0.7)

    plt.xlabel('Angle (radians)')
    plt.ylabel('Curvature')
    plt.title('Curvature Comparison Across Representations')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()


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
