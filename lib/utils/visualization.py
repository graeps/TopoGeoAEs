import torch
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from random import sample
from .evaluation import compute_curvature_learned, compute_curvature_true, compute_curvature_error
import pandas as pd
import plotly.graph_objects as go


def show_training_history(history: dict) -> None:
    """
    Displays the training history of the Variational Autoencoder (VAE) model.

    Args:
        history: A dictionary containing the training history.
    """

    _, axs = plt.subplots(figsize=(14, 4), ncols=3)

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
    axs[1].set_title('Likelihood Loss History')
    axs[1].legend()

    axs[2].plot(history['train_kl_loss'], color='orange', label='train')
    axs[2].plot(history['test_kl_loss'], color='blue', label='val')
    axs[2].set_xlabel('epoch')
    axs[2].set_ylabel('loss')
    axs[2].set_title('KL Loss History')
    axs[2].legend()

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
            labels.extend(y.numpy())

    latent_vectors = torch.cat(latent_vectors, dim=0).numpy()
    labels = np.array(labels)

    # Randomly select n_samples points
    if len(latent_vectors) > n_samples:
        indices = sample(range(len(latent_vectors)), n_samples)
        latent_vectors = latent_vectors[indices]
        labels = labels[indices]

    # Apply PCA if latent space has more than 2 dimensions
    if latent_vectors.shape[1] > 2:
        pca = PCA(n_components=2)
        latent_vectors = pca.fit_transform(latent_vectors)

    # Plot the latent space
    plt.figure(figsize=(10, 7))
    scatter = plt.scatter(
        latent_vectors[:, 0], latent_vectors[:, 1], c=labels, cmap='tab10', alpha=0.7
    )
    plt.colorbar(scatter, label="Class Label")
    plt.title("Latent Space Visualization")
    plt.xlabel("Dimension 1")
    plt.ylabel("Dimension 2")
    plt.show()


def curvature_compute_plot(config, dataset, labels, model):
    """Compute and plot curvature results."""
    # Compute
    print("Computing learned curvature...")
    z_grid, geodesic_dist, _, curv_norms_learned = compute_curvature_learned(
        model=model,
        config=config,
        embedding_dim=dataset.shape[1],
        n_grid_points=config.n_grid_points,
    )

    curv_norm_learned_profile = pd.DataFrame(
        {
            "geodesic_dist": geodesic_dist,
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
        z_grid, geodesic_dist, _, curv_norms_true = compute_curvature_true(
            config, n_grid_points=config.n_grid_points
        )
        print("Computing curvature error for synthetic data...")

        curvature_error = compute_curvature_error(
            z_grid, curv_norms_learned, curv_norms_true, config
        )
        norm_val = max(curv_norms_true)

        curv_norm_true_profile = pd.DataFrame(
            {
                "geodesic_dist": geodesic_dist,
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

    if config.dataset_name in (
            "s1_synthetic",
            "experimental",
            "three_place_cells_synthetic",
    ):
        fig_neural_manifold_learned = plot_neural_manifold_learned(
            curv_norm_learned_profile=curv_norm_learned_profile,
            config=config,
            labels=labels,
        )


def plot_curvature_norms(angles, curvature_norms, config, norm_val, profile_type):
    fig = plt.figure(figsize=(24, 12))
    colormap = plt.get_cmap("hsv")
    if norm_val is not None:
        color_norm = mpl.colors.Normalize(0.0, norm_val)
    else:
        color_norm = mpl.colors.Normalize(0.0, max(curvature_norms))
    if config.dataset_name in ("s1_synthetic", "experimental"):
        ax1 = fig.add_subplot(121)
        ax1.plot(angles, curvature_norms, linewidth=10)
        ax1.set_xlabel("angle", fontsize=30)
        ax1.set_ylabel("mean curvature norm", fontsize=30)

        ax2 = fig.add_subplot(122, projection="polar")
        sc = ax2.scatter(
            angles,
            np.ones_like(angles),
            c=curvature_norms,
            s=400,
            cmap=colormap,
            norm=color_norm,
            linewidths=0,
        )
        ax2.set_yticks([])
        ax2.set_xlabel("angle", fontsize=30)

        ax1.set_title(f"{profile_type} mean curvature norm profile", fontsize=30)
        ax2.set_title(f"{profile_type} mean curvature norm profile", fontsize=30)

    elif config.dataset_name == "s2_synthetic":
        ax = fig.add_subplot(111, projection="3d")
        x = config.radius * [np.sin(angle[0]) * np.cos(angle[1]) for angle in angles]
        y = config.radius * [np.sin(angle[0]) * np.sin(angle[1]) for angle in angles]
        z = config.radius * [np.cos(angle[0]) for angle in angles]
        sc = ax.scatter3D(
            x, y, z, s=400, c=curvature_norms, cmap="Spectral", norm=color_norm
        )
        plt.colorbar(sc)
        ax.set_title(f"{profile_type} mean curvature norm profile", fontsize=30)
    elif config.dataset_name == "t2_synthetic":
        ax = fig.add_subplot(111, projection="3d")
        x = [
            (config.major_radius - config.minor_radius * np.cos(angle[0]))
            * np.cos(angle[1])
            for angle in angles
        ]
        y = [
            (config.major_radius - config.minor_radius * np.cos(angle[0]))
            * np.sin(angle[1])
            for angle in angles
        ]
        z = [config.minor_radius * np.sin(angle[0]) for angle in angles]
        sc = ax.scatter3D(
            x, y, z, s=400, c=curvature_norms, cmap="Spectral", norm=color_norm
        )
        plt.colorbar(sc)
        ax.set_title(f"{profile_type} mean curvature norm profile", fontsize=30)
        ax.set_xlim(
            -(config.major_radius + config.minor_radius),
            (config.major_radius + config.minor_radius),
        )
        ax.set_ylim(
            -(config.major_radius + config.minor_radius),
            (config.major_radius + config.minor_radius),
        )
        ax.set_zlim(
            -(config.major_radius + config.minor_radius),
            (config.major_radius + config.minor_radius),
        )
        plt.axis("off")

    if config.dataset_name in ["s2_synthetic", "t2_synthetic", "grid_cells"]:
        if norm_val is not None:
            plotly_fig = go.Figure(
                data=[
                    go.Scatter3d(
                        x=x,
                        y=y,
                        z=z,
                        mode="markers",
                        marker=dict(
                            size=10,
                            color=curvature_norms,  # set color to an array/list of desired values
                            colorscale="plasma",  # choose a colorscale
                            opacity=0.8,
                            cmin=0,
                            cmax=float(norm_val),
                            colorbar=dict(title="Norm of curvature", tickmode="auto"),
                        ),
                    )
                ]
            )
        else:
            plotly_fig = go.Figure(
                data=[
                    go.Scatter3d(
                        x=x,
                        y=y,
                        z=z,
                        mode="markers",
                        marker=dict(
                            size=10,
                            color=curvature_norms,  # set color to an array/list of desired values
                            colorscale="plasma",  # choose a colorscale
                            opacity=0.8,
                            colorbar=dict(title="Norm of curvature", tickmode="auto"),
                        ),
                    )
                ]
            )

        plotly_fig.update_layout(
            scene=dict(xaxis_title="X", yaxis_title="Y", zaxis_title="Z"),
            title=dict(text="Profile of Curvature Norm", font=dict(size=24), x=0.5),
        )

        plotly_fig.update_layout(
            scene=dict(xaxis_title="X", yaxis_title="Y", zaxis_title="Z"),
            title=dict(text="Profile of Curvature Norm", font=dict(size=24), x=0.5),
        )

    return fig


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
            figsize=(20, 4),
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
