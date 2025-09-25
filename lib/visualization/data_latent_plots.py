import torch
import numpy as np
import matplotlib.pyplot as plt
import os

from .utils import scatter_datapoints
from ..utils.vectors import get_vectors
from ..datasets.lookup import get_dataset_category
from ..models.lookup import is_euclidean_model


def plot_data_latents_recon(config, model, data_loader):
    """
    Generates and visualizes comparison plots for input data, latent representations, and reconstructed data.
    The function creates three subplots: original dataset, latent space, and reconstruction.
    It also saves the generated plots to the specified logging directory. The visualization offers support
    for both 2D and 3D datasets, dynamically adjusting rendering based on the dataset type.

    Args:
        config (Any): Contains configuration parameters for the visualization, including dataset options,
                      logging directory, and normalization flags.
        model (Any): The trained model used to generate latent representations and reconstructions.
        data_loader (Any): The data loader providing input data and labels for visualization.
    """
    recons, latents, inputs, labels = get_vectors(config, model, data_loader, config.n_plot_points)
    if torch.isnan(latents).any():
        print("NaNs detected in the dataset!")

    def _set_axis_limits(ax, dataset_name: str):
        if ax is None:
            return
        # Spherical datasets
        if dataset_name in {"s1_low", "s1_high", "s2_low", "s2_high"}:
            try:
                ax.set_xlim(-1.5, 1.5)
                ax.set_ylim(-1.5, 1.5)
                if hasattr(ax, "set_zlim"):
                    ax.set_zlim(-1.1, 1.1)
                # Ensure aspect is compatible with fixed limits to avoid warnings
                ax.set_aspect('equal', adjustable='box')
            except Exception:
                pass
        # Toroidal datasets
        if dataset_name in {"t2_low", "t2_high"}:
            ax_lim = 3
            try:
                ax.set_xlim(-ax_lim, ax_lim)
                ax.set_ylim(-ax_lim, ax_lim)
                if hasattr(ax, "set_zlim"):
                    ax.set_zlim(-ax_lim, ax_lim)
                # Ensure aspect is compatible with fixed limits to avoid warnings
                ax.set_aspect('equal', adjustable='box')
            except Exception:
                pass

    if getattr(config, "type", None) == "torus_ae" and getattr(config, "normalize", False):
        u1, u2 = latents[:, :2], latents[:, 2:]
        theta = torch.atan2(u1[:, 1], u1[:, 0])
        phi = torch.atan2(u2[:, 1], u2[:, 0])
        R, r = 2.0, 1.0
        x = (R + r * torch.cos(phi)) * torch.cos(theta)
        y = (R + r * torch.cos(phi)) * torch.sin(theta)
        z = r * torch.sin(phi)
        latents = torch.stack((x, y, z), dim=-1)

    if labels.ndim > 1 and labels.shape[1] == 2:
        colors = labels[:, 0]
        color_map = "hsv"
    elif get_dataset_category(config.dataset_name) == "multi_entity":
        tab10_colors = plt.cm.get_cmap('tab10', 10)
        colors = np.array([tab10_colors(int(label)) for label in labels[:, 0]])
        color_map = None
    else:
        colors = labels.squeeze()
        color_map = "viridis"

    fig = plt.figure(figsize=(18, 5))

    if config.dataset_name == "s1_low":
        pca_dim = 2
    else:
        pca_dim = 3

    # Dataset plot
    ax1 = fig.add_subplot(1, 3, 1, projection='3d' if inputs.shape[1] >= 3 and pca_dim == 3 else None)
    scatter_datapoints(ax=ax1, data=inputs, title="Input Data", colors=colors, cmap=color_map, pca_dim=pca_dim)

    # Latent space plot
    ax2 = fig.add_subplot(1, 3, 2, projection='3d' if latents.shape[1] >= 3 and pca_dim == 3 else None)
    scatter_datapoints(ax=ax2, data=latents, title="Latent Representation", colors=colors, cmap=color_map,
                        pca_dim=pca_dim, apply_pca=False)
    if not is_euclidean_model(getattr(config, "model_type", "")):
        _set_axis_limits(ax2, config.dataset_name)

    # Reconstruction plot
    ax3 = fig.add_subplot(1, 3, 3, projection='3d' if recons.shape[1] >= 3 and pca_dim == 3 else None)
    scatter_datapoints(ax=ax3, data=recons, title="Reconstructed Data", colors=colors, cmap=color_map, pca_dim=pca_dim)

    plt.tight_layout()
    plt.show()

    # Save combined plot
    if config.log_dir is not None:
        save_path = os.path.join(config.log_dir, "input_latents_recons_plot.png")
        plt.savefig(save_path)

        if config.log_dir is not None:
            # Input Data
            fig_input = plt.figure()
            ax_input = fig_input.add_subplot(1, 1, 1,
                                             projection='3d' if inputs.shape[1] >= 3 and pca_dim == 3 else None)
            scatter_datapoints(ax=ax_input, data=inputs, title="", colors=colors, cmap=color_map, pca_dim=pca_dim)
            fig_input.tight_layout()
            fig_input.savefig(os.path.join(config.log_dir, "input_data.png"))
            plt.close(fig_input)

            # Latent Representation
            fig_latent = plt.figure()
            ax_latent = fig_latent.add_subplot(1, 1, 1,
                                               projection='3d' if latents.shape[1] >= 3 and pca_dim == 3 else None)
            if not is_euclidean_model(getattr(config, "model_type", "")):
                apply_pca=False
            else:
                apply_pca=True
            scatter_datapoints(ax=ax_latent, data=latents, apply_pca=apply_pca, title="", colors=colors, cmap=color_map, pca_dim=pca_dim)
            if not is_euclidean_model(getattr(config, "model_type", "")):
                _set_axis_limits(ax_latent, config.dataset_name)
            fig_latent.tight_layout()
            fig_latent.savefig(os.path.join(config.log_dir, "latent_representation.png"))
            plt.close(fig_latent)

            # Reconstruction
            fig_recon = plt.figure()
            ax_recon = fig_recon.add_subplot(1, 1, 1,
                                             projection='3d' if recons.shape[1] >= 3 and pca_dim == 3 else None)
            scatter_datapoints(ax=ax_recon, data=recons, title="", colors=colors, cmap=color_map, pca_dim=pca_dim)
            fig_recon.tight_layout()
            fig_recon.savefig(os.path.join(config.log_dir, "reconstructed_data.png"))
            plt.close(fig_recon)
