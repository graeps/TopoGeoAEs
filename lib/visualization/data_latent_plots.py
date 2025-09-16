import torch
import numpy as np
import matplotlib.pyplot as plt
import os

from .utils import scatter_datapoints
from ..utils.vectors import get_vectors


def plot_data_latents_recon(config, model, data_loader):

    recons, latents, inputs, labels = get_vectors(config, model, data_loader, config.n_plot_points)
    if torch.isnan(latents).any():
        print("NaNs detected in the dataset!")

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
    elif config.dataset_name in {"nested_spheres", "nested_spheres_high_dim", "interlocked_tori", "interlocked_tubes"}:
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
                        pca_dim=pca_dim)

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
            scatter_datapoints(ax=ax_input, data=inputs, title=None, colors=colors, cmap=color_map, pca_dim=pca_dim)
            fig_input.tight_layout()
            fig_input.savefig(os.path.join(config.log_dir, "input_data.png"))
            plt.close(fig_input)

            # Latent Representation
            fig_latent = plt.figure()
            ax_latent = fig_latent.add_subplot(1, 1, 1,
                                               projection='3d' if latents.shape[1] >= 3 and pca_dim == 3 else None)
            scatter_datapoints(ax=ax_latent, data=latents, title=None, colors=colors, cmap=color_map, pca_dim=pca_dim)
            fig_latent.tight_layout()
            fig_latent.savefig(os.path.join(config.log_dir, "latent_representation.png"))
            plt.close(fig_latent)

            # Reconstruction
            fig_recon = plt.figure()
            ax_recon = fig_recon.add_subplot(1, 1, 1,
                                             projection='3d' if recons.shape[1] >= 3 and pca_dim == 3 else None)
            scatter_datapoints(ax=ax_recon, data=recons, title=None, colors=colors, cmap=color_map, pca_dim=pca_dim)
            fig_recon.tight_layout()
            fig_recon.savefig(os.path.join(config.log_dir, "reconstructed_data.png"))
            plt.close(fig_recon)
