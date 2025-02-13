import numpy as np
import torch
import matplotlib.pyplot as plt


def plot_losses(train_losses, test_losses, num_epochs):
    """
    Plot training and testing losses over epochs.

    Parameters
    ----------
    train_losses : list of float
        List of training losses, where each element represents the loss for a specific epoch.
    test_losses : list of float
        List of testing losses, where each element represents the loss for a specific epoch.
    num_epochs : int
        Total number of epochs for which the losses are recorded.

    Returns
    -------
    None
    """
    plt.figure(figsize=(8, 6))
    plt.plot(range(1, num_epochs + 1), train_losses, label="Train Loss", marker="o")
    plt.plot(range(1, num_epochs + 1), test_losses, label="Test Loss", marker="o")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Train Loss vs Test Loss")
    plt.legend()
    plt.grid(True)
    plt.show()


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


def plot_test_latents_on_torus(model, test_loader, device):
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
