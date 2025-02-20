import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA


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


def plot_latent_space_euclidean(model, test_loader, device="cpu"):
    """
    Visualizes the latent space of the test dataset mapped through the model.

    Args:
        model (nn.Module): Trained EuclideanVAE model.
        test_loader (DataLoader): DataLoader for the test dataset.
        device (str): Device to use for computation (e.g., "cpu" or "cuda").
    """
    model.eval()  # Set the model to evaluation mode
    latent_vectors = []
    labels = []

    with torch.no_grad():
        for data, target in test_loader:
            data = data.to(device)
            z, _, _ = model(data)  # Get latent space representation
            latent_vectors.append(z.cpu().numpy())
            labels.append(target.cpu().numpy())

    # Concatenate all batches
    latent_vectors = np.concatenate(latent_vectors, axis=0)
    labels = np.concatenate(labels, axis=0)

    # If latent space is > 2D, reduce dimensionality using t-SNE
    if latent_vectors.shape[1] > 2:
        tsne = TSNE(n_components=2, random_state=42)
        latent_vectors = tsne.fit_transform(latent_vectors)

    # Plot the latent space
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(latent_vectors[:, 0], latent_vectors[:, 1], c=labels, cmap="tab10", alpha=0.6)
    plt.colorbar(scatter, label="Class")
    plt.title("Latent Space Visualization")
    plt.xlabel("Latent Dimension 1")
    plt.ylabel("Latent Dimension 2")
    plt.show()


def show_recon_mnist(model, loader, num_images=8, device="cpu"):
    model.eval()
    with torch.no_grad():
        x, _ = next(iter(loader))  # Get a batch of images
        x = x.to(device)  # Original inputs (flattened 784)

        # Forward pass through the model
        _, x_recon, _ = model(x)  # Extract only the reconstructed images

    fig, axes = plt.subplots(2, num_images, figsize=(15, 4))
    for i in range(num_images):
        # Reshape and plot the original input images
        axes[0, i].imshow(x[i].view(28, 28).cpu().numpy(), cmap="gray")
        axes[0, i].axis("off")
        # Reshape and plot the reconstructed images
        axes[1, i].imshow(x_recon[i].view(28, 28).cpu().numpy(), cmap="gray")
        axes[1, i].axis("off")

    axes[0, 0].set_title("Original Images")
    axes[1, 0].set_title("Reconstructed Images")
    plt.show()


def plot_posterior_params(model, test_loader, device="cpu"):
    """
    Plot the mu (x-axis) and sigma (y-axis) from posterior parameters of the test set.

    Args:
        model (EuclideanVAE): The trained VAE model.
        test_loader (DataLoader): DataLoader for the test dataset.
        device (str): Device to run the computation on.
    """
    model.eval()
    all_mu = []
    all_sigma = []

    with torch.no_grad():
        for batch in test_loader:
            x = batch[0].to(device)  # Assuming test_loader yields (data, labels)
            mu, logvar = model.encode(x)
            sigma = torch.exp(0.5 * logvar)  # Convert logvar to std (sigma)
            all_mu.append(mu.cpu())
            all_sigma.append(sigma.cpu())

    all_mu = torch.cat(all_mu, dim=0).numpy()
    all_sigma = torch.cat(all_sigma, dim=0).numpy()

    # Dimensionality reduction to ensure mu is on x-axis and sigma on y-axis
    pca = PCA(n_components=2)
    mu_reduced = pca.fit_transform(all_mu)
    sigma_reduced = pca.fit_transform(all_sigma)

    plt.figure(figsize=(8, 6))
    plt.scatter(mu_reduced[:, 0], sigma_reduced[:, 1], alpha=0.6, c="blue", label="Posterior Parameters")
    plt.xlabel("Reduced $\mu$ (x-axis)")
    plt.ylabel("Reduced $\sigma$ (y-axis)")
    plt.title("Posterior Parameters Visualization")
    plt.legend()
    plt.grid()
    plt.show()
