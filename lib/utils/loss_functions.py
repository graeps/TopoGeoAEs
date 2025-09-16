import torch
import torch.nn as nn
from torch_topological.nn import SignatureLoss, VietorisRipsComplex

from ..distributions import VonMisesFisher, HypersphericalUniform


def elbo(latent_distribution, x, z, x_recon, posterior_params, config):
    """
    Compute the Evidence Lower Bound (ELBO) for a VAE with various latent distributions.

    The ELBO combines:
        1. Reconstruction loss between input x and reconstruction x_recon.
        2. KL divergence between posterior q(z|x) and prior p(z).
        3. Optional adds topological loss based on persistent homology.

    Args:
        latent_distribution (str): Type of latent distribution. Supported:
            {"gaussian_vae", "hyperspherical", "vmf_spherical_vae", "vmf_toroidal_vae"}.
        x (torch.Tensor): Input data, shape [batch_size, data_dim].
        z (torch.Tensor): Latent sample, shape [batch_size, latent_dim].
        x_recon (torch.Tensor): Reconstruction of x, same shape as x.
        posterior_params (tuple): Parameters of q(z|x), dependent on distribution:
            - "gaussian_vae": (mu, logvar)
            - "hyperspherical": (mu, kappa)
            - "vmf_spherical_vae": (mu, kappa)
            - "vmf_toroidal_vae": (theta_mu, theta_kappa, phi_mu, phi_kappa)
        config: Configuration object with attributes:
            * latent_dim (int): Dimensionality of latent space.
            * recon_loss (str): "BCE" or "MSE".
            * topo_loss (bool): Whether to compute topological loss.
            * alpha (float): Weight of reconstruction loss.
            * beta (float): Weight of KL divergence.
            * gamma (float): Weight of topological loss.
            * device (torch.device): Torch device for computations.
            * dim_topo_loss (int): Homology dimension for Vietoris–Rips complex.

    Returns:
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
            - elbo_loss: Total weighted ELBO value.
            - recon_loss: Reconstruction loss.
            - kl_loss: KL divergence term.
            - topo_loss: Topological loss term.
    """
    latent_dim = config.latent_dim
    recon_loss_type = config.recon_loss
    topo_flag = config.topo_loss
    alpha = config.alpha
    beta = config.beta
    gamma = config.gamma
    device = config.device

    # KL divergence term
    if latent_distribution == "gaussian_vae":
        z_mu, z_logvar = posterior_params
        z_var = torch.exp(z_logvar)
        kl_loss = -0.5 * torch.sum(1 + z_logvar - z_mu.pow(2) - z_var)

    elif latent_distribution == "hyperspherical":
        z_mu, z_kappa = posterior_params
        q_z = VonMisesFisher(z_mu, z_kappa)
        p_z = HypersphericalUniform(latent_dim - 1, device=device)
        kl_loss = torch.distributions.kl.kl_divergence(q_z, p_z).mean()

    elif latent_distribution == "vmf_spherical_vae":
        z_theta, z_kappa = posterior_params
        q_z = VonMisesFisher(z_theta, z_kappa)
        p_z = HypersphericalUniform(latent_dim - 1, device=device)
        kl_loss = torch.distributions.kl.kl_divergence(q_z, p_z).mean()

    elif latent_distribution == "vmf_toroidal_vae":
        z_theta_mu, z_theta_kappa, z_phi_mu, z_phi_kappa = posterior_params
        q_z_theta = VonMisesFisher(z_theta_mu, z_theta_kappa)
        q_z_phi = VonMisesFisher(z_phi_mu, z_phi_kappa)
        p_z = HypersphericalUniform(latent_dim - 1, device=device)
        kld_theta = torch.distributions.kl.kl_divergence(q_z_theta, p_z).mean()
        kld_phi = torch.distributions.kl.kl_divergence(q_z_phi, p_z).mean()
        kl_loss = kld_theta + kld_phi

    else:
        raise NotImplementedError(f"Unknown latent distribution: {latent_distribution}")

    # Reconstruction loss
    if recon_loss_type == "BCE":
        recon_loss = nn.functional.binary_cross_entropy(x_recon, x, reduction="sum")
    else:
        recon_loss = nn.functional.mse_loss(x_recon, x, reduction="sum")

    # Optional topological loss
    if topo_flag:
        vr = VietorisRipsComplex(dim=config.dim_topo_loss)
        pi_x = vr(x)
        pi_z = vr(z)
        topo_loss = SignatureLoss(p=2, dimensions=config.dim_topo_loss)([x, pi_x], [z, pi_z])
    else:
        topo_loss = 0

    elbo_loss = alpha * recon_loss + beta * kl_loss + gamma * topo_loss
    return elbo_loss, recon_loss, kl_loss, topo_loss


def topo_ae_loss(config, x, z, x_recon):
    """
    Compute loss for topological autoencoders.

    Combines:
        1. Reconstruction MSE.
        2. Optional topological signature loss.

    Args:
        config: Configuration object with attributes:
            * alpha (float): Weight of reconstruction loss.
            * gamma (float): Weight of topological loss.
            * topo_loss (bool): Whether to compute topological loss.
            * dim_topo_loss (int): Homology dimension for Vietoris–Rips complex.
        x (torch.Tensor): Input data [batch_size, data_dim].
        z (torch.Tensor): Latent representation [batch_size, latent_dim].
        x_recon (torch.Tensor): Reconstruction of x.

    Returns:
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            - total_loss: Weighted total loss.
            - recon_loss: Reconstruction loss.
            - topo_loss: Topological loss.
    """
    alpha = config.alpha
    gamma = config.gamma
    topo_flag = config.topo_loss

    recon_loss = nn.functional.mse_loss(x_recon, x, reduction="mean")

    if topo_flag:
        vr = VietorisRipsComplex(dim=config.dim_topo_loss)
        pi_x = vr(x)
        pi_z = vr(z)
        topo_loss = SignatureLoss(p=2, dimensions=config.dim_topo_loss)([x, pi_x], [z, pi_z])
    else:
        topo_loss = 0

    loss = alpha * recon_loss + gamma * topo_loss
    return loss, recon_loss, topo_loss


def latent_regularization_loss(labels, z, config):
    """
    Compute a latent regularization loss to align latent coordinates with known
    manifold parameters (e.g., angles on S^1 or S^2).

    Args:
        labels (torch.Tensor): Ground-truth manifold coordinates.
            - S^1: shape [batch_size]
            - S^2: shape [batch_size, 2] (theta, phi)
        z (torch.Tensor): Latent representation [batch_size, latent_dim].
        config: Configuration with attribute:
            * dataset_name (str): One of {"s1_low", "s2_low"}.

    Returns:
        torch.Tensor: Scalar latent regularization loss.
    """
    if config.dataset_name == "s1_low":
        latent_angles = (torch.atan2(z[:, 1], z[:, 0]) + 2 * torch.pi) % (2 * torch.pi)
        angle_loss = torch.mean(1 - torch.cos(latent_angles - labels))
        latent_loss = angle_loss

    elif config.dataset_name == "s2_low":
        latent_thetas = torch.arccos(z[:, 2])
        latent_phis = (torch.atan2(z[:, 1], z[:, 0]) + 2 * torch.pi) % (2 * torch.pi)
        thetas_loss = torch.mean(1 - torch.cos(latent_thetas - labels[:, 0]))
        phis_loss = torch.mean(
            torch.sin(latent_thetas)
            * torch.sin(labels[:, 0])
            * (1 - torch.cos(latent_phis - labels[:, 1]))
        )
        latent_loss = thetas_loss + phis_loss

    else:
        latent_loss = 0

    return latent_loss ** 2
