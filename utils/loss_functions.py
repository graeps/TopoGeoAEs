import torch
import torch.nn as nn

from ..distributions import VonMisesFisher, HypersphericalUniform


def elbo_gaussian(reconstructed, x, mu, logvar):
    """
    Computes the VAE loss function as the sum of reconstruction loss and KL divergence.

    Args:
        reconstructed (Tensor): The reconstructed output from the decoder.
        x (Tensor): The original input.
        mu (Tensor): Mean of the latent Gaussian distribution.
        logvar (Tensor): Log-variance of the latent Gaussian distribution.

    Returns:
        tuple: Total loss, reconstruction loss, KL divergence loss.
    """
    recon_loss = nn.functional.binary_cross_entropy(reconstructed, x, reduction="sum")
    kld_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + kld_loss, recon_loss, kld_loss


def elbo(posterior_type, x, x_recon, posterior_params, latent_dim, device):
    if posterior_type == "gaussian":
        z_mu, z_logvar = posterior_params
        z_var = torch.exp(z_logvar)
        kld = -0.5 * torch.sum(1 + z_logvar - z_mu.pow(2) - z_var)

    elif posterior_type == "hyperspherical":
        z_mu, z_kappa = posterior_params
        q_z = VonMisesFisher(z_mu, z_kappa)
        p_z = HypersphericalUniform(
            latent_dim - 1, device=device
        )
        kld = torch.distributions.kl.kl_divergence(q_z, p_z).mean()

    elif posterior_type == "toroidal":
        z_theta_mu, z_theta_kappa, z_phi_mu, z_phi_kappa = posterior_params
        q_z_theta = VonMisesFisher(z_theta_mu, z_theta_kappa)
        q_z_phi = VonMisesFisher(z_phi_mu, z_phi_kappa)
        p_z = HypersphericalUniform(
            latent_dim - 1, device=device
        )
        kld_theta = torch.distributions.kl.kl_divergence(q_z_theta, p_z).mean()
        kld_phi = torch.distributions.kl.kl_divergence(q_z_phi, p_z).mean()
        kld = kld_theta + kld_phi

    else:
        print(posterior_type, posterior_params)
        raise NotImplementedError

    mse_loss = nn.MSELoss()
    recon_loss = mse_loss(x_recon, x)

    elbo_loss = (recon_loss + kld)
    return elbo_loss, recon_loss, kld
