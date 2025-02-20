import torch
import torch.nn as nn

from ..distributions import VonMisesFisher, HypersphericalUniform


def elbo(posterior_type, x, x_recon, posterior_params, latent_dim, recon_loss, device="cpu"):
    if posterior_type == "gaussian":
        z_mu, z_logvar = posterior_params
        z_var = torch.exp(z_logvar)
        kl_loss = -0.5 * torch.sum(1 + z_logvar - z_mu.pow(2) - z_var)

    elif posterior_type == "hyperspherical":
        z_mu, z_kappa = posterior_params
        q_z = VonMisesFisher(z_mu, z_kappa)
        p_z = HypersphericalUniform(
            latent_dim - 1, device=device
        )
        kl_loss = torch.distributions.kl.kl_divergence(q_z, p_z).mean()

    elif posterior_type == "old_toroidal":
        z_theta_mu, z_theta_kappa, z_phi_mu, z_phi_kappa = posterior_params
        q_z_theta = VonMisesFisher(z_theta_mu, z_theta_kappa)
        q_z_phi = VonMisesFisher(z_phi_mu, z_phi_kappa)
        p_z = HypersphericalUniform(
            latent_dim - 1, device=device
        )
        kld_theta = torch.distributions.kl.kl_divergence(q_z_theta, p_z).mean()
        kld_phi = torch.distributions.kl.kl_divergence(q_z_phi, p_z).mean()
        kl_loss = kld_theta + kld_phi

    elif posterior_type == "toroidal":
        kl_loss = 0
        p_z = HypersphericalUniform(latent_dim - 1, device=device)  # Prior distribution

        for i in range(latent_dim):
            mu = posterior_params[:, i, :2]  # Mean direction for the vMF distribution
            kappa = posterior_params[:, i, 2].unsqueeze(-1)  # Concentration parameter for the vMF distribution
            q_z = VonMisesFisher(mu, kappa)  # Posterior distribution
            kl_loss += torch.distributions.kl.kl_divergence(q_z, p_z).mean()

    # TODO
    # elif posterior_type == "wrapped_normal":

    else:
        print(posterior_type, posterior_params)
        raise NotImplementedError

    if recon_loss == "BCE":
        recon_loss = nn.functional.binary_cross_entropy(x_recon, x, reduction="sum")
    else:
        recon_loss = nn.functional.mse_loss(x_recon, x, reduction="sum")


    # TODO
    # implement various reconstruction loss

    elbo_loss = (recon_loss + kl_loss)
    return elbo_loss, recon_loss, kl_loss
