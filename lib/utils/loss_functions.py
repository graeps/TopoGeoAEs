import torch
import torch.nn as nn

from ..distributions import VonMisesFisher, HypersphericalUniform

from torch_topological.nn import SignatureLoss
from torch_topological.nn import VietorisRipsComplex


def elbo(posterior_type, x, z, x_recon, posterior_params, config):
    latent_dim = config.latent_dim
    recon_loss = config.recon_loss
    topo_loss = config.topo_loss
    alpha = config.alpha
    beta = config.beta
    gamma = config.gamma
    device = config.device

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

    elif posterior_type == "vmf_toroidal":
        z_theta_mu, z_theta_kappa, z_phi_mu, z_phi_kappa = posterior_params
        q_z_theta = VonMisesFisher(z_theta_mu, z_theta_kappa)
        q_z_phi = VonMisesFisher(z_phi_mu, z_phi_kappa)
        p_z = HypersphericalUniform(
            latent_dim - 1, device=device
        )
        kld_theta = torch.distributions.kl.kl_divergence(q_z_theta, p_z).mean()
        kld_phi = torch.distributions.kl.kl_divergence(q_z_phi, p_z).mean()
        kl_loss = kld_theta + kld_phi

    elif posterior_type == "vm_toroidal":
        p_z = HypersphericalUniform(1, device=device)  # Prior over S^1 (circle)
        mu = posterior_params[:, :, :2]  # Shape: [batch_size, latent_dim, 2]
        kappa = posterior_params[:, :, 2:]  # Shape: [batch_size, latent_dim, 1]
        q_z = VonMisesFisher(mu, kappa)  # Posterior distribution
        kl_loss = torch.distributions.kl.kl_divergence(q_z, p_z).sum().mean()

    elif posterior_type == "mgvm_toroidal":
        low = torch.zeros(latent_dim)
        high = torch.full((latent_dim,), 2 * torch.pi)
        p_z = torch.distributions.Independent(torch.distributions.Uniform(low, high), 1)

        mu, kappa, w = posterior_params
        q_z = VonMisesFisher(mu, kappa)  # Posterior distribution
        kl_loss = torch.distributions.kl.kl_divergence(q_z, p_z).sum().mean()

    # TODO
    # elif posterior_type == "wrapped_normal":

    else:
        print(posterior_type, posterior_params)
        raise NotImplementedError

    if recon_loss == "BCE":
        recon_loss = nn.functional.binary_cross_entropy(x_recon, x, reduction="sum")
    else:
        recon_loss = nn.functional.mse_loss(x_recon, x, reduction="sum")

    if topo_loss:
        vr = VietorisRipsComplex(dim=0)
        pi_x = vr(x)
        pi_z = vr(z)
        topo_loss = SignatureLoss(p=2)([x, pi_x], [z, pi_z])
    else:
        topo_loss = 0

    elbo_loss = (alpha * recon_loss + beta * kl_loss + gamma * topo_loss)
    return elbo_loss, recon_loss, kl_loss
