import torch
import torch.nn as nn
from torch_topological.nn import SignatureLoss
from torch_topological.nn import VietorisRipsComplex

from ..distributions import VonMisesFisher, HypersphericalUniform

def elbo(type, x, z, x_recon, posterior_params, labels, config):
    latent_dim = config.latent_dim
    recon_loss = config.recon_loss
    topo_loss = config.topo_loss
    alpha = config.alpha
    beta = config.beta
    gamma = config.gamma
    device = config.device

    if type == "gaussian_vae":
        z_mu, z_logvar = posterior_params
        z_var = torch.exp(z_logvar)
        kl_loss = -0.5 * torch.sum(1 + z_logvar - z_mu.pow(2) - z_var)

    elif type == "hyperspherical":
        z_mu, z_kappa = posterior_params
        q_z = VonMisesFisher(z_mu, z_kappa)
        p_z = HypersphericalUniform(
            latent_dim - 1, device=device
        )
        kl_loss = torch.distributions.kl.kl_divergence(q_z, p_z).mean()

    elif type == "vmf_spherical_vae":
        z_theta, z_kappa = posterior_params
        q_z = VonMisesFisher(z_theta, z_kappa)
        p_z = HypersphericalUniform(latent_dim - 1, device=device)
        kl_loss = torch.distributions.kl.kl_divergence(q_z, p_z).mean()

    elif type == "vmf_toroidal_vae":
        z_theta_mu, z_theta_kappa, z_phi_mu, z_phi_kappa = posterior_params
        q_z_theta = VonMisesFisher(z_theta_mu, z_theta_kappa)
        q_z_phi = VonMisesFisher(z_phi_mu, z_phi_kappa)
        p_z = HypersphericalUniform(
            latent_dim - 1, device=device
        )
        kld_theta = torch.distributions.kl.kl_divergence(q_z_theta, p_z).mean()
        kld_phi = torch.distributions.kl.kl_divergence(q_z_phi, p_z).mean()
        kl_loss = kld_theta + kld_phi

    elif type == "vm_toroidal_vae":
        p_z = HypersphericalUniform(1, device=device)  # Prior over S^1 (circle)
        mu = posterior_params[:, :, :2]  # Shape: [batch_size, latent_dim, 2]
        kappa = posterior_params[:, :, 2:]  # Shape: [batch_size, latent_dim, 1]
        q_z = VonMisesFisher(mu, kappa)  # Posterior distribution
        kl_loss = torch.distributions.kl.kl_divergence(q_z, p_z).sum().mean()

    elif type == "mgvm_toroidal":
        low = torch.zeros(latent_dim)
        high = torch.full((latent_dim,), 2 * torch.pi)
        p_z = torch.distributions.Independent(torch.distributions.Uniform(low, high), 1)

        mu, kappa, w = posterior_params
        q_z = VonMisesFisher(mu, kappa)  # Posterior distribution
        kl_loss = torch.distributions.kl.kl_divergence(q_z, p_z).sum().mean()

    # TODO
    # elif type == "wrapped_normal":

    else:
        print(type, posterior_params)
        raise NotImplementedError

    if recon_loss == "BCE":
        recon_loss = nn.functional.binary_cross_entropy(x_recon, x, reduction="sum")
    else:
        recon_loss = nn.functional.mse_loss(x_recon, x, reduction="sum")

    if topo_loss:
        vr = VietorisRipsComplex(dim=config.dim_topo_loss)
        pi_x = vr(x)
        pi_z = vr(z)
        topo_loss = SignatureLoss(p=2, dimensions=config.dim_topo_loss)([x, pi_x], [z, pi_z])
    else:
        topo_loss = 0

    elbo_loss = (alpha * recon_loss + beta * kl_loss + gamma * topo_loss)
    return elbo_loss, recon_loss, kl_loss, topo_loss


def topo_ae_loss(config,x, z, x_recon):
    alpha = config.alpha
    gamma = config.gamma
    topo_loss = config.topo_loss

    recon_loss = nn.functional.mse_loss(x_recon, x, reduction="mean")

    if topo_loss:
        vr = VietorisRipsComplex(dim=config.dim_topo_loss)
        pi_x = vr(x)
        pi_z = vr(z)
        topo_loss = SignatureLoss(p=2, dimensions=config.dim_topo_loss)([x, pi_x], [z, pi_z])
    else:
        topo_loss = 0

    loss = (alpha * recon_loss + gamma * topo_loss)
    return loss, recon_loss, topo_loss


def latent_regularization_loss(labels, z, config):
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
    return latent_loss**2