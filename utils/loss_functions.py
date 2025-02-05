import torch
import torch.nn as nn


def euclid_gaussian_loss(reconstructed, x, mu, logvar):
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
