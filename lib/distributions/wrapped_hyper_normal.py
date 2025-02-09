from geoopt import Lorentz
import torch


def _sample_reparameterized_normal(mu, logvar):
    std = torch.exp(0.5 * logvar)
    eps = torch.randn_like(std)
    return mu + eps * std


def add_zero_to_tensor(tensor):
    tangent_tensor = torch.cat([torch.tensor([0.0]), tensor])
    return tangent_tensor


def parallel_transport(mu_0,mu):
 kdjkdfj