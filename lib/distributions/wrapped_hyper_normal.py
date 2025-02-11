import torch
from geoopt.manifolds.lorentz import Lorentz


class WrappedHyperNormal(torch.distributions.Distribution):
    def __init__(self, mu, sigma, k):
        # Initialize parameters
        self.mu = mu
        self.sigma = sigma
        self.dim = mu.shape[-1]  # Assumes mu has the shape (batch_size, dim)
        self.k = k  # Curvature
        self.man = Lorentz(k=k)
        # The Lorentz manifold-specific logic can go here.
        # For example, initializing any Lorentzian geometry parameters if necessary.
        super().__init__()

    def rsample(self, sample_shape=torch.Size()):
        # Sample v from normal distribution
        v = self.__sample_normal(0, self.sigma)

        # Interpret v as element in the tangent space at zero: T_0 H^n
        v = self.__add_zero_to_tensor(v)

        # Move v to the tangent space at mu: T_mu H^n
        mu_end = self.man.projx(self.mu)
        transported_vector = self.man.transp0(mu_end, v)

        # Project the moved vector to the manifold
        return self.man.projx(transported_vector)

    @staticmethod
    def __sample_normal(mean, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mean + eps * std

    @staticmethod
    def __add_zero_to_tensor(v):
        tangent_tensor = torch.cat([torch.tensor([0.0]), v])
        return tangent_tensor
