import torch.nn as nn
from torch.nn import functional as F

from ...distributions import VonMisesFisher


class VMFSphericalVAE(nn.Module):
    """
    Variational Autoencoder with a spherical latent space: S^{d−1} ⊂ R^d.
    Modified version from T. Davidson et al. 2018. "Hyperspherical Variational Auto-encoders".

    The latent space is modeled as the unit (d−1)-sphere, with posterior q(z|x)
    modeled as a von Mises–Fisher (vMF) distribution on S^{d−1}.

    The encoder maps input x ∈ R^n to the posterior parameters (mu, kappa), where
    mu in S^{d−1} ⊂ R^d is a unit direction vector and kappa in R⁺ is a concentration scalar.

    The decoder reconstructs the input from the sampled spherical latent vector z ∈ S^{d−1}.
    """
    def __init__(self, config):
        super().__init__()
        self.type = "vmf_spherical_vae"                     # Model type identifier
        self.data_dim = config.embedding_dim            # Input data dimension
        self.latent_dim = config.latent_dim             # z in S^{latent_dim - 1} ⊂ R^{latent_dim}

        if config.activation == "relu":
            self.activation = F.relu
        elif config.activation == "softplus":
            self.sftbeta = config.sftbeta
            self.activation = lambda x: F.softplus(x, beta=self.sftbeta)
        else:
            raise NotImplementedError

        encoder_widths = config.encoder_widths          # Encoder widths
        decoder_widths = config.decoder_widths          # Decoder widths

        # Encoder network
        self.encoder_linears = nn.ModuleList()
        in_dim = self.data_dim
        for out_dim in encoder_widths:
            self.encoder_linears.append(nn.Linear(in_dim, out_dim))
            in_dim = out_dim

        # Posterior parameters:
        # mu in R^d normalized mean vector, kappa in R scalar concentration parameter
        self.fc_z_mu = nn.Linear(in_dim, self.latent_dim)
        self.fc_z_kappa = nn.Linear(in_dim, 1)

        # Decoder network
        self.decoder_linears = nn.ModuleList()
        in_dim = self.latent_dim
        for out_dim in decoder_widths:
            self.decoder_linears.append(nn.Linear(in_dim, out_dim))
            in_dim = out_dim
        self.fc_x_recon = nn.Linear(in_dim, self.data_dim)

    def encode(self, x):
        h = x
        for layer in self.encoder_linears:
            h = self.activation(layer(h))
        z_mu = self.fc_z_mu(h)
        z_mu = z_mu / z_mu.norm(dim=-1, keepdim=True)  # Normalize to unit norm
        z_kappa = F.softplus(self.fc_z_kappa(h)) + 1.0  # Ensure κ > 1 for numerical stability
        return z_mu, z_kappa

    def reparameterize(self, posterior_params):
        """
        Reparameterize using vMF distribution on S^{d−1}.

        Args:
            posterior_params: tuple (mu, kappa), with shapes [batch_size, d], [batch_size, 1]

        Returns:
            z: Sampled latent vector on S^{d−1}, shape [batch_size, d]
        """
        z_mu, z_kappa = posterior_params
        q_z = VonMisesFisher(z_mu, z_kappa)  # Single vMF on S^{d−1}
        z = q_z.rsample()
        return z

    def decode(self, z):
        h = z
        for layer in self.decoder_linears:
            h = self.activation(layer(h))

        x_recon = self.fc_x_recon(h)
        return x_recon

    def forward(self, x):
        posterior_params = self.encode(x)
        z = self.reparameterize(posterior_params)
        x_recon = self.decode(z)
        return z, x_recon, posterior_params
