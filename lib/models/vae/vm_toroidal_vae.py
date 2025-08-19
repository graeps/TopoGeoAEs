import torch
from torch.nn import functional as F
import torch.nn as nn

from ...distributions import VonMisesFisher


class VMToroidalVAE(torch.nn.Module):
    """
    Variational Autoencoder with a toroidal latent space: T^d = (S^1)^d.
    Extensively modified version of the one used in F. Acosta et al. 2023. "Quantifying Extrinsic Curvature in Neural Manifolds".

    The latent space is the product of d unit circles, modeled via independent
    von Mises (vM) posteriors on S^1 embedded in R^2.

    The encoder maps input x ∈ ℝ^d to posterior parameters of q(z|x),
    where each z_i ∈ S^1 ⊂ R^2 for i = 1,...,d.

    The decoder reconstructs input from the sampled toroidal latent vector z,
    which is flattened is R^{2d} or, for d = 2, embedded via S^1 x S^1 -> T^2 ⊂ R^3.
    """

    def __init__(
            self,
            config
    ):
        super().__init__()
        self.type = "vm_toroidal_vae"
        self.data_dim = config.embedding_dim  # Input dimension
        self.latent_dim = config.latent_dim  # Here latent_dim = d for T^d latent space (manifold dim)
        if config.activation == "relu":
            self.activation = F.relu
        elif config.activation == "softplus":
            self.sftbeta = config.sftbeta
            self.activation = lambda x: F.softplus(x, beta=self.sftbeta)
        else:
            raise NotImplementedError

        encoder_widths = config.encoder_widths  # Encoder layer widths
        decoder_widths = config.decoder_widths  # Decoder layer widths

        # Encoder network
        self.encoder_linears = nn.ModuleList()
        in_dim = self.data_dim
        for out_dim in encoder_widths:
            self.encoder_linears.append(nn.Linear(in_dim, out_dim))
            in_dim = out_dim

        # Posterior parameters:
        # mu in R^{d×2} (mean vectors, d vectors in R^2), kappa in R^d (concentration parameters, d scalars)
        self.fc_z_mu = nn.Linear(in_dim, 2 * self.latent_dim)
        self.fc_z_kappa = nn.Linear(in_dim, self.latent_dim)

        # Decoder network
        self.decoder_linears = nn.ModuleList()
        if self.latent_dim == 2:
            in_dim = 3  # For T^2 embedded in R^3
        else:
            in_dim = 2 * self.latent_dim  # Flattened product of S^1s
        for out_dim in decoder_widths:
            self.decoder_linears.append(nn.Linear(in_dim, out_dim))
            in_dim = out_dim
        self.fc_x_recon = nn.Linear(in_dim, self.data_dim)

    def encode(self, x):
        h = x
        for layer in self.encoder_linears:
            h = self.activation(layer(h))

        mu = self.fc_z_mu(h).view(-1, self.latent_dim, 2)  # [batch_size, latent_dim, 2]
        kappa = self.activation(self.fc_z_kappa(h)) + 1  # [batch_size, latent_dim], +1 added as in Acosta's version

        posterior_params = torch.cat((mu, kappa.unsqueeze(-1)), dim=-1)  # [batch_size, latent_dim, 3]
        return posterior_params

    def reparameterize(self, posterior_params):
        """
        Reparameterize using samples from vMF posteriors on S^1.

        Args:
            posterior_params: Tensor of shape [batch_size, latent_dim, 3] = (mu, kappa) for each S^1

        Returns:
            z: Sampled latent vector
                - If d > 2: shape [batch_size, 2*latent_dim], flattened product of S^1s
                - If d == 2: shape [batch_size, 3], embedded T^2 subset of R^3
        """
        mu, kappa = posterior_params[:, :, :2], posterior_params[:, :, 2:]
        q_z = VonMisesFisher(mu, kappa)  # Product of vM distributions
        z = q_z.rsample()  # shape: [batch_size, latent_dim, 2], z in (S^1)^latent_dim
        z_flat = z.view(-1, self.latent_dim * 2)

        # Apply torus embedding only when latent_dim == 2
        if self.latent_dim == 2:
            z_theta = z[:, 0, :]  # [batch_size, 2]
            z_phi = z[:, 1, :]  # [batch_size, 2]
            z_torus = self._build_torus(z_theta, z_phi)
            return z_torus  # shape: [batch_size, 3] – embedded in ℝ³

        return z_flat  # shape: [batch_size, latent_dim * 2]

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

    def _build_torus(self, z_theta, z_phi):
        # theta = torch.atan2(z_theta[:, 1] / z_theta[:, 0])
        # phi = torch.atan2(z_phi[:, 1] / z_phi[:, 0])
        cos_theta = z_theta[:, 0]
        sin_theta = z_theta[:, 1]

        cos_phi = z_phi[:, 0]
        sin_phi = z_phi[:, 1]

        major_radius = 2
        minor_radius = 1

        # Parametrization of standard torus in ℝ³
        x = (major_radius - minor_radius * cos_theta) * cos_phi
        y = (major_radius - minor_radius * cos_theta) * sin_phi
        z = minor_radius * sin_theta

        return torch.stack([x, y, z], dim=-1)
