# Modified version of Acosta's paper. Latent Space actual n-Torus S^1 x ... x S^1
import torch
from torch.nn import functional as F
import torch.nn as nn

from ...distributions import VonMisesFisher


class VMToroidalVAE(torch.nn.Module):
    def __init__(
            self,
            config
    ):
        super().__init__()
        self.posterior_type = "vm_toroidal"
        self.data_dim = config.data_dim
        self.latent_dim = config.latent_dim  # Here latent_dim = d for T^d latent space (manifold dim)
        self.sftbeta = config.sftbeta
        self.activation = F.softplus  # TODO: chose activation as parameter in config

        encoder_widths = config.encoder_widths
        decoder_widths = config.decoder_widths

        self.encoder_linears = nn.ModuleList()
        in_dim = self.data_dim
        for out_dim in encoder_widths:
            self.encoder_linears.append(nn.Linear(in_dim, out_dim))
            in_dim = out_dim

        self.fc_z_mu = nn.Linear(in_dim, 2 * self.latent_dim)
        self.fc_z_kappa = nn.Linear(in_dim, self.latent_dim)

        self.decoder_linears = nn.ModuleList()
        in_dim = self.latent_dim * 2
        for out_dim in decoder_widths:
            self.decoder_linears.append(nn.Linear(in_dim, out_dim))
            in_dim = out_dim

        self.fc_x_recon = nn.Linear(in_dim, self.data_dim)

    def encode(self, x):
        h = x
        for layer in self.encoder_linears:
            h = self.activation(layer(h), beta=self.sftbeta)

        mu = self.fc_z_mu(h).view(-1, self.latent_dim, 2)  # [batch_size, latent_dim, 2]
        kappa = self.activation(self.fc_z_kappa(h), beta=self.sftbeta) + 1  # [batch_size, latent_dim]

        posterior_params = torch.cat((mu, kappa.unsqueeze(-1)), dim=-1)  # [batch_size, latent_dim, 3]
        return posterior_params

    def reparameterize(self, posterior_params):
        # Split into mu and kappa. mu.shape=[batch_size, latent_dim, 2], kappa.shape=[batch_size, latent_dim, 1]
        mu, kappa = posterior_params[:, :, :2], posterior_params[:, :, 2:]
        q_z = VonMisesFisher(mu, kappa)  # 2D vMF distribution
        z = q_z.rsample()  # [batch_size,latent_dim,2]
        z = z.view(-1, self.latent_dim * 2)  # Flatten latent dimensions [batch_size, latent_dim*2]
        return z

    def decode(self, z):
        h = z
        for layer in self.decoder_linears:
            h = self.activation(layer(h), beta=self.sftbeta)

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

        x = (major_radius - minor_radius * cos_theta) * cos_phi
        y = (major_radius - minor_radius * cos_theta) * sin_phi
        z = minor_radius * sin_theta

        return torch.stack([x, y, z], dim=-1)
