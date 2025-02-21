import torch
from torch.nn import functional as F
import torch.nn as nn

from .utils.valid_config import is_valid_model_config
from ..distributions import VonMisesFisher


class ToroidalVAE(torch.nn.Module):
    def __init__(
            self,
            config
    ):
        is_valid_model_config(config)
        super().__init__()
        self.posterior_type = "toroidal"
        self.data_dim = config["data_dim"]
        self.sftbeta = config["sftbeta"]
        self.latent_dim = config["latent_dim"]
        self.encoder_width = config["encoder_width"]
        self.encoder_depth = config["encoder_depth"]
        self.decoder_width = config["decoder_width"]
        self.decoder_depth = config["decoder_depth"]

        self.encoder_flatten = nn.Flatten()
        self.encoder_fc = nn.Linear(self.data_dim, self.encoder_width)
        self.encoder_linears = nn.ModuleList(
            [
                nn.Linear(self.encoder_width, self.encoder_width)
                for _ in range(self.encoder_depth)
            ]
        )

        self.fc_z_mu = nn.Linear(self.encoder_width, 2 * self.latent_dim)
        self.fc_z_kappa = nn.Linear(self.encoder_width, self.latent_dim)

        self.decoder_fc = nn.Linear(self.latent_dim * 2, self.decoder_width)
        self.decoder_linears = nn.ModuleList(
            [
                nn.Linear(self.decoder_width, self.decoder_width)
                for _ in range(self.decoder_depth)
            ]
        )

        self.fc_x_recon = nn.Linear(self.decoder_width, self.data_dim)

    def encode(self, x):
        h = self.encoder_flatten(x)
        h = F.softplus(self.encoder_fc(h), beta=self.sftbeta)

        for layer in self.encoder_linears:
            h = F.softplus(layer(h), beta=self.sftbeta)

        mu = self.fc_z_mu(h).view(-1, self.latent_dim, 2)  # [batch_size, latent_dim, 2]
        kappa = F.softplus(self.fc_z_kappa(h), beta=self.sftbeta) + 1  # [batch_size, latent_dim]

        posterior_params = torch.cat((mu, kappa.unsqueeze(-1)), dim=-1)  # [batch_size, latent_dim, 3]
        return posterior_params

    def reparameterize(self, posterior_params):
        mu, kappa = posterior_params[:, :, :2], posterior_params[:, :, 2:]  # Split into mu and kappa
        q_z = VonMisesFisher(mu, kappa)  # 2D vMF distribution
        z = q_z.sample().view(-1, self.latent_dim * 2)  # Flatten latent dimensions

        return z

    def decode(self, z):
        h = F.softplus(self.decoder_fc(z), beta=self.sftbeta)

        for layer in self.decoder_linears:
            h = F.softplus(layer(h), beta=self.sftbeta)

        x_recon = F.sigmoid(self.fc_x_recon(h))
        return x_recon.view(-1, 1, 28, 28)

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
