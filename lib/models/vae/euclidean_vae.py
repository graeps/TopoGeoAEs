import torch
import torch.nn as nn
from torch.nn import functional as F

from ..utils.valid_config import is_valid_model_config


class EuclideanVAE(nn.Module):
    def __init__(
            self,
            config
    ):
        is_valid_model_config(config)
        super().__init__()
        self.posterior_type = "gaussian"
        self.data_dim = config["data_dim"]
        self.sftbeta = config["sftbeta"]
        self.latent_dim = config["latent_dim"]

        self.activation = F.relu

        self.encoder_width = config["encoder_width"]
        self.encoder_depth = config["encoder_depth"]
        self.decoder_width = config["decoder_width"]
        self.decoder_depth = config["decoder_depth"]

        self.encoder_flatten = nn.Flatten()
        self.encoder_fc = torch.nn.Linear(self.data_dim, self.encoder_width)

        self.encoder_linears = torch.nn.ModuleList(
            [
                torch.nn.Linear(self.encoder_width, self.encoder_width)
                for _ in range(self.encoder_depth)
            ]
        )

        self.fc_mu = nn.Linear(self.encoder_width, self.latent_dim)
        self.fc_logvar = nn.Linear(self.encoder_width, self.latent_dim)  # Diagonal covariance

        self.decoder_fc = torch.nn.Linear(self.latent_dim, self.decoder_width)
        self.decoder_linears = torch.nn.ModuleList(
            [
                torch.nn.Linear(self.decoder_width, self.decoder_width)
                for _ in range(self.decoder_depth)
            ]
        )

        self.fc_x_recon = torch.nn.Linear(self.decoder_width, self.data_dim)

    def encode(self, x):
        h = self.encoder_flatten(x)
        h = self.activation(self.encoder_fc(h))

        for layer in self.encoder_linears:
            h = self.activation(layer(h))

        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

    @staticmethod
    def reparameterize(mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h = self.activation(self.decoder_fc(z))

        for layer in self.decoder_linears:
            h = self.activation(layer(h))

        h = nn.functional.sigmoid(self.fc_x_recon(h))
        return h

    def forward(self, x):
        posterior_params = self.encode(x)
        mu, logvar = posterior_params
        z = self.reparameterize(mu, logvar)
        x_recon = self.decode(z)
        return z, x_recon, posterior_params
