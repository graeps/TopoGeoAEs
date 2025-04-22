import torch
import torch.nn as nn
from torch.nn import functional as F


class EuclideanVAE(nn.Module):
    def __init__(
            self,
            config
    ):
        super().__init__()
        self.posterior_type = "gaussian"
        self.data_dim = config.data_dim
        self.sftbeta = config.sftbeta
        self.latent_dim = config.latent_dim
        self.activation = F.softplus

        encoder_widths = config.encoder_widths
        decoder_widths = config.decoder_widths

        self.encoder_linears = nn.ModuleList()
        in_dim = self.data_dim
        for out_dim in encoder_widths:
            self.encoder_linears.append(nn.Linear(in_dim, out_dim))
            in_dim = out_dim

        self.fc_mu = nn.Linear(in_dim, self.latent_dim)
        self.fc_logvar = nn.Linear(in_dim, self.latent_dim)  # Diagonal covariance

        self.decoder_linears = nn.ModuleList()
        in_dim = self.latent_dim
        for out_dim in decoder_widths:
            self.decoder_linears.append(nn.Linear(in_dim, out_dim))
            in_dim = out_dim

        self.fc_x_recon = torch.nn.Linear(in_dim, self.data_dim)

    def encode(self, x):
        h = x
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
        h = z
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
