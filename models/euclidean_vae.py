import torch
import torch.nn as nn
from torch.nn import functional as f


class EuclideanVAE(nn.Module):
    def __init__(
            self,
            config
    ):

        self.data_dim = config.data_dim
        self.sftbeta = config.sftbeta
        self.latent_dim = config.latent_dim
        self.encoder_width = config.encoder_width
        self.encoder_depth = config.encoder_depth
        self.decoder_width = config.decoder_width
        self.decoder_depth = config.decoder_depth

        self.encoder_fc = torch.nn.Linear(self.data_dim, config.encoder_width)
        self.encoder_linears = torch.nn.ModuleList(
            [
                torch.nn.Linear(self.encoder_width, self.encoder_width)
                for _ in range(self.encoder_depth)
            ]
        )

        self.fc_mu = nn.Linear(self.encoder_width, self.latent_dim)
        self.fc_logvar = nn.Linear(self.encoder_width, self.latent_dim)

        self.decoder_fc = torch.nn.Linear(self.latent_dim, self.decoder_width)
        self.decoder_linears = torch.nn.ModuleList(
            [
                torch.nn.Linear(self.decoder_width, self.decoder_width)
                for _ in range(self.decoder_depth)
            ]
        )

        self.fc_x_recon = torch.nn.Linear(self.decoder_width, self.data_dim)
        super().__init__()

    def encode(self, x):
        h = f.softplus(self.encoder_fc(x), beta=self.sftbeta)

        for layer in self.encoder_linears:
            h = f.softplus(layer(h), beta=self.sftbeta)

        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

    @staticmethod
    def reparameterize(mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h = f.softplus(self.decoder_fc(z), beta=self.sftbeta)

        for layer in self.decoder_linears:
            h = f.softplus(layer(h), beta=self.sftbeta)

        return self.fc_x_recon(h)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        x_recon = self.decode(z)
        return x_recon, mu, logvar
