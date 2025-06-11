import torch
import torch.nn as nn
from torch.nn import functional as F


class EuclideanAE(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.type = "euclidean_ae"
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
        self.fc_final_encoder = nn.Linear(in_dim, self.latent_dim)

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
        s = self.fc_final_encoder(h)
        return s

    def decode(self, z):
        h = z
        for layer in self.decoder_linears:
            h = self.activation(layer(h))
        h = self.fc_x_recon(h)
        return h

    def forward(self, x):
        z = self.encode(x)
        x_recon = self.decode(z)

        return z, x_recon
