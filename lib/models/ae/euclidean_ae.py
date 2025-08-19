import torch
import torch.nn as nn
from torch.nn import functional as F


class EuclideanAE(nn.Module):
    """
    Standard autoencoder with Euclidean latent space.

    The encoder maps input data into a latent vector in R^{latent_dim}.
    The decoder reconstructs the input from the latent vector.
    """
    def __init__(self, config):
        super().__init__()
        self.type = "euclidean_ae"
        self.data_dim = config.data_dim        # Input dimension
        self.latent_dim = config.latent_dim    # Latent dimension
        if config.activation == "relu":
            self.activation = F.relu
        elif config.activation == "softplus":
            self.sftbeta = config.sftbeta
            self.activation = lambda x: F.softplus(x, beta=self.sftbeta)
        else:
            raise NotImplementedError

        encoder_widths = config.encoder_widths      # Encoder widths, implicitly defining network depth by array length
        decoder_widths = config.decoder_widths      # Decoder widths, implicitly defining network depth by array length

        # Encoder network
        self.encoder_linears = nn.ModuleList()
        in_dim = self.data_dim
        for out_dim in encoder_widths:
            self.encoder_linears.append(nn.Linear(in_dim, out_dim))
            in_dim = out_dim
        self.fc_final_encoder = nn.Linear(in_dim, self.latent_dim)

        # Decoder network
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
        z0 = self.encode(x)
        z = z0  # Identity embedding, to align with spherical ae and toroidal ae
        x_recon = self.decode(z)

        return z0, z, x_recon
