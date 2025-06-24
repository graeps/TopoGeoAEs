import torch
from torch.nn import functional as F
import torch.nn as nn

from ...distributions import VonMisesFisher


class VMFSphericalVAE(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.posterior_type = "vmf_spherical"
        self.data_dim = config.embedding_dim
        self.latent_dim = config.latent_dim
        if config.activation == "relu":
            self.activation = F.relu
        elif config.activation == "softplus":
            self.sftbeta = config.sftbeta
            self.activation = lambda x: F.softplus(x, beta=self.sftbeta)
        else:
            raise NotImplementedError

        encoder_widths = config.encoder_widths
        decoder_widths = config.decoder_widths

        self.encoder_linears = nn.ModuleList()
        in_dim = self.data_dim
        for out_dim in encoder_widths:
            self.encoder_linears.append(nn.Linear(in_dim, out_dim))
            in_dim = out_dim

        self.fc_z_mu = torch.nn.Linear(in_dim, self.latent_dim)
        self.fc_z_kappa = torch.nn.Linear(in_dim, 1)

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
        z_mu = z_mu / z_mu.norm(dim=-1, keepdim=True)
        z_kappa = F.softplus(self.fc_z_kappa(h)) + 1

        #print("z_mu, z_kappa", z_mu, z_kappa)

        return z_mu, z_kappa

    def reparameterize(self, posterior_params):
        z_mu, z_kappa = posterior_params
        q_z = VonMisesFisher(z_mu, z_kappa)
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
