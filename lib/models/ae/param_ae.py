import torch
import torch.nn as nn
from torch.nn import functional as F


class ParamAE(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.type = "param_ae"
        self.data_dim = config.data_dim
        self.sftbeta = config.sftbeta
        self.latent_dim = config.latent_dim
        self.manifold_dim = config.manifold_dim
        self.activation = F.softplus

        encoder_widths = config.encoder_widths
        param_widths = config.param_widths
        decoder_widths = config.decoder_widths

        self.encoder_linears = nn.ModuleList()
        in_dim = self.data_dim
        for out_dim in encoder_widths:
            self.encoder_linears.append(nn.Linear(in_dim, out_dim))
            in_dim = out_dim
        self.fc_final_encoder = nn.Linear(in_dim, self.manifold_dim)

        self.param1_linears = nn.ModuleList()
        in_dim = self.manifold_dim
        for out_dim in param_widths:
            self.param1_linears.append(nn.Linear(in_dim, out_dim))
            in_dim = out_dim
        self.fc_final_param1 = nn.Linear(in_dim, 1)

        self.param2_linears = nn.ModuleList()
        in_dim = self.manifold_dim
        for out_dim in param_widths:
            self.param2_linears.append(nn.Linear(in_dim, out_dim))
            in_dim = out_dim
        self.fc_final_param2 = nn.Linear(in_dim, 1)

        self.decoder_linears = nn.ModuleList()
        in_dim = 2
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

    def parameterize(self, s):
        h1 = s
        h2 = s
        for layer in self.param1_linears:
            h1 = self.activation(layer(h1))
        z1 = self.fc_final_param1(h1)
        for layer in self.param2_linears:
            h2 = self.activation(layer(h2))
        z2 = self.fc_final_param2(h2)
        z = torch.cat((z1, z2), 1)
        return z

    def decode(self, z):
        h = z
        for layer in self.decoder_linears:
            h = self.activation(layer(h))

        h = self.fc_x_recon(h)
        return h

    def forward(self, x):
        s = self.encode(x)
        z = self.parameterize(s)
        x_recon = self.decode(z)

        return s, z, x_recon
