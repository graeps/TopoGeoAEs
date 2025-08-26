import torch
import torch.nn as nn
from torch.nn import functional as F


class ToroidalAE(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.type = "torus_ae"
        self.data_dim = config.embedding_dim
        self.latent_dim = config.latent_dim
        self.use_angle_constraint = config.use_angle_constraint
        assert self.latent_dim == 3, "TorusAE requires latent_dim == 3"
        if config.activation == "relu":
            self.activation = F.relu
        elif config.activation == "softplus":
            self.sftbeta = config.sftbeta
            self.activation = lambda x: F.softplus(x, beta=self.sftbeta)
        else:
            raise NotImplementedError

        self.R = 2.0
        self.r = 1.0

        encoder_widths = config.encoder_widths
        decoder_widths = config.decoder_widths

        self.encoder_linears = nn.ModuleList()
        in_dim = self.data_dim
        for out_dim in encoder_widths:
            self.encoder_linears.append(nn.Linear(in_dim, out_dim))
            in_dim = out_dim
        self.fc_final_encoder = nn.Linear(in_dim, self.latent_dim - 1)

        self.decoder_linears = nn.ModuleList()
        in_dim = 3  # (x, y, z) in ℝ³
        for out_dim in decoder_widths:
            self.decoder_linears.append(nn.Linear(in_dim, out_dim))
            in_dim = out_dim

        self.fc_x_recon = nn.Linear(in_dim, self.data_dim)

    def encode(self, x):
        h = x
        for layer in self.encoder_linears:
            h = self.activation(layer(h))
        raw_angles = self.fc_final_encoder(h)  # shape: [batch, 2]

        if self.use_angle_constraint:
            # θ, φ ∈ [−π, π]
            angles = torch.pi * torch.tanh(raw_angles)
        else:
            angles = raw_angles

        return angles

    def decode(self, z):
        h = z
        for layer in self.decoder_linears:
            h = self.activation(layer(h))
        return self.fc_x_recon(h)

    def project_to_torus(self, angles):
        theta = angles[..., 0]  # angle around the major circle
        phi = angles[..., 1]    # angle around the minor circle

        cos_phi = torch.cos(phi)
        sin_phi = torch.sin(phi)
        cos_theta = torch.cos(theta)
        sin_theta = torch.sin(theta)

        x = (self.R + self.r * cos_phi) * cos_theta
        y = (self.R + self.r * cos_phi) * sin_theta
        z = self.r * sin_phi

        coords = torch.stack((x, y, z), dim=-1)
        return coords

    def forward(self, x):
        angles = self.encode(x)
        z = self.project_to_torus(angles)
        x_recon = self.decode(z)
        return angles, z, x_recon
