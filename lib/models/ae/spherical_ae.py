import torch
import torch.nn as nn
from torch.nn import functional as F


class SphericalAE(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.type = "spherical_ae"
        self.data_dim = config.embedding_dim
        self.latent_dim = config.latent_dim
        self.use_angle_constraint = config.use_angle_constraint
        self.normalize = config.normalize
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
        self.fc_final_encoder = nn.Linear(in_dim, self.latent_dim - 1)
        self.fc_final_encoder_norm = nn.Linear(in_dim, self.latent_dim)

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

        if self.normalize:
            z = self.fc_final_encoder_norm(h)
            z = z / torch.norm(z, dim=-1, keepdim=True)
            return z

        else:
            raw_angles = self.fc_final_encoder(h)

        if self.use_angle_constraint:
            if self.latent_dim == 2:
                # One angle, map to [-π, π]
                angles = torch.pi * torch.tanh(raw_angles)
            elif self.latent_dim == 3:
                # Two angles: phi ∈ [0, π], theta ∈ [−π, π]
                phi_raw = raw_angles[..., 0]
                theta_raw = raw_angles[..., 1]
                phi = 0.5 * torch.pi * (torch.tanh(phi_raw) + 1.0)     # [0, π]
                theta = torch.pi * torch.tanh(theta_raw)               # [−π, π]
                angles = torch.stack((phi, theta), dim=-1)
            else:
                raise NotImplementedError("Angle constraint for latent_dim > 3 not implemented.")
        else:
            angles = raw_angles

        return angles

    def decode(self, z):
        h = z
        for layer in self.decoder_linears:
            h = self.activation(layer(h))
        h = self.fc_x_recon(h)
        return h

    def project_to_sphere(self, angles):
        if self.latent_dim == 2:
            angles = angles.squeeze(-1)
            x_coord = torch.cos(angles)
            y_coord = torch.sin(angles)
            coords = torch.stack((x_coord, y_coord), dim=-1)
        elif self.latent_dim == 3:
            phi = angles[..., 0]
            theta = angles[..., 1]
            x_coord = torch.sin(phi) * torch.cos(theta)
            y_coord = torch.sin(phi) * torch.sin(theta)
            z_coord = torch.cos(phi)
            coords = torch.stack((x_coord, y_coord, z_coord), dim=-1)
        else:
            raise NotImplementedError
        return coords

    def forward(self, x):
        angles = self.encode(x)
        if self.normalize:
            z = angles
        else:
            z = self.project_to_sphere(angles)
        x_recon = self.decode(z)

        return angles, z, x_recon
