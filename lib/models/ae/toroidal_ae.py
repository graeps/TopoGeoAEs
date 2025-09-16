import torch
import torch.nn as nn
from torch.nn import functional as F


class ToroidalAE(nn.Module):
    """
    Autoencoder with latent space topologically equivalent to a 2-torus T^2 = S^1 × S^1.

    Modes:
    -------
    * normalize = False:
        - Encoder outputs 2 raw angles (theta, phi).
        - Optional tanh constraint to [-π, π].
        - Project to R^3 using standard torus embedding with radii (R, r).
        - Decoder maps from R^3 back to data.

    * normalize = True:
        - Encoder outputs 4 Euclidean coordinates.
        - Split into two pairs, normalize each pair to unit length to obtain
          a point in S^1 × S^1 ⊂ R^4.
        - Decoder maps directly from R^4 back to data.
        - No R^3 torus embedding is used.
    """

    def __init__(self, config):
        super().__init__()
        self.type = "torus_ae"
        self.data_dim = config.embedding_dim
        self.latent_dim = config.latent_dim
        self.use_angle_constraint = config.use_angle_constraint
        self.normalize = config.normalize

        # Activation
        if config.activation == "relu":
            self.activation = F.relu
        elif config.activation == "softplus":
            self.sftbeta = config.sftbeta
            self.activation = lambda x: F.softplus(x, beta=self.sftbeta)
        else:
            raise NotImplementedError

        # Torus radii for explicit R^3 embedding
        self.R = 2.0
        self.r = 1.0

        # Encoder network
        encoder_widths = config.encoder_widths
        decoder_widths = config.decoder_widths
        self.encoder_linears = nn.ModuleList()
        in_dim = self.data_dim
        for out_dim in encoder_widths:
            self.encoder_linears.append(nn.Linear(in_dim, out_dim))
            in_dim = out_dim

        # Final encoder layers
        self.fc_final_encoder = nn.Linear(in_dim, self.latent_dim - 1)  # -> 2 angles
        if self.normalize:
            self.fc_final_encoder_norm = nn.Linear(in_dim, 4)           # -> R^4 for S^1 x S^1

        # Decoder network
        dec_in_dim = 4 if self.normalize else 3
        self.decoder_linears = nn.ModuleList()
        in_dim = dec_in_dim
        for out_dim in decoder_widths:
            self.decoder_linears.append(nn.Linear(in_dim, out_dim))
            in_dim = out_dim
        self.fc_x_recon = nn.Linear(in_dim, self.data_dim)

    def encode(self, x):
        h = x
        for layer in self.encoder_linears:
            h = self.activation(layer(h))

        if self.normalize:
            # Extrinsic coordinates in R^4 normalized to S^1 x S^1
            z = self.fc_final_encoder_norm(h)
            z1, z2 = z[..., :2], z[..., 2:]
            eps = 1e-8
            z1 = z1 / (torch.norm(z1, dim=-1, keepdim=True) + eps)
            z2 = z2 / (torch.norm(z2, dim=-1, keepdim=True) + eps)
            z = torch.cat((z1, z2), dim=-1)
            return z
        else:
            # Raw angles for torus embedding in R^3
            raw_angles = self.fc_final_encoder(h)

        if self.use_angle_constraint:
            angles = torch.pi * torch.tanh(raw_angles)      # (phi,theta) in [-pi,pi]^2
        else:
            angles = raw_angles     # (phi, theta) in R^2
        return angles

    def decode(self, z):
        h = z
        for layer in self.decoder_linears:
            h = self.activation(layer(h))
        h = self.fc_x_recon(h)
        return h

    def project_to_torus(self, angles):
        # Standard torus embedding in R^3
        theta = angles[..., 0]
        phi = angles[..., 1]

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
        if self.normalize:
            z = angles                        # S^1 x S^1 in R^4
        else:
            z = self.project_to_torus(angles)  # Embedded torus in R^3
        x_recon = self.decode(z)
        return angles, z, x_recon
