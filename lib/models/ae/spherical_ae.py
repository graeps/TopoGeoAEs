import torch
import torch.nn as nn
from torch.nn import functional as F


class SphericalAE(nn.Module):
    """
    Autoencoder constrained to a spherical latent space (S^1 or S^2 embedded in R^2 or R^3).

    Depending on the configuration, the encoder maps inputs to angular coordinates
    (on S^1 or S^2), or to normalized vectors on the sphere.

    - For latent_dim == 2: the latent space is the unit circle S^1 as subset of R^2.
    - For latent_dim == 3: the latent space is the unit sphere S^2 as subset of R^3.

    Two modes:
    - If normalize=True: encode directly to R^n and normalize to S^{n-1}.
    - If normalize=False: encode to angles and then map to sphere via parametrization.

    The decoder reconstructs input from the spherical latent vector.

    """
    def __init__(self, config):
        super().__init__()
        self.type = "spherical_ae"
        self.data_dim = config.embedding_dim           # Input data dimension
        self.latent_dim = config.latent_dim            # Must be 2 (S^1) or 3 (S^2) for angle mode
        self.use_angle_constraint = config.use_angle_constraint  # Constrain angles to standard intervals
        self.normalize = config.normalize              # Use ℓ2-normalized encoding if True

        if config.activation == "relu":
            self.activation = F.relu
        elif config.activation == "softplus":
            self.sftbeta = config.sftbeta
            self.activation = lambda x: F.softplus(x, beta=self.sftbeta)
        else:
            raise NotImplementedError

        encoder_widths = config.encoder_widths      # Encoder widths, implicitly defining network depth by array length
        decoder_widths = config.decoder_widths      # Decoder widths, implicitly defining network depth by array length

        # Encoder Network
        self.encoder_linears = nn.ModuleList()
        in_dim = self.data_dim
        for out_dim in encoder_widths:
            self.encoder_linears.append(nn.Linear(in_dim, out_dim))
            in_dim = out_dim

        # Two possible last layers for encoder:
        # - For parametrization: output (latent_dim - 1) angles
        # - For normalization: output latent_dim-dimensional vector to be normalized
        self.fc_final_encoder = nn.Linear(in_dim, self.latent_dim - 1)
        self.fc_final_encoder_norm = nn.Linear(in_dim, self.latent_dim)

        # Build decoder network
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
                angles = torch.pi * torch.tanh(raw_angles)      # Single angle theta in (-pi,pi)
            elif self.latent_dim == 3:
                phi_raw = raw_angles[..., 0]
                theta_raw = raw_angles[..., 1]
                phi = 0.5 * torch.pi * (torch.tanh(phi_raw) + 1.0)     # phi in (0, pi)
                theta = torch.pi * torch.tanh(theta_raw)               # theta in (-pi,pi)
                angles = torch.stack((phi, theta), dim=-1)
            else:
                raise NotImplementedError("Angle constraint for latent_dim > 3 not implemented.")
        else:
            angles = raw_angles     # No constrains, angles in R^{latent_dim-1}

        return angles

    def decode(self, z):
        h = z
        for layer in self.decoder_linears:
            h = self.activation(layer(h))
        h = self.fc_x_recon(h)
        return h

    def forward(self, x):
        angles = self.encode(x)
        if self.normalize:
            z = angles
        else:
            z = self._project_to_sphere(angles)
        x_recon = self.decode(z)

        return angles, z, x_recon

    def _project_to_sphere(self, angles):
        """
        Map angular coordinates to S^1 or S^2.

        Inputs:
            angles:
              - For S^1: theta in R (or constrained to (-pi,pi))
              - For S^2: (phi, theta) ∈ R^2 (or constrained to (0, pi) × (-pi,pi))

        Returns:
            coordinates z ∈ S^1 subset of R^2 or z ∈ S^2 subset of R^3
        """
        if self.latent_dim == 2:
            # S^1 embedding: (cos(theta), sin(theta))
            angles = angles.squeeze(-1)
            x_coord = torch.cos(angles)
            y_coord = torch.sin(angles)
            coords = torch.stack((x_coord, y_coord), dim=-1)
        elif self.latent_dim == 3:
            # S^2 embedding: (sin(phi) cos(theta), sin(phi) sin(theta), cos(phi)
            phi = angles[..., 0]
            theta = angles[..., 1]
            x_coord = torch.sin(phi) * torch.cos(theta)
            y_coord = torch.sin(phi) * torch.sin(theta)
            z_coord = torch.cos(phi)
            coords = torch.stack((x_coord, y_coord, z_coord), dim=-1)
        else:
            raise NotImplementedError
        return coords
