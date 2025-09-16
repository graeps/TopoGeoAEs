import torch
import torch.nn as nn
from torch.nn import functional as F


class SphericalAE(nn.Module):
    """
    Autoencoder with a spherical latent space.

    Depending on the configuration, the encoder produces either
    angular coordinates or normalized vectors lying on a sphere:

    * ``latent_dim == 2`` → latent space is the unit circle S¹ ⊂ ℝ².
    * ``latent_dim == 3`` → latent space is the unit sphere S² ⊂ ℝ³.

    Two encoding modes:
    * ``normalize=True``: Encode directly to ℝ^n and normalize to S^{n−1}.
    * ``normalize=False``: Encode to angles and project to the sphere.

    The decoder reconstructs inputs from the spherical latent vector.
    """

    def __init__(self, config):
        """
        Initialize the spherical autoencoder.

        Args:
            config: Configuration object with the following required attributes:
                * embedding_dim (int): Dimensionality of the input data.
                * latent_dim (int): Dimension of the spherical latent space
                  (must be 2 for S¹ or 3 for S²).
                * use_angle_constraint (bool): If True, constrain angles to
                  standard intervals.
                * normalize (bool): If True, encode directly to ℝ^n and
                  normalize to the sphere; if False, encode to angles.
                * activation (str): Activation function ("relu" or "softplus").
                * sftbeta (float): Beta parameter for Softplus activation
                  (only used if activation is "softplus").
                * encoder_widths (List[int]): Hidden layer sizes of the encoder.
                * decoder_widths (List[int]): Hidden layer sizes of the decoder.

        Raises:
            NotImplementedError: If an unsupported activation or latent_dim
            is specified.
        """
        super().__init__()
        self.type = "spherical_ae"
        self.data_dim = config.embedding_dim
        self.latent_dim = config.latent_dim
        self.use_angle_constraint = config.use_angle_constraint
        self.normalize = config.normalize

        # Activation function
        if config.activation == "relu":
            self.activation = F.relu
        elif config.activation == "softplus":
            self.sftbeta = config.sftbeta
            self.activation = lambda x: F.softplus(x, beta=self.sftbeta)
        else:
            raise NotImplementedError("Unsupported activation function.")

        # Encoder network
        encoder_widths = config.encoder_widths
        self.encoder_linears = nn.ModuleList()
        in_dim = self.data_dim
        for out_dim in encoder_widths:
            self.encoder_linears.append(nn.Linear(in_dim, out_dim))
            in_dim = out_dim

        # Two alternative final encoder layers:
        # - Angle mode: output latent_dim-1 angles
        # - Normalized mode: output latent_dim-dimensional vector
        self.fc_final_encoder = nn.Linear(in_dim, self.latent_dim - 1)
        self.fc_final_encoder_norm = nn.Linear(in_dim, self.latent_dim)

        # Decoder network
        decoder_widths = config.decoder_widths
        self.decoder_linears = nn.ModuleList()
        in_dim = self.latent_dim
        for out_dim in decoder_widths:
            self.decoder_linears.append(nn.Linear(in_dim, out_dim))
            in_dim = out_dim
        self.fc_x_recon = nn.Linear(in_dim, self.data_dim)

    def encode(self, x):
        """
        Encode input data into angular or normalized spherical coordinates.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, data_dim).

        Returns:
            torch.Tensor:
                * If ``normalize=True``: Normalized latent vector on S^{latent_dim−1}
                  of shape (batch_size, latent_dim).
                * If ``normalize=False``: Raw or constrained angles of shape
                  (batch_size, latent_dim-1).
        """
        h = x
        for layer in self.encoder_linears:
            h = self.activation(layer(h))

        if self.normalize:
            z = self.fc_final_encoder_norm(h)
            z = z / torch.norm(z, dim=-1, keepdim=True)
            return z

        raw_angles = self.fc_final_encoder(h)

        if self.use_angle_constraint:
            if self.latent_dim == 2:
                # Single angle θ in (−π, π)
                angles = torch.pi * torch.tanh(raw_angles)
            elif self.latent_dim == 3:
                # φ ∈ (0, π), θ ∈ (−π, π)
                phi_raw = raw_angles[..., 0]
                theta_raw = raw_angles[..., 1]
                phi = 0.5 * torch.pi * (torch.tanh(phi_raw) + 1.0)
                theta = torch.pi * torch.tanh(theta_raw)
                angles = torch.stack((phi, theta), dim=-1)
            else:
                raise NotImplementedError(
                    "Angle constraint for latent_dim > 3 not implemented."
                )
        else:
            angles = raw_angles
        return angles

    def decode(self, z):
        """
        Decode latent spherical coordinates back to the input space.

        Args:
            z (torch.Tensor): Spherical latent vector of shape
                (batch_size, latent_dim).

        Returns:
            torch.Tensor: Reconstructed input of shape (batch_size, data_dim).
        """
        h = z
        for layer in self.decoder_linears:
            h = self.activation(layer(h))
        h = self.fc_x_recon(h)
        return h

    def forward(self, x):
        """
        Perform a full forward pass through the spherical autoencoder.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, data_dim).

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
                * angles: Raw angle representation (if ``normalize=False``)
                  or normalized latent vector (if ``normalize=True``).
                * z: Spherical latent vector on S^{latent_dim−1}.
                * x_recon: Reconstructed input of shape (batch_size, data_dim).
        """
        angles = self.encode(x)
        if self.normalize:
            z = angles
        else:
            z = self._project_to_sphere(angles)
        x_recon = self.decode(z)
        return angles, z, x_recon

    def _project_to_sphere(self, angles):
        """
        Convert angular coordinates to points on S¹ or S².

        Args:
            angles (torch.Tensor):
                * For S¹: tensor of shape (batch_size, 1) with angle θ.
                * For S²: tensor of shape (batch_size, 2) with angles (φ, θ).

        Returns:
            torch.Tensor:
                Coordinates on S¹ (shape (batch_size, 2)) or
                S² (shape (batch_size, 3)).
        """
        if self.latent_dim == 2:
            theta = angles.squeeze(-1)
            x_coord = torch.cos(theta)
            y_coord = torch.sin(theta)
            coords = torch.stack((x_coord, y_coord), dim=-1)
        elif self.latent_dim == 3:
            phi = angles[..., 0]
            theta = angles[..., 1]
            x_coord = torch.sin(phi) * torch.cos(theta)
            y_coord = torch.sin(phi) * torch.sin(theta)
            z_coord = torch.cos(phi)
            coords = torch.stack((x_coord, y_coord, z_coord), dim=-1)
        else:
            raise NotImplementedError(
                "Projection only implemented for latent_dim 2 or 3."
            )
        return coords
