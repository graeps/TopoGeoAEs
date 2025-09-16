import torch
import torch.nn as nn
from torch.nn import functional as F


class ToroidalAE(nn.Module):
    """
    Autoencoder with a latent space homeomorphic to a 2-torus T² = S¹ × S¹.

    Two encoding modes are supported:

    * ``normalize = False``:
        The encoder outputs two raw angles (θ, φ). Optionally, the angles are
        constrained to the range [-π, π] via a hyperbolic tangent. The latent
        representation is then obtained by projecting these angles to ℝ³ using
        the standard torus embedding with radii (R, r). The decoder maps from
        this ℝ³ representation back to the input space.

    * ``normalize = True``:
        The encoder outputs a 4-dimensional Euclidean vector, which is split
        into two pairs. Each pair is normalized to unit length, producing a
        point on S¹ × S¹ ⊂ ℝ⁴. The decoder maps directly from this ℝ⁴
        representation back to the input space without an explicit ℝ³ torus
        embedding.
    """

    def __init__(self, config):
        """
        Initialize the toroidal autoencoder.

        Args:
            config: Configuration object with the following required attributes:
                * embedding_dim (int): Dimensionality of the input data.
                * latent_dim (int): Latent dimensionality (must be 3 for
                  angle-based mode or 4 for normalized mode).
                * use_angle_constraint (bool): If True, constrain the predicted
                  angles to [-π, π] via a hyperbolic tangent.
                * normalize (bool): If True, use the normalized S¹ × S¹ encoding.
                * activation (str): Activation function ("relu" or "softplus").
                * sftbeta (float): Beta parameter for Softplus activation
                  (used only if activation is "softplus").
                * encoder_widths (List[int]): Hidden layer sizes for the encoder.
                * decoder_widths (List[int]): Hidden layer sizes for the decoder.

        Raises:
            NotImplementedError: If an unsupported activation is specified.
        """
        super().__init__()
        self.type = "torus_ae"
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

        # Radii of the standard torus embedding (for normalize=False mode)
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
        # Angle mode: outputs 2 raw angles
        self.fc_final_encoder = nn.Linear(in_dim, self.latent_dim - 1)
        if self.normalize:
            # Normalized mode: outputs 4 extrinsic coordinates
            self.fc_final_encoder_norm = nn.Linear(in_dim, 4)

        # Decoder network
        dec_in_dim = 4 if self.normalize else 3
        self.decoder_linears = nn.ModuleList()
        in_dim = dec_in_dim
        for out_dim in decoder_widths:
            self.decoder_linears.append(nn.Linear(in_dim, out_dim))
            in_dim = out_dim
        self.fc_x_recon = nn.Linear(in_dim, self.data_dim)

    def encode(self, x):
        """
        Encode input data into toroidal latent coordinates.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, data_dim).

        Returns:
            torch.Tensor:
                * If ``normalize=True``: Normalized latent coordinates on
                  S¹ × S¹ of shape (batch_size, 4).
                * If ``normalize=False``: Raw or constrained angles of shape
                  (batch_size, 2).
        """
        h = x
        for layer in self.encoder_linears:
            h = self.activation(layer(h))

        if self.normalize:
            # Extrinsic coordinates in ℝ⁴ normalized to S¹ × S¹
            z = self.fc_final_encoder_norm(h)
            z1, z2 = z[..., :2], z[..., 2:]
            eps = 1e-8
            z1 = z1 / (torch.norm(z1, dim=-1, keepdim=True) + eps)
            z2 = z2 / (torch.norm(z2, dim=-1, keepdim=True) + eps)
            z = torch.cat((z1, z2), dim=-1)
            return z

        # Angle mode
        raw_angles = self.fc_final_encoder(h)
        if self.use_angle_constraint:
            angles = torch.pi * torch.tanh(raw_angles)  # angles ∈ [-π, π]^2
        else:
            angles = raw_angles
        return angles

    def decode(self, z):
        """
        Decode toroidal latent coordinates back to the input space.

        Args:
            z (torch.Tensor): Latent tensor of shape
                (batch_size, 4) if ``normalize=True`` or
                (batch_size, 3) if ``normalize=False`` after torus embedding.

        Returns:
            torch.Tensor: Reconstructed input of shape (batch_size, data_dim).
        """
        h = z
        for layer in self.decoder_linears:
            h = self.activation(layer(h))
        h = self.fc_x_recon(h)
        return h

    def project_to_torus(self, angles):
        """
        Map angular coordinates to a standard torus embedding in ℝ³.

        Args:
            angles (torch.Tensor): Tensor of shape (batch_size, 2)
                containing angles (θ, φ).

        Returns:
            torch.Tensor: Embedded torus coordinates of shape (batch_size, 3).
        """
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
        """
        Perform a full forward pass through the toroidal autoencoder.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, data_dim).

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
                * angles: Raw angles (if ``normalize=False``) or
                  normalized S¹ × S¹ coordinates (if ``normalize=True``).
                * z: Latent torus embedding in ℝ³ (angle mode) or ℝ⁴ (normalized mode).
                * x_recon: Reconstructed input of shape (batch_size, data_dim).
        """
        angles = self.encode(x)
        if self.normalize:
            z = angles                        # S¹ × S¹ in ℝ⁴
        else:
            z = self.project_to_torus(angles)  # Standard torus embedding in ℝ³
        x_recon = self.decode(z)
        return angles, z, x_recon
