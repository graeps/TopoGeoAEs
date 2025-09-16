import torch
import torch.nn as nn
from torch.nn import functional as F


class EuclideanAE(nn.Module):
    """
    Autoencoder with a Euclidean latent space.

    The encoder maps input data into a latent vector in R^latent_dim.
    The decoder reconstructs the input from this latent vector.
    """

    def __init__(self, config):
        """
        Initialize the Euclidean autoencoder.

        Args:
            config: Configuration object with the following required attributes:
                * data_dim (int): Dimensionality of the input data.
                * latent_dim (int): Dimensionality of the latent space.
                * activation (str): Activation function, either ``"relu"`` or ``"softplus"``.
                * sftbeta (float): Beta parameter for the Softplus activation (used if activation is "softplus").
                * encoder_widths (List[int]): Sizes of hidden layers for the encoder.
                * decoder_widths (List[int]): Sizes of hidden layers for the decoder.

        Raises:
            NotImplementedError: If an unsupported activation is specified.
        """
        super().__init__()
        self.type = "euclidean_ae"
        self.data_dim = config.data_dim
        self.latent_dim = config.latent_dim

        # Activation selection
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
        self.fc_final_encoder = nn.Linear(in_dim, self.latent_dim)

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
        Encode input data into the latent space.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, data_dim).

        Returns:
            torch.Tensor: Latent representation of shape (batch_size, latent_dim).
        """
        h = x
        for layer in self.encoder_linears:
            h = self.activation(layer(h))
        s = self.fc_final_encoder(h)
        return s

    def decode(self, z):
        """
        Decode latent representations back to the input space.

        Args:
            z (torch.Tensor): Latent tensor of shape (batch_size, latent_dim).

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
        Perform a full forward pass through the autoencoder.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, data_dim).

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
                * z0: Latent encoding produced by the encoder
                  (shape ``[batch_size, latent_dim]``).
                * z:  Latent embedding (identical to z0, provided for API consistency).
                * x_recon: Reconstructed input
                  (shape ``[batch_size, data_dim]``).
        """
        z0 = self.encode(x)
        z = z0  # Identity embedding, to align with spherical or toroidal AEs
        x_recon = self.decode(z)
        return z0, z, x_recon
