import torch
import torch.nn as nn
from torch.nn import functional as F


class EuclideanVAE(nn.Module):
    """
    Variational Autoencoder (VAE) with a Euclidean latent space ℝ^n.

    The encoder maps input data x ∈ ℝ^{data_dim} to the parameters of a
    Gaussian posterior q(z|x) over latent vectors z ∈ ℝ^{latent_dim},
    represented by a mean vector and a diagonal log-variance vector.
    Sampling is performed using the reparameterization trick to allow
    gradient-based optimization.

    The decoder reconstructs the input from the latent sample z.
    """

    def __init__(self, config):
        """
        Initialize the Euclidean VAE.

        Args:
            config: Configuration object with the following required attributes:
                * data_dim (int): Dimensionality of the input data.
                * latent_dim (int): Dimensionality of the Euclidean latent space.
                * activation (str): Activation function ("relu" or "softplus").
                * sftbeta (float): Beta parameter for the Softplus activation
                  (used only if activation is "softplus").
                * encoder_widths (List[int]): Hidden layer sizes of the encoder.
                * decoder_widths (List[int]): Hidden layer sizes of the decoder.

        Raises:
            NotImplementedError: If an unsupported activation is specified.
        """
        super().__init__()
        self.type = "gaussian_vae"
        self.data_dim = config.data_dim
        self.latent_dim = config.latent_dim

        # Activation function
        if config.activation == "relu":
            self.activation = F.relu
        elif config.activation == "softplus":
            self.sftbeta = config.sftbeta
            self.activation = lambda x: F.softplus(x, beta=self.sftbeta)
        else:
            raise NotImplementedError("Unsupported activation function.")

        encoder_widths = config.encoder_widths
        decoder_widths = config.decoder_widths

        # Encoder network
        self.encoder_linears = nn.ModuleList()
        in_dim = self.data_dim
        for out_dim in encoder_widths:
            self.encoder_linears.append(nn.Linear(in_dim, out_dim))
            in_dim = out_dim

        # Output layers for posterior parameters
        self.fc_mu = nn.Linear(in_dim, self.latent_dim)  # Mean of q(z|x)
        self.fc_logvar = nn.Linear(in_dim, self.latent_dim)  # Log-variance of q(z|x)

        # Decoder network
        self.decoder_linears = nn.ModuleList()
        in_dim = self.latent_dim
        for out_dim in decoder_widths:
            self.decoder_linears.append(nn.Linear(in_dim, out_dim))
            in_dim = out_dim
        self.fc_x_recon = nn.Linear(in_dim, self.data_dim)

    def encode(self, x):
        """
        Encode input data into posterior parameters.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, data_dim).

        Returns:
            Tuple[torch.Tensor, torch.Tensor]:
                * mu: Mean of the approximate posterior q(z|x), shape
                  (batch_size, latent_dim).
                * logvar: Log-variance of q(z|x), shape (batch_size, latent_dim).
        """
        h = x
        for layer in self.encoder_linears:
            h = self.activation(layer(h))
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

    @staticmethod
    def reparameterize(mu, logvar):
        """
        Sample latent vectors using the reparameterization trick.

        Args:
            mu (torch.Tensor): Mean of the posterior, shape (batch_size, latent_dim).
            logvar (torch.Tensor): Log-variance of the posterior, shape
                (batch_size, latent_dim).

        Returns:
            torch.Tensor: Sampled latent vector z of shape (batch_size, latent_dim).
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        """
        Decode latent vectors back to the input space.

        Args:
            z (torch.Tensor): Latent vector of shape (batch_size, latent_dim).

        Returns:
            torch.Tensor: Reconstructed input of shape (batch_size, data_dim).
        """
        h = z
        for layer in self.decoder_linears:
            h = self.activation(layer(h))
        x_recon = self.fc_x_recon(h)
        return x_recon

    def forward(self, x):
        """
        Perform a full forward pass through the VAE.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, data_dim).

        Returns:
            Tuple[torch.Tensor, torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
                * z: Sampled latent vector of shape (batch_size, latent_dim).
                * x_recon: Reconstructed input of shape (batch_size, data_dim).
                * posterior_params: Tuple (mu, logvar) of the approximate
                  posterior parameters.
        """
        posterior_params = self.encode(x)
        mu, logvar = posterior_params
        z = self.reparameterize(mu, logvar)
        x_recon = self.decode(z)
        return z, x_recon, posterior_params
