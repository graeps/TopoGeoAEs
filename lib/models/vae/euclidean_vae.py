import torch
import torch.nn as nn
from torch.nn import functional as F


class EuclideanVAE(nn.Module):
    """
    Variational Autoencoder with a Euclidean latent space R^n.

    The encoder maps input data x ∈ R^{data_dim} to the parameters of a Gaussian posterior q(z|x)
    over latent vectors z ∈ R^{latent_dm}, represented by mean and (diagonal) log-variance.
    Sampling is performed via the reparameterization trick.

    The decoder reconstructs the input from the latent sample z.
    """
    def __init__(self, config):
        super().__init__()
        self.type = "gaussian_vae"              # Type of posterior distribution
        self.data_dim = config.data_dim               # Input data dimension
        self.latent_dim = config.latent_dim           # Latent dimension (dim of R^n)

        # Activation function
        if config.activation == "relu":
            self.activation = F.relu
        elif config.activation == "softplus":
            self.sftbeta = config.sftbeta
            self.activation = lambda x: F.softplus(x, beta=self.sftbeta)
        else:
            raise NotImplementedError

        encoder_widths = config.encoder_widths      # Encoder widths, implicitly defining network depth by array length
        decoder_widths = config.decoder_widths      # Decoder widths, implicitly defining network depth by array length

        # Encoder network
        self.encoder_linears = nn.ModuleList()
        in_dim = self.data_dim
        for out_dim in encoder_widths:
            self.encoder_linears.append(nn.Linear(in_dim, out_dim))
            in_dim = out_dim

        # Output layers for posterior parameters
        self.fc_mu = nn.Linear(in_dim, self.latent_dim)        # Mean of q(z|x)
        self.fc_logvar = nn.Linear(in_dim, self.latent_dim)    # Log-variance of q(z|x), diagonal covariance

        # Decoder network
        self.decoder_linears = nn.ModuleList()
        in_dim = self.latent_dim
        for out_dim in decoder_widths:
            self.decoder_linears.append(nn.Linear(in_dim, out_dim))
            in_dim = out_dim
        self.fc_x_recon = torch.nn.Linear(in_dim, self.data_dim)  # Final reconstruction layer

    def encode(self, x):
        h = x
        for layer in self.encoder_linears:
            h = self.activation(layer(h))
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

    @staticmethod
    def reparameterize(mu, logvar):
        std = torch.exp(0.5 * logvar)             # Standard deviation
        eps = torch.randn_like(std)               # Sample from N(0, I)
        return mu + eps * std

    def decode(self, z):
        h = z
        for layer in self.decoder_linears:
            h = self.activation(layer(h))
        x_recon = self.fc_x_recon(h)
        return x_recon

    def forward(self, x):
        posterior_params = self.encode(x)
        mu, logvar = posterior_params
        z = self.reparameterize(mu, logvar)
        x_recon = self.decode(z)
        return z, x_recon, posterior_params
