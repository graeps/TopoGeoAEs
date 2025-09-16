import torch.nn as nn
from torch.nn import functional as F

from ...distributions import VonMisesFisher


class VMFSphericalVAE(nn.Module):
    """
    Variational Autoencoder (VAE) with a hyperspherical latent space S^{d−1} ⊂ ℝ^d.

    The latent space is modeled using a von Mises–Fisher (vMF) distribution,
    following T. Davidson et al. (2018), "Hyperspherical Variational Auto-Encoders".
    The encoder outputs the mean direction and concentration parameters of the vMF
    posterior, and sampling is performed using the reparameterization trick.

    The decoder reconstructs the input from samples drawn on the hypersphere.

    Adapted from:

    Davidson, T. R., Falorsi, L., De Cao, N., Kipf, T.,
    and Tomczak, J. M. (2018). Hyperspherical Variational
    Auto-Encoders. 34th Conference on Uncertainty in Artificial Intelligence (UAI-18).
    http://arxiv.org/abs/1804.00891
    https://github.com/nicola-decao/s-vae-pytorch
    """

    def __init__(self, config):
        """
        Initialize the VMFSphericalVAE.

        Args:
            config: Configuration object with the following required attributes:
                * embedding_dim (int): Dimensionality of the input data.
                * latent_dim (int): Dimensionality of the hyperspherical latent space
                  (latent variable z lies on S^{latent_dim−1} ⊂ ℝ^{latent_dim}).
                * activation (str): Activation function ("relu" or "softplus").
                * sftbeta (float): Beta parameter for the Softplus activation
                  (used only if activation is "softplus").
                * encoder_widths (List[int]): Sizes of hidden layers in the encoder.
                * decoder_widths (List[int]): Sizes of hidden layers in the decoder.

        Raises:
            NotImplementedError: If an unsupported activation function is specified.
        """
        super().__init__()
        self.type = "vmf_spherical_vae"
        self.data_dim = config.embedding_dim
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

        # Posterior parameter layers
        self.fc_z_mu = nn.Linear(in_dim, self.latent_dim)  # Mean direction in ℝ^latent_dim
        self.fc_z_kappa = nn.Linear(in_dim, 1)             # Concentration scalar

        # Decoder network
        self.decoder_linears = nn.ModuleList()
        in_dim = self.latent_dim
        for out_dim in decoder_widths:
            self.decoder_linears.append(nn.Linear(in_dim, out_dim))
            in_dim = out_dim
        self.fc_x_recon = nn.Linear(in_dim, self.data_dim)

    def encode(self, x):
        """
        Encode input data into the parameters of the vMF posterior.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, data_dim).

        Returns:
            Tuple[torch.Tensor, torch.Tensor]:
                * z_mu: Mean direction on the hypersphere, shape (batch_size, latent_dim),
                  normalized to unit norm.
                * z_kappa: Concentration parameter κ > 1, shape (batch_size, 1).
        """
        h = x
        for layer in self.encoder_linears:
            h = self.activation(layer(h))
        z_mu = self.fc_z_mu(h)
        z_mu = z_mu / z_mu.norm(dim=-1, keepdim=True)  # Normalize to S^{d−1}
        z_kappa = F.softplus(self.fc_z_kappa(h)) + 1.0  # Ensure positive concentration
        return z_mu, z_kappa

    def reparameterize(self, posterior_params):
        """
        Sample latent vectors using the reparameterization trick for vMF distributions.

        Args:
            posterior_params (Tuple[torch.Tensor, torch.Tensor]):
                Mean direction and concentration (z_mu, z_kappa) of shapes
                (batch_size, latent_dim) and (batch_size, 1).

        Returns:
            torch.Tensor: Sampled latent vectors on S^{latent_dim−1}, shape
            (batch_size, latent_dim).
        """
        z_mu, z_kappa = posterior_params
        q_z = VonMisesFisher(z_mu, z_kappa)
        z = q_z.rsample()
        return z

    def decode(self, z):
        """
        Decode latent samples from the hypersphere back to the input space.

        Args:
            z (torch.Tensor): Latent tensor on S^{latent_dim−1}, shape
                (batch_size, latent_dim).

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
        Perform a full forward pass through the VMFSphericalVAE.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, data_dim).

        Returns:
            Tuple[torch.Tensor, torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
                * z: Sampled latent vectors on S^{latent_dim−1}, shape
                  (batch_size, latent_dim).
                * x_recon: Reconstructed input, shape (batch_size, data_dim).
                * posterior_params: Tuple (z_mu, z_kappa) of posterior parameters.
        """
        posterior_params = self.encode(x)
        z = self.reparameterize(posterior_params)
        x_recon = self.decode(z)
        return z, x_recon, posterior_params
