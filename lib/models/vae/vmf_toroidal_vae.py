import torch
from torch.nn import functional as F
import torch.nn as nn

from ...distributions import VonMisesFisher


class VMFToroidalVAE(nn.Module):
    """
    Variational Autoencoder (VAE) with a toroidal latent space S^n × S^m ⊂ ℝ^{n+1} × ℝ^{m+1}.

    The encoder outputs the parameters of two independent von Mises–Fisher (vMF) distributions
    over S^n and S^m. Samples from these distributions are projected to ℝ³ using a standard
    torus embedding when n = m = 1.

    Adapted from:
    Acosta, Francisco, et al. "Quantifying extrinsic curvature in neural manifolds."
    Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 2023.
    https://arxiv.org/abs/2212.10414
    https://github.com/geometric-intelligence/neurometry
    """

    def __init__(self, config):
        """
        Initialize the VMFToroidalVAE.

        Args:
            config: Configuration object with the following attributes:
                * embedding_dim (int): Dimensionality of the input data.
                * latent_dim (int): Dimension of each S^k component of the latent torus.
                * sftbeta (float): Beta parameter for the Softplus activation.
                * activation (str): Activation function ("relu" or "softplus").
                * encoder_widths (List[int]): Hidden layer widths for the encoder.
                * decoder_widths (List[int]): Hidden layer widths for the decoder.

        Raises:
            NotImplementedError: If an unsupported activation function is specified.
        """
        super().__init__()
        self.posterior_type = "vmf_toroidal_vae"
        self.data_dim = config.embedding_dim
        self.latent_dim = config.latent_dim
        self.sftbeta = config.sftbeta

        # Activation
        if config.activation == "relu":
            self.activation = F.relu
        elif config.activation == "softplus":
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

        self.fc_z_theta_mu = nn.Linear(in_dim, self.latent_dim)
        self.fc_z_theta_kappa = nn.Linear(in_dim, 1)

        self.fc_z_phi_mu = nn.Linear(in_dim, self.latent_dim)
        self.fc_z_phi_kappa = nn.Linear(in_dim, 1)

        # Decoder network
        self.decoder_linears = nn.ModuleList()
        in_dim = 3  # Torus embedding in ℝ³
        for out_dim in decoder_widths:
            self.decoder_linears.append(nn.Linear(in_dim, out_dim))
            in_dim = out_dim
        self.fc_x_recon = nn.Linear(in_dim, self.data_dim)

    def encode(self, x):
        """
        Encode input into parameters of the toroidal posterior.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, data_dim).

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
                * z_theta_mu: Mean directions for the S^n component, shape (batch_size, latent_dim).
                * z_theta_kappa: Concentrations for S^n, shape (batch_size, 1).
                * z_phi_mu: Mean directions for the S^m component, shape (batch_size, latent_dim).
                * z_phi_kappa: Concentrations for S^m, shape (batch_size, 1).
        """
        h = x
        for layer in self.encoder_linears:
            h = self.activation(layer(h))

        z_theta_mu = self.fc_z_theta_mu(h)
        z_theta_kappa = self.activation(self.fc_z_theta_kappa(h)) + 1.0

        z_phi_mu = self.fc_z_phi_mu(h)
        z_phi_kappa = self.activation(self.fc_z_phi_kappa(h)) + 1.0

        return z_theta_mu, z_theta_kappa, z_phi_mu, z_phi_kappa

    def _build_torus(self, z_theta, z_phi):
        """
        Embed two S^1 samples into a standard torus in ℝ³.

        Args:
            z_theta (torch.Tensor): Samples on S^1, shape (batch_size, 2).
            z_phi (torch.Tensor): Samples on S^1, shape (batch_size, 2).

        Returns:
            torch.Tensor: Embedded torus coordinates, shape (batch_size, 3).
        """
        cos_theta, sin_theta = z_theta[:, 0], z_theta[:, 1]
        cos_phi, sin_phi = z_phi[:, 0], z_phi[:, 1]

        major_radius = 2.0
        minor_radius = 1.0

        x = (major_radius - minor_radius * cos_theta) * cos_phi
        y = (major_radius - minor_radius * cos_theta) * sin_phi
        z = minor_radius * sin_theta

        return torch.stack([x, y, z], dim=-1)

    def reparameterize(self, posterior_params):
        """
        Sample from the vMF posteriors and embed into a torus.

        Args:
            posterior_params (tuple): Parameters returned by ``encode``.

        Returns:
            torch.Tensor: Latent samples embedded in ℝ³, shape (batch_size, 3).
        """
        z_theta_mu, z_theta_kappa, z_phi_mu, z_phi_kappa = posterior_params
        q_z_theta = VonMisesFisher(z_theta_mu, z_theta_kappa)
        q_z_phi = VonMisesFisher(z_phi_mu, z_phi_kappa)

        z_theta = q_z_theta.rsample()
        z_phi = q_z_phi.rsample()
        return self._build_torus(z_theta, z_phi)

    def decode(self, z):
        """
        Decode torus-embedded latent samples back to input space.

        Args:
            z (torch.Tensor): Latent tensor of shape (batch_size, 3).

        Returns:
            torch.Tensor: Reconstructed input of shape (batch_size, data_dim).
        """
        h = z
        for layer in self.decoder_linears:
            h = self.activation(layer(h))
        return self.fc_x_recon(h)

    def forward(self, x):
        """
        Perform a full forward pass of the VAE.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, data_dim).

        Returns:
            Tuple[torch.Tensor, torch.Tensor, Tuple[torch.Tensor, torch.Tensor,
                torch.Tensor, torch.Tensor]]:
                * z: Latent samples embedded in ℝ³.
                * x_recon: Reconstructed inputs.
                * posterior_params: Tuple of posterior parameters
                  (z_theta_mu, z_theta_kappa, z_phi_mu, z_phi_kappa).
        """
        posterior_params = self.encode(x)
        z = self.reparameterize(posterior_params)
        x_recon = self.decode(z)
        return z, x_recon, posterior_params
