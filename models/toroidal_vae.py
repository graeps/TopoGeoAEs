import torch
from torch.nn import functional as f

from ..utils import is_valid_model_config
from ..distributions.von_mises_fisher import VonMisesFisher


class ToroidalVAE(torch.nn.Module):
    """VAE with Linear (fully connected) layers, three-dimensional toroidal latent space, uniform prior distribution
    and von Mises-Fisher likelihood.

    Parameters
    ----------
    data_dim : int
        Dimension of input data.
        Example: 40 for neural recordings of 40 units/clusters.
    latent_dim : int
        Dimension of the latent space.
        Example: 2.
    """

    def __init__(
            self,
            config
    ):
        is_valid_model_config(config)
        super().__init__()
        self.posterior_type = "toroidal"
        self.data_dim = config["data_dim"]
        self.sftbeta = config["sftbeta"]
        self.latent_dim = config["latent_dim"]
        self.encoder_width = config["encoder_width"]
        self.encoder_depth = config["encoder_depth"]
        self.decoder_width = config["decoder_width"]
        self.decoder_depth = config["decoder_depth"]

        self.encoder_fc = torch.nn.Linear(self.data_dim, self.encoder_width)
        self.encoder_linears = torch.nn.ModuleList(
            [
                torch.nn.Linear(self.encoder_width, self.encoder_width)
                for _ in range(self.encoder_depth)
            ]
        )

        self.fc_z_theta_mu = torch.nn.Linear(self.encoder_width, self.latent_dim)
        self.fc_z_theta_kappa = torch.nn.Linear(self.encoder_width, 1)

        self.fc_z_phi_mu = torch.nn.Linear(self.encoder_width, self.latent_dim)
        self.fc_z_phi_kappa = torch.nn.Linear(self.encoder_width, 1)

        self.decoder_fc = torch.nn.Linear(3, self.decoder_width)
        self.decoder_linears = torch.nn.ModuleList(
            [
                torch.nn.Linear(self.decoder_width, self.decoder_width)
                for _ in range(self.decoder_depth)
            ]
        )

        self.fc_x_recon = torch.nn.Linear(self.decoder_width, self.data_dim)

    def encode(self, x):
        """Encode input into mean and log-variance.

        The parameters mean (mu) and variance (computed
        from logvar) defines a multivariate Gaussian
        that represents the approximate posterior of the
        latent variable z given the input x.

        Parameters
        ----------
        x : array-like, shape=[batch_size, data_dim]
            Input data.

        Returns
        -------
        mu : array-like, shape=[batch_size, latent_dim]
            Mean of multivariate Gaussian in latent space.
        logvar : array-like, shape=[batch_size, latent_dim]
            Vector representing the diagonal covariance of the
            multivariate Gaussian in latent space.
        """
        h = f.softplus(self.encoder_fc(x), beta=self.sftbeta)

        for layer in self.encoder_linears:
            h = f.softplus(layer(h), beta=self.sftbeta)

        z_theta_mu = self.fc_z_theta_mu(h)
        z_theta_kappa = f.softplus(self.fc_z_theta_kappa(h)) + 1

        z_phi_mu = self.fc_z_phi_mu(h)
        z_phi_kappa = f.softplus(self.fc_z_phi_kappa(h)) + 1

        return z_theta_mu, z_theta_kappa, z_phi_mu, z_phi_kappa

    def _build_torus(self, z_theta, z_phi):
        # theta = torch.atan2(z_theta[:, 1] / z_theta[:, 0])
        # phi = torch.atan2(z_phi[:, 1] / z_phi[:, 0])
        cos_theta = z_theta[:, 0]
        sin_theta = z_theta[:, 1]

        cos_phi = z_phi[:, 0]
        sin_phi = z_phi[:, 1]

        major_radius = 2
        minor_radius = 1

        x = (major_radius - minor_radius * cos_theta) * cos_phi
        y = (major_radius - minor_radius * cos_theta) * sin_phi
        z = minor_radius * sin_theta

        return torch.stack([x, y, z], dim=-1)

    def reparameterize(self, posterior_params):
        """
        Apply reparameterization trick. We 'eternalize' the
        randomness in z by re-parameterizing the variable as
        a deterministic and differentiable function of x,
        the encoder weights, and a new random variable eps.

        Parameters
        ----------
        posterior_params : tuple
            Distributional parameters of approximate posterior. ((e.g.), (z_mu,z_logvar) for Gaussian.

        Returns
        -------

        z: array-like, shape = [batch_size, latent_dim]
            Re-parameterized latent variable.
        """

        z_theta_mu, z_theta_kappa, z_phi_mu, z_phi_kappa = posterior_params

        q_z_theta = VonMisesFisher(z_theta_mu, z_theta_kappa)

        q_z_phi = VonMisesFisher(z_phi_mu, z_phi_kappa)

        z_theta = q_z_theta.rsample()

        z_phi = q_z_phi.rsample()

        return self._build_torus(z_theta, z_phi)

    def decode(self, z):
        """Decode latent variable z into data.

        Parameters
        ----------
        z : array-like, shape=[batch_size, latent_dim]
            Input to the decoder.

        Returns
        -------
        _ : array-like, shape=[batch_size, data_dim]
            Reconstructed data corresponding to z.
        """

        h = f.softplus(self.decoder_fc(z), beta=self.sftbeta)

        for layer in self.decoder_linears:
            h = f.softplus(layer(h), beta=self.sftbeta)

        return self.fc_x_recon(h)

    def forward(self, x):
        """Run VAE: Encode, sample and decode.

        Parameters
        ----------
        x : array-like, shape=[batch_size, data_dim]
            Input data.

        Returns
        -------
        _ : array-like, shape=[batch_size, data_dim]
            Reconstructed data corresponding to z.
        mu : array-like, shape=[batch_size, latent_dim]
            Mean of multivariate Gaussian in latent space.
        logvar : array-like, shape=[batch_size, latent_dim]
            Vector representing the diagonal covariance of the
            multivariate Gaussian in latent space.
        """

        posterior_params = self.encode(x)

        z = self.reparameterize(posterior_params)

        x_recon = self.decode(z)

        return x_recon, posterior_params
