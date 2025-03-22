import torch
from torch.nn import functional as F
import torch.nn as nn

from code.mvae.lib.models.utils.valid_config import is_valid_model_config
from code.mvae.lib.distributions import MGVonMises


class MGVMToroidalVAE(torch.nn.Module):
    def __init__(
            self,
            config
    ):
        is_valid_model_config(config)
        super().__init__()
        self.posterior_type = "mgvm_toroidal"
        self.data_dim = config["data_dim"]
        self.sftbeta = config["sftbeta"]
        self.latent_dim = config["latent_dim"]  # Here latent_dim = d for T^d latent space (manifold dim)
        self.encoder_width = config["encoder_width"]
        self.encoder_depth = config["encoder_depth"]
        self.decoder_width = config["decoder_width"]
        self.decoder_depth = config["decoder_depth"]

        self.encoder_flatten = nn.Flatten()
        self.encoder_fc = nn.Linear(self.data_dim, self.encoder_width)
        self.encoder_linears = nn.ModuleList(
            [
                nn.Linear(self.encoder_width, self.encoder_width)
                for _ in range(self.encoder_depth)
            ]
        )

        self.fc_z_mu = nn.Linear(self.encoder_width, self.latent_dim)
        self.fc_z_kappa = nn.Linear(self.encoder_width, self.latent_dim)
        self.fc_z_l = nn.Linear(self.encoder_width, ((2 * self.latent_dim) * (2 * self.latent_dim + 1) / 2))

        self.decoder_fc = nn.Linear(self.latent_dim, self.decoder_width)
        self.decoder_linears = nn.ModuleList(
            [
                nn.Linear(self.decoder_width, self.decoder_width)
                for _ in range(self.decoder_depth)
            ]
        )

        self.fc_x_recon = nn.Linear(self.decoder_width, self.data_dim)

    def encode(self, x):
        h = self.encoder_flatten(x)
        h = F.softplus(self.encoder_fc(h), beta=self.sftbeta)

        for layer in self.encoder_linears:
            h = F.softplus(layer(h), beta=self.sftbeta)

        mu = self.fc_z_mu(h)
        kappa = F.softplus(self.fc_z_kappa(h), beta=self.sftbeta)

        l_elements = self.fc_z_l(h)
        __l = torch.zeros(x.shape[0], self.n, self.n, device=x.device)
        indices = torch.tril_indices(self.n, self.n)
        __l[:, indices[0], indices[1]] = l_elements  # l lower triangular matrix
        __l[:, range(self.n), range(self.n)] = torch.exp(
            __l[:, range(self.n), range(self.n)])  # positive elements on diagonal
        w = __l @ __l.transpose(-2, -1)  # l*l^T Cholesky factorization for w SPD matrix

        posterior_params = mu, kappa, w
        return posterior_params

    def reparameterize(self, posterior_params):
        mu, kappa, w = posterior_params
        print("mu.shape", mu.shape)
        print("kappa.shape", kappa.shape)
        q_z = MGVonMises(mu, kappa, w)  # mGvM Distribution on [0,2pi]^latent_dim
        z = q_z.rsample()
        print("sample z.shape", z.shape)

        return z

    def decode(self, z):
        h = F.softplus(self.decoder_fc(z), beta=self.sftbeta)

        for layer in self.decoder_linears:
            h = F.softplus(layer(h), beta=self.sftbeta)

        x_recon = F.sigmoid(self.fc_x_recon(h))
        return x_recon.view(-1, 1, 28, 28)

    def forward(self, x):
        posterior_params = self.encode(x)
        z = self.reparameterize(posterior_params)
        x_recon = self.decode(z)

        return z, x_recon, posterior_params


