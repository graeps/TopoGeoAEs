import torch
from torch.nn import functional as F

from .utils.valid_config import is_valid_model_config
from ..distributions import VonMisesFisher


class ToroidalVAE(torch.nn.Module):
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

        self.encoder_flatten = torch.nn.Flatten()
        self.encoder_fc = torch.nn.Linear(self.data_dim, self.encoder_width)
        self.encoder_linears = torch.nn.ModuleList(
            [
                torch.nn.Linear(self.encoder_width, self.encoder_width)
                for _ in range(self.encoder_depth)
            ]
        )

        self.fc_z_mu = torch.nn.ModuleList([
            torch.nn.Linear(self.encoder_width, 2)
            for _ in range(self.latent_dim)
        ])

        self.fc_z_kappa = torch.nn.ModuleList([
            torch.nn.Linear(self.encoder_width, 1)
            for _ in range(self.latent_dim)
        ])

        self.decoder_fc = torch.nn.Linear(self.latent_dim * 2, self.decoder_width)
        self.decoder_linears = torch.nn.ModuleList(
            [
                torch.nn.Linear(self.decoder_width, self.decoder_width)
                for _ in range(self.decoder_depth)
            ]
        )

        self.fc_x_recon = torch.nn.Linear(self.decoder_width, self.data_dim)

    def encode(self, x):
        h = self.encoder_flatten(x)
        h = F.softplus(self.encoder_fc(h), beta=self.sftbeta)

        for layer in self.encoder_linears:
            h = F.softplus(layer(h), beta=self.sftbeta)

        posterior_params_list = []
        #print("HHHHHHHHHHHHH",h)
        for mu_layer, kappa_layer in zip(self.fc_z_mu, self.fc_z_kappa):
            #print("layer", mu_layer.weight, kappa_layer.weight)
            mu = mu_layer(h)  # Shape: [batch_size, 2]
            kappa = F.softplus(kappa_layer(h), beta=self.sftbeta) + 1  # Shape: [batch_size, 1]

            one_dim_params = torch.cat((mu, kappa), dim=-1)  # Shape: [batch_size, 2, 2]
            posterior_params_list.append(one_dim_params)
            # print("1.", mu)
            # print("2.", kappa)
            # print("3.", one_dim_params)

        # Stack across latent dimensions
        posterior_params = torch.stack(posterior_params_list, dim=1)  # Shape: [batch_size, latent_dim, 2]
        # print("4.", posterior_params)
        return posterior_params

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
        z_samples = []  # Store samples for each latent dimension

        for i in range(self.latent_dim):
            # print("posterior_params size", posterior_params.shape)
            mu = posterior_params[:, i, :2]  # Shape: [batch_size, 2]
            # print("mu in reparametrization", mu)
            kappa = posterior_params[:, i, 2].unsqueeze(-1)  # Shape: [batch_size, 1]
            # print("kappa in reparametrization", kappa)
            q_z = VonMisesFisher(mu, kappa)  # Create vMF distribution
            theta = q_z.sample()  # Shape: [batch_size, 2]
            # print("sample in reparametrization", theta)
            z_samples.append(theta)  # Append to list

        z = torch.cat(z_samples, dim=-1)  # Shape: [batch_size, latent_dim * 2]
        #  print("z in reparametrization", z)
        return z

    def decode(self, z):
        h = F.softplus(self.decoder_fc(z), beta=self.sftbeta)

        for layer in self.decoder_linears:
            h = F.softplus(layer(h), beta=self.sftbeta)

        return self.fc_x_recon(h).view(-1, 1, 28, 28)

    def forward(self, x):
        posterior_params = self.encode(x)
        z = self.reparameterize(posterior_params)
        x_recon = self.decode(z)

        return z, x_recon, posterior_params
