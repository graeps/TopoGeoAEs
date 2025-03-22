import torch.nn as nn


class EuclideanAE(nn.Module):
    def __init__(self, data_dim, hidden_dim1=10, hidden_dim2=8, latent_dim=2):
        super().__init__()
        self.type = "euclidean_ae"
        self.latent_dim = latent_dim

        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(data_dim, hidden_dim1),
            nn.ReLU(),
            nn.Linear(hidden_dim1, hidden_dim2),
            nn.ReLU(),
            nn.Linear(hidden_dim2, self.latent_dim),
        )

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(self.latent_dim, hidden_dim2),
            nn.ReLU(),
            nn.Linear(hidden_dim2, hidden_dim1),
            nn.ReLU(),
            nn.Linear(hidden_dim1, data_dim),
        )

    def forward(self, x):
        z = self.encoder(x)
        x_recon = self.decoder(z)

        return z, x_recon
