import torch
import torch.nn as nn


def get_zero_grad_hook(mask, device="cpu"):
    def hook(grad):
        return grad * mask.to(device)

    return hook


class ShapeMatrix(nn.Module):
    def __init__(self, latent_dim=2, device="cpu"):
        super(ShapeMatrix, self).__init__()

        self.mask_triu = torch.triu(torch.ones(latent_dim, latent_dim)).bool()
        self.mask_tril = torch.tril(torch.ones(latent_dim, latent_dim)).bool().fill_diagonal_(False)
        self.nonlinearity = nn.ReLU()
        self.layer_size = latent_dim

        self.upper = nn.Linear(latent_dim, latent_dim, bias=False)
        self.lower = nn.Linear(latent_dim, latent_dim, bias=False)

        with torch.no_grad():
            self.upper.weight.data.copy_(torch.triu(self.upper.weight.data))
            self.lower.weight.data.copy_(torch.tril(self.lower.weight.data))
            self.lower.weight.data.fill_diagonal_(1)

        self.upper.weight.register_hook(get_zero_grad_hook(self.mask_triu, device))
        self.lower.weight.register_hook(get_zero_grad_hook(self.mask_tril, device))

    def forward(self, x):
        x = torch.flatten(x, 1)
        x = self.lower(x)
        x = self.upper(x)
        return x


class ShapeToroidalAE(nn.Module):
    def __init__(self, latent_dim=2):
        super().__init__()

        # Encoder with LU transformation
        self.encoder = nn.Sequential(
            nn.Flatten(start_dim=1),
            nn.Linear(784, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, latent_dim),
            nn.Sigmoid(),  # Normalize to [0,1]
        )
        self.lu_transform = ShapeMatrix(latent_dim)  # LU decomposition layer

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 784),
            nn.Unflatten(-1, (1, 28, 28))
        )

    def forward(self, x):
        x = self.encoder(x)
        z = self.lu_transform(x)  # Apply LU transformation
        x_recon = self.decoder(z)
        return z, x_recon  # Decode back
