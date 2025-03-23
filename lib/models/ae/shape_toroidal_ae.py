import torch
import torch.nn as nn


def get_zero_grad_hook(mask, device="cpu"):
    def hook(grad):
        return grad * mask.to(device)

    return hook


def project_to_torus(theta: torch.Tensor):  # map theta in [0,1]^d to T^d=S^1x...xS^1
    cos_theta = torch.cos(2 * torch.pi * theta)
    sin_theta = torch.sin(2 * torch.pi * theta)

    return torch.cat([cos_theta, sin_theta], dim=-1)  # ∈ ℝ^{2d}


class ShapeMatrix(nn.Module):
    def __init__(self, latent_dim=2, device="cpu"):
        super(ShapeMatrix, self).__init__()

        self.mask_triu = torch.triu(torch.ones(latent_dim, latent_dim)).bool()
        self.mask_tril = torch.tril(torch.ones(latent_dim, latent_dim)).bool().fill_diagonal_(False)

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
    def __init__(self, data_dim, hidden_dim1, hidden_dim2, latent_dim=2, activation=nn.ReLU):
        super().__init__()
        self.type = "shape_toroidal_ae"
        self.latent_dim = latent_dim

        # Encoder with LU transformation
        self.encoder = nn.Sequential(
            nn.Linear(data_dim, hidden_dim1),
            activation(),
            nn.Linear(hidden_dim1, hidden_dim1),
            activation(),
            nn.Linear(hidden_dim1, hidden_dim1),
            activation(),
            nn.Linear(hidden_dim1, hidden_dim1),
            activation(),
            nn.Linear(hidden_dim1, hidden_dim2),
            activation(),
            nn.Linear(hidden_dim2, hidden_dim2),
            activation(),
            nn.Linear(hidden_dim2, hidden_dim2),
            activation(),
            nn.Linear(hidden_dim2, latent_dim),
            # nn.Sigmoid()
        )
        # self.shape_matrix = ShapeMatrix(latent_dim)  # LU decomposition layer

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(2*latent_dim, hidden_dim2),
            activation(),
            nn.Linear(hidden_dim2, data_dim),
        )

    def forward(self, x):
        theta = self.encoder(x)
        A = 1
        A_inv_T_theta = 1

        # A_inv_T_theta = self.shape_matrix(theta)  # Apply A = LU
        # z = project_to_torus(A_inv_T_theta)

        # Compute (1/2π)(A^{-1})^T torus_coords
        # Get A matrix
        # L = torch.tril(self.shape_matrix.lower.weight, diagonal=-1) + torch.eye(self.latent_dim, device=x.device)
        # U = torch.triu(self.shape_matrix.upper.weight)
        # A_inv_T_theta = L @ U
        # A = torch.linalg.inv(A_inv_T_theta).T  # (A^{-1})^T

        # # Apply (1/2π)(A^{-1})^T to each sin/cos separately and then flatten
        # z = z.view(-1, 2, self.latent_dim)  # shape [batch, 2, d]
        # z = (1 / (2 * torch.pi)) * torch.einsum('bij,jk->bik', z, A)
        # z = z.flatten(start_dim=1)  # shape [batch, 2d]

        z = project_to_torus(theta)

        x_recon = self.decoder(z)

        return z, x_recon, (A, A_inv_T_theta), theta
