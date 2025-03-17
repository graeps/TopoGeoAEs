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
    def __init__(self, data_dim=2, latent_dim=2):
        super().__init__()
        self.latent_dim = latent_dim

        # Encoder with LU transformation
        self.encoder = nn.Sequential(
            # nn.Flatten(start_dim=1),
            nn.Linear(data_dim, 5),
            nn.ReLU(),
            nn.Linear(5, 4),
            nn.ReLU(),
            nn.Linear(4, latent_dim),
            nn.Sigmoid(),
        )
        self.shape_matrix = ShapeMatrix(latent_dim)  # LU decomposition layer

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(2 * latent_dim, 5),
            nn.ReLU(),
            nn.Linear(5, 5),
            nn.ReLU(),
            nn.Linear(5, data_dim),
            # nn.Unflatten(-1, (1, 28, 28))
        )

    def forward(self, x):
        theta = self.encoder(x)  # theta ∈ [0,1]^d
        z = project_to_torus(theta)
        x_recon = self.decoder(z)
        dummy = 1

        """
        A_inv_T_theta = self.shape_matrix(theta)  # Apply A = LU
        z = project_to_torus(A_inv_T_theta)

        # Compute (1/2π)(A^{-1})^T torus_coords
        # Get A matrix
        L = torch.tril(self.shape_matrix.lower.weight, diagonal=-1) + torch.eye(self.latent_dim, device=x.device)
        U = torch.triu(self.shape_matrix.upper.weight)
        A_inv_T_theta = L @ U

        A = torch.inverse(A_inv_T_theta).T  # (A^{-1})^T
        z = z.view(-1, 2, self.latent_dim)  # shape [batch, 2, d]

        # Apply (1/2π)(A^{-1})^T to each sin/cos separately and then flatten
        scaled_coords = (1 / (2 * torch.pi)) * torch.einsum('bij,jk->bik', z, A)
        scaled_coords = scaled_coords.flatten(start_dim=1)  # shape [batch, 2d]

        x_recon = self.decoder(scaled_coords)
        
        return theta, x_recon, A, A_inv_T_theta        
        """
        return theta, x_recon, dummy, dummy
