import os

from .errors import InvalidConfigError

os.environ["GEOMSTATS_BACKEND"] = "pytorch"
import torch
import geomstats.backend as gs
import numpy as np


def get_z_grid(config, n_grid_points):
    eps = 1e-4
    if config.dataset_name in {"s1_low", "scrunchy", "s1_high"}:
        z_grid = torch.linspace(eps, 2 * gs.pi - eps, n_grid_points)
    elif config.dataset_name in {"s2_low", "s2_high"}:
        thetas = gs.arccos(np.linspace(0.99, -0.99, int(np.sqrt(n_grid_points))))
        phis = gs.linspace(0.01, 2 * gs.pi - 0.01, int(np.sqrt(n_grid_points)))
        z_grid = torch.cartesian_prod(thetas, phis)
    elif config.dataset_name in {"t2_low", "t2_high"}:
        thetas = gs.linspace(eps, 2 * gs.pi - eps, int(np.sqrt(n_grid_points)))
        phis = gs.linspace(eps, 2 * gs.pi - eps, int(np.sqrt(n_grid_points)))
        z_grid = torch.cartesian_prod(thetas, phis)
    else:
        raise InvalidConfigError(f"Unknown dataset: {config.dataset_name}")
    return z_grid


def shift_z_grid(z_grid, anchore, config):
    def circle_inverse(x, y):
        theta = torch.atan2(y, x) % (2 * torch.pi)
        return theta

    # 1D case: Circle (S¹)
    if z_grid.ndim == 1:
        anchore1, anchore2 = anchore
        x1, y1 = anchore1
        x2, y2 = anchore2
        angle1 = circle_inverse(x1, y1)
        angle2 = circle_inverse(x2, y2)

        # Compute circular distances from angle1 to all z_grid points
        dists = torch.remainder(z_grid - angle1 + torch.pi, 2 * torch.pi) - torch.pi
        idx_min = torch.argmin(torch.abs(dists))
        z_grid_shifted = torch.roll(z_grid, -idx_min.item(), dims=0)

        # After shifting, align angle2 and decide orientation
        shifted_angles = torch.remainder(z_grid_shifted - z_grid_shifted[0], 2 * torch.pi)
        angle2_shifted = torch.remainder(angle2 - angle1, 2 * torch.pi)
        idx2 = torch.argmin(torch.abs(shifted_angles - angle2_shifted))

        # If angle2 appears before angle1 in the grid, flip
        if idx2 < torch.numel(z_grid) // 2:
            z_grid_shifted = torch.flip(z_grid_shifted, dims=[0])

        return z_grid_shifted

    return z_grid  # fallback
