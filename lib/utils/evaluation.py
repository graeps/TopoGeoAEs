import os
import time
import numpy as np
import torch

from geomstats.geometry.base import ImmersedSet
from geomstats.geometry.euclidean import Euclidean
from geomstats.geometry.pullback_metric import PullbackMetric
from geomstats.geometry.special_orthogonal import SpecialOrthogonal

from ..datasets.synthetic_sphere_like import (
    get_s1_synthetic_immersion,
    get_s2_synthetic_immersion,
    get_t2_synthetic_immersion,
)

os.environ["GEOMSTATS_BACKEND"] = "pytorch"
import geomstats.backend as gs


class NeuralManifoldIntrinsic(ImmersedSet):
    def __init__(self, dim, neural_embedding_dim, neural_immersion, equip=True):
        self.neural_embedding_dim = neural_embedding_dim
        super().__init__(dim=dim, equip=equip)
        self.neural_immersion = neural_immersion

    def immersion(self, point):
        return self.neural_immersion(point)

    def _define_embedding_space(self):
        return Euclidean(dim=self.neural_embedding_dim)


def get_learned_immersion(model, config):
    def immersion(angle):
        if config.dataset_name == "s1_synthetic":
            z = gs.array([gs.cos(angle[0]), gs.sin(angle[0])])

        elif config.dataset_name == "s2_synthetic":
            theta, phi = angle[0], angle[1]
            z = gs.array([
                gs.sin(theta) * gs.cos(phi),
                gs.sin(theta) * gs.sin(phi),
                gs.cos(theta),
            ])

        elif config.dataset_name == "t2_synthetic":
            theta, phi = angle[0], angle[1]
            z = gs.array([
                (config.major_radius - config.minor_radius * gs.cos(theta)) * gs.cos(phi),
                (config.major_radius - config.minor_radius * gs.cos(theta)) * gs.sin(phi),
                config.minor_radius * gs.sin(theta),
            ])
        else:
            raise InvalidConfigError(f"Unknown dataset: {config.dataset_name}")

        z = z.to(config.device)
        if z.ndim == 1:
            z = z.unsqueeze(0)
        return model.decode(z)

    return immersion


def get_true_immersion(config):
    rot = torch.eye(n=config.embedding_dim)
    if config.synthetic_rotation == "random":
        rot = SpecialOrthogonal(n=config.embedding_dim).random_point()

    if config.dataset_name == "s1_synthetic":
        return get_s1_synthetic_immersion(
            config.geodesic_distortion_func,
            config.radius,
            config.n_wiggles,
            config.geodesic_distortion_amp,
            config.embedding_dim,
            rot,
        )
    elif config.dataset_name == "s2_synthetic":
        return get_s2_synthetic_immersion(
            config.radius,
            config.geodesic_distortion_amp,
            config.embedding_dim,
            rot,
        )
    elif config.dataset_name == "t2_synthetic":
        return get_t2_synthetic_immersion(
            config.major_radius,
            config.minor_radius,
            config.geodesic_distortion_amp,
            config.embedding_dim,
            rot,
        )
    else:
        raise InvalidConfigError(f"Unknown dataset: {config.dataset_name}")


def get_z_grid(config, n_grid_points=2000):
    if config.dataset_name == "s1_synthetic":
        return torch.linspace(0, 2 * gs.pi, n_grid_points)
    elif config.dataset_name == "s2_synthetic":
        thetas = gs.linspace(0.01, gs.pi, int(np.sqrt(n_grid_points)))
        phis = gs.linspace(0, 2 * gs.pi, int(np.sqrt(n_grid_points)))
        return torch.cartesian_prod(thetas, phis)
    else:
        raise InvalidConfigError(f"Unknown dataset: {config.dataset_name}")


def _compute_curvature(z_grid, immersion, dim, embedding_dim):
    manifold = NeuralManifoldIntrinsic(dim, embedding_dim, immersion, equip=False)
    manifold.equip_with_metric(PullbackMetric)

    if dim == 1:
        curv = gs.zeros(len(z_grid), embedding_dim)
        for i_z, z in enumerate(z_grid):
            curv[i_z, :] = manifold.metric.mean_curvature_vector(torch.unsqueeze(z, dim=0))
        geodesic_dist = gs.zeros(len(z_grid))
    else:
        curv = torch.full((len(z_grid), embedding_dim), torch.nan)
        for i, z_i in enumerate(z_grid):
            try:
                curv[i, :] = manifold.metric.mean_curvature_vector(z_i)
            except Exception as e:
                print(f"Error at i={i}: {e}")
        geodesic_dist = gs.zeros(len(z_grid))

    curv_norm = torch.linalg.norm(curv, dim=1)
    return geodesic_dist, curv, curv_norm


def compute_curvature_learned(model, config, embedding_dim, n_grid_points=2000):
    z_grid = get_z_grid(config, n_grid_points)
    immersion = get_learned_immersion(model, config)
    start = time.time()
    result = _compute_curvature(z_grid, immersion, config.manifold_dim, embedding_dim)
    print(f"Computation time: {time.time() - start:.3f} s")
    return z_grid, *result


def compute_curvature_true(config, n_grid_points=2000):
    z_grid = get_z_grid(config, n_grid_points)
    immersion = get_true_immersion(config)
    start = time.time()
    result = _compute_curvature(z_grid, immersion, config.manifold_dim, config.embedding_dim)
    print(f"Computation time: {time.time() - start:.3f} s")
    return z_grid, *result


def _compute_curvature_error_s1(thetas, learned, true):
    learned = np.array(learned)
    true = np.array(true)
    diff = np.trapezoid((learned - true) ** 2, thetas)
    norm = np.trapezoid(learned ** 2 + true ** 2, thetas)
    return diff / norm


def _integrate_s2(thetas, phis, h):
    thetas = torch.unique(thetas)
    phis = torch.unique(phis)
    sum_phis = torch.zeros_like(thetas)
    for t, theta in enumerate(thetas):
        sum_phis[t] = torch.trapz(h[t*len(phis):(t+1)*len(phis)], phis) * np.sin(theta)
    return torch.trapz(sum_phis, thetas)


def _compute_curvature_error_s2(thetas, phis, learned, true):
    diff = _integrate_s2(thetas, phis, (learned - true) ** 2)
    norm = _integrate_s2(thetas, phis, learned ** 2 + true ** 2)
    return diff / norm


def compute_curvature_error(z_grid, learned, true, config):
    if config.dataset_name == "s1_synthetic":
        return _compute_curvature_error_s1(z_grid, learned, true)
    elif config.dataset_name in ("s2_synthetic", "t2_synthetic"):
        return _compute_curvature_error_s2(z_grid[:, 0], z_grid[:, 1], learned, true)
    else:
        raise InvalidConfigError(f"Unknown dataset: {config.dataset_name}")


class InvalidConfigError(Exception):
    pass
