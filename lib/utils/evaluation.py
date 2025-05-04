import os
import time
import numpy as np
import torch

from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import PCA
from numpy.linalg import lstsq

from geomstats.geometry.base import ImmersedSet
from geomstats.geometry.euclidean import Euclidean
from geomstats.geometry.pullback_metric import PullbackMetric
from geomstats.geometry.special_orthogonal import SpecialOrthogonal

from ..datasets.synthetic_sphere_like import (
    get_s1_synthetic_immersion,
    get_s2_synthetic_immersion,
    get_t2_synthetic_immersion,
    get_scrunchy_immersion,
    get_interlocking_rings_immersion
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
    def immersion_vm(angle):
        if config.dataset_name == "s1_synthetic":
            z = gs.array([gs.cos(angle[0]), gs.sin(angle[0])])

        elif config.dataset_name == "s2_synthetic":
            theta, phi = angle
            z = gs.array([
                gs.sin(theta) * gs.cos(phi),
                gs.sin(theta) * gs.sin(phi),
                gs.cos(theta),
            ])

        elif config.dataset_name == "t2_synthetic":
            theta, phi = angle
            z = gs.array([
                (config.major_radius - config.minor_radius * gs.cos(theta)) * gs.cos(phi),
                (config.major_radius - config.minor_radius * gs.cos(theta)) * gs.sin(phi),
                config.minor_radius * gs.sin(theta),
            ])
        else:
            raise InvalidConfigError(f"Unknown dataset: {config.dataset_name}")

        z = z.to(config.device)
        return model.decode(z)

    def immersion_euclidean(z):
        z = z.to(config.device)
        return model.decode(z)

    if config.model_type == 'EuclideanVAE':
        return immersion_euclidean
    if config.model_type == 'VonMisesVAE':
        return immersion_vm
    else:
        raise InvalidConfigError(f"Unknown model type: {config.model_type}")


def get_true_immersion(config):
    rot = torch.eye(n=config.embedding_dim)
    if config.rotation == "random":
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
    if config.dataset_name == "scrunchy_synthetic":
        return get_scrunchy_immersion(
            config.radius,
            config.n_wiggles,
            config.geodesic_distortion_amp,
            config.embedding_dim,
            rot,
        )
    if config.dataset_name == "interlocking_rings_synthetic":
        return get_interlocking_rings_immersion(
            config.radius,
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
    if config.dataset_name in {"s1_synthetic", "scrunchy_synthetic"}:
        return torch.linspace(0, 2 * gs.pi, n_grid_points)
    elif config.dataset_name == "interlocking_rings_synthetic":
        return torch.linspace(0, 4 * gs.pi, n_grid_points)
    elif config.dataset_name == "s2_synthetic":
        thetas = gs.linspace(0.01, gs.pi, int(np.sqrt(n_grid_points)))
        phis = gs.linspace(0, 2 * gs.pi, int(np.sqrt(n_grid_points)))
        return torch.cartesian_prod(thetas, phis)
    elif config.dataset_name == "t2_synthetic":
        thetas = gs.linspace(0, 2 * gs.pi, int(np.sqrt(n_grid_points)))
        phis = gs.linspace(0, 2 * gs.pi, int(np.sqrt(n_grid_points)))
        z_grid = torch.cartesian_prod(thetas, phis)
    else:
        raise InvalidConfigError(f"Unknown dataset: {config.dataset_name}")
    return z_grid


def get_latent_vectors(config, model, test_loader, n_grid_points=2000):
    model.eval()
    latent_vectors = []
    labels = []
    device = config.device
    with torch.no_grad():
        for x, label in test_loader:
            x = x.to(device)
            z, _, _ = model.forward(x)
            latent_vectors.append(z.cpu())
            labels.append(label)

    labels = torch.cat(labels)
    latent_vectors = torch.cat(latent_vectors, dim=0)
    return latent_vectors, labels


def _compute_curvature(z_grid, immersion, dim, embedding_dim):
    """Compute mean curvature vector and its norm at each point."""
    # neural_metric = PullbackMetric(
    #     dim=dim, embedding_dim=embedding_dim, immersion=immersion
    # )
    neural_manifold = NeuralManifoldIntrinsic(
        dim, embedding_dim, immersion, equip=False
    )
    neural_manifold.equip_with_metric(PullbackMetric)
    torch.unsqueeze(z_grid[0], dim=0)
    if dim == 1:
        curv = gs.zeros(len(z_grid), embedding_dim)
        for i_z, z in enumerate(z_grid):
            z = torch.unsqueeze(z, dim=0)
            curv[i_z, :] = neural_manifold.metric.mean_curvature_vector(z)
    else:
        curv = torch.full((len(z_grid), embedding_dim), torch.nan)
        for i, z_i in enumerate(z_grid):
            try:
                curv[i, :] = neural_manifold.metric.mean_curvature_vector(z_i)
            except Exception as e:
                print(f"An error occurred for i={i}: {e}")
                print(neural_manifold.metric.metric_matrix(z_i))
    curv_norm = torch.linalg.norm(curv, dim=1, keepdim=True).squeeze()

    return curv, curv_norm


def compute_curvature_learned(model, test_loader, config, n_grid_points=2000):
    if config.model_type == 'EuclideanVAE':
        z_grid, labels = get_latent_vectors(config, model, test_loader, n_grid_points)
    elif config.model_type == 'VonMisesVAE':
        z_grid = get_z_grid(config, n_grid_points)
        labels = None
    else:
        raise InvalidConfigError(f"Unknown model type: {config.model_type}")
    immersion = get_learned_immersion(model, config)
    manifold_dim = z_grid.shape[1] if z_grid.ndim > 1 else 1
    result = _compute_curvature(z_grid, immersion, manifold_dim, config.embedding_dim)
    return z_grid, labels, *result  # *result = curv, curv_norm


def compute_curvature_true(config, n_grid_points=2000):
    z_grid = get_z_grid(config, n_grid_points)
    immersion = get_true_immersion(config)
    if config.dataset_name in {"s1_synthetic", "interlocking_rings_synthetic", "scrunchy_synthetic"}:
        manifold_dim = 1
    elif config.dataset_name == "s2_synthetic" or config.dataset_name == "t2_synthetic":
        manifold_dim = 2
    else:
        raise InvalidConfigError(f"Unknown dataset: {config.dataset_name}")
    start = time.time()
    result = _compute_curvature(z_grid, immersion, manifold_dim, config.embedding_dim)
    print(f"Computation time: {time.time() - start:.3f} s")
    return z_grid, *result


def compute_curvature_true_latents(config, angles):
    immersion = get_true_immersion(config)
    if config.dataset_name in {"s1_synthetic", "interlocking_rings_synthetic", "scrunchy_synthetic"}:
        manifold_dim = 1
    elif config.dataset_name == "s2_synthetic" or config.dataset_name == "t2_synthetic":
        manifold_dim = 2
    else:
        raise InvalidConfigError(f"Unknown dataset: {config.dataset_name}")
    result = _compute_curvature(angles, immersion, manifold_dim, config.embedding_dim)
    return angles, *result


def _compute_curvature_error_s1(thetas, learned, true):
    learned = np.array(learned)
    true = np.array(true)
    diff = np.trapz((learned - true) ** 2, thetas)
    norm = np.trapz(learned ** 2 + true ** 2, thetas)
    return diff / norm


def _integrate_s2(thetas, phis, h):
    thetas = torch.unique(thetas)
    phis = torch.unique(phis)
    sum_phis = torch.zeros_like(thetas)
    for t, theta in enumerate(thetas):
        sum_phis[t] = torch.trapz(h[t * len(phis):(t + 1) * len(phis)], phis) * np.sin(theta)
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


# Empiric curvature estimate
def estimate_curvature_1d_quadric(points, k=20):
    n, d = points.shape
    nbrs = NearestNeighbors(n_neighbors=k).fit(points)
    _, indices = nbrs.kneighbors(points)
    curvatures = []
    for i in range(n):
        neighbors = points[indices[i]]
        centroid = neighbors.mean(axis=0)
        centered = neighbors - centroid
        pca = PCA(n_components=2).fit(centered)
        tangent = pca.components_[0]
        normal = pca.components_[1]
        coords = centered @ tangent
        heights = centered @ normal
        A = np.column_stack([coords ** 2, coords, np.ones_like(coords)])
        coeffs, _, _, _ = lstsq(A, heights, rcond=None)
        curvature = abs(2 * coeffs[0])
        curvatures.append(curvature)
    return np.array(curvatures)


def estimate_curvature_2d_quadric(points, k=30):
    n, d = points.shape
    nbrs = NearestNeighbors(n_neighbors=k).fit(points)
    _, indices = nbrs.kneighbors(points)
    curvatures = []
    for i in range(n):
        neighbors = points[indices[i]]
        centroid = neighbors.mean(axis=0)
        centered = neighbors - centroid
        pca = PCA(n_components=3).fit(centered)
        normal = pca.components_[-1]
        tangent = pca.components_[:2]
        local_coords = centered @ tangent.T
        heights = centered @ normal
        X, Y = local_coords[:, 0], local_coords[:, 1]
        A = np.column_stack([X ** 2, Y ** 2, X * Y, X, Y, np.ones_like(X)])
        coeffs, _, _, _ = lstsq(A, heights, rcond=None)
        a, b, c = coeffs[0], coeffs[1], coeffs[2]
        II = np.array([[2 * a, c], [c, 2 * b]])
        H = 0.5 * np.trace(II)
        curvatures.append(abs(H))
    return np.array(curvatures)


class InvalidConfigError(Exception):
    pass
