import os
import numpy as np
import torch
from tqdm import tqdm

from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import PCA
from numpy.linalg import lstsq
from scipy.spatial.distance import pdist
from scipy.signal import savgol_filter

from geomstats.geometry.base import ImmersedSet
from geomstats.geometry.euclidean import Euclidean
from geomstats.geometry.pullback_metric import PullbackMetric
from geomstats.geometry.special_orthogonal import SpecialOrthogonal

from ..datasets.synthetic_sphere_like import (
    get_s1_synthetic_immersion,
    get_s2_synthetic_immersion,
    get_t2_synthetic_immersion,
    get_scrunchy_immersion,
    get_interlocking_rings_immersion,
    get_flower_scrunchy_immersion
)

from ..datasets.topo_datasets import (
    get_clelia_immersion,
    get_8_curve_immersion,
    get_torus_immersion,
    get_sphere_immersion,
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
    trans = torch.zeros(config.embedding_dim)
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
    elif config.dataset_name == "scrunchy":
        return get_scrunchy_immersion(
            config.radius,
            config.n_wiggles,
            config.geodesic_distortion_amp,
            config.embedding_dim,
            rot,
        )
    elif config.dataset_name == "flower_scrunchy":
        return get_flower_scrunchy_immersion(
            config.radius,
            config.n_wiggles,
            config.geodesic_distortion_amp,
            config.embedding_dim,
            rot,
        )
    elif config.dataset_name == "torus":
        return get_torus_immersion(
            major_radius=config.major_radius,
            minor_radius=config.minor_radius,
            embedding_dim=config.embedding_dim,
            deformation_amp=config.deformation_amp,
            translation=trans,
            rotation=rot
        )
    elif config.dataset_name == "interlocked_tori":
        immersion1 = get_torus_immersion(major_radius=config.major_radius,
                                         minor_radius=config.minor_radius,
                                         embedding_dim=3,
                                         deformation_amp=config.deformation_amp,
                                         translation=torch.zeros(3),
                                         rotation=torch.eye(n=3),
                                         )
        immersion2 = get_torus_immersion(major_radius=config.major_radius,
                                         minor_radius=config.minor_radius,
                                         embedding_dim=3,
                                         deformation_amp=config.deformation_amp,
                                         translation=torch.zeros(3),
                                         rotation=torch.eye(n=3),
                                         )
        return immersion1, immersion2
    elif config.dataset_name == "nested_spheres":
        immersion_inner = get_sphere_immersion(radius=config.minor_radius, embedding_dim=3,
                                               deformation_amp=config.deformation_amp,
                                               translation=torch.zeros(3), rotation=torch.eye(n=3))
        immersion_mid = get_sphere_immersion(radius=config.minor_radius, embedding_dim=3,
                                             deformation_amp=config.deformation_amp,
                                             translation=torch.zeros(3), rotation=torch.eye(n=3))
        immersion_outer = get_sphere_immersion(radius=config.minor_radius, embedding_dim=3,
                                               deformation_amp=config.deformation_amp,
                                               translation=torch.zeros(3), rotation=torch.eye(n=3))
        return immersion_inner, immersion_mid, immersion_outer
    elif config.dataset_name == "s2_synthetic":
        return get_s2_synthetic_immersion(
            radius=config.radius,
            geodesic_distortion_amp=config.geodesic_distortion_amp,
            embedding_dim=config.embedding_dim,
            rot=rot,
        )
    elif config.dataset_name == "t2_synthetic":
        return get_t2_synthetic_immersion(
            config.major_radius,
            config.minor_radius,
            config.geodesic_distortion_amp,
            config.embedding_dim,
            rot,
        )
    elif config.dataset_name == "clelia_curve":
        return get_clelia_immersion(
            r=config.radius,
            c=config.clelia_c,
            embedding_dim=config.embedding_dim,
            translation=trans,
            rotation=rot
        )
    elif config.dataset_name == "8_curve":
        return get_8_curve_immersion(
            embedding_dim=config.embedding_dim,
            translation=trans,
            rotation=rot
        )
    else:
        raise InvalidConfigError(f"Unknown dataset: {config.dataset_name}")


def get_z_grid(config, n_grid_points=200):
    if config.dataset_name in {"s1_synthetic", "scrunchy"}:
        return torch.linspace(0, 2 * gs.pi, n_grid_points)
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


def get_vectors(config, model, data_loader, n_samples):
    if config.verbose:
        print("Forwarding data through model to compute latents and recons...")

    model.eval()
    inputs, latents, recons, labels = [], [], [], []

    with torch.no_grad():
        for x, y in data_loader:
            x = x.to(config.device)
            y = y.to(config.device)

            z, x_recon, _ = model(x)

            inputs.append(x.cpu())
            latents.append(z.cpu())
            recons.append(x_recon.cpu())
            labels.append(y.cpu())

    inputs = torch.cat(inputs, dim=0)
    latents = torch.cat(latents, dim=0)
    recons = torch.cat(recons, dim=0)
    labels = torch.cat(labels, dim=0)

    n_total = latents.shape[0]
    n_samples = min(n_samples, n_total)
    indices = torch.randperm(n_total)[:n_samples]

    # Apply random sampling
    inputs = inputs[indices]
    latents = latents[indices]
    recons = recons[indices]
    labels = labels[indices]

    # Reshaping labels if 1 dim array
    if labels.ndim == 1:
        labels = labels.unsqueeze(1)  # Shape: (N, 1)

    # Sort by label if label dim = 1
    if labels.shape[1] == 1:
        labels = labels.squeeze()
        sort_idx = torch.argsort(labels)

    # lexicographic sort if label dim = 2
    elif labels.shape[1] == 2:
        sort_idx = np.lexsort((labels[:, 1].numpy(), labels[:, 0].numpy()))
        sort_idx = torch.from_numpy(sort_idx)

    # lexicographic sort if label dim = 3, for nested_spheres and interlocked_tori.
    elif labels.shape[1] == 3:
        sort_idx = np.lexsort((
            labels[:, 2].numpy(),  # phi (3rd column)
            labels[:, 1].numpy(),  # theta (2nd column)
            labels[:, 0].numpy(),  # entity index (1st column)
        ))

    else:
        raise NotImplementedError(f"Labels should either be one-dimensional or two-dimensional")

    labels = labels[sort_idx]
    inputs = inputs[sort_idx]
    latents = latents[sort_idx]
    recons = recons[sort_idx]

    return recons, latents, inputs, labels


def _compute_curvature(z_grid, immersion, dim, embedding_dim):
    neural_manifold = NeuralManifoldIntrinsic(
        dim, embedding_dim, immersion, equip=False
    )
    neural_manifold.equip_with_metric(PullbackMetric)
    if dim == 1:
        curv = gs.zeros(len(z_grid), embedding_dim)
        for i, z in tqdm(enumerate(z_grid), desc="Computing curvature from immersion (Manifold Dim = 1)",
                         total=len(z_grid)):
            if not torch.is_tensor(z):
                z = torch.tensor(z)
            z = torch.unsqueeze(z, dim=0)
            curv[i, :] = neural_manifold.metric.mean_curvature_vector(z)
    else:
        curv = torch.full((len(z_grid), embedding_dim), torch.nan)
        for i, z in tqdm(enumerate(z_grid), desc="Computing curvature from immersion  (Manifold Dim > 1)",
                         total=len(z_grid)):
            try:
                curv[i, :] = neural_manifold.metric.mean_curvature_vector(z)
            except Exception as e:
                print(f"An error occurred for i={i}: {e}")
                print(neural_manifold.metric.metric_matrix(z))
    curv_norm = torch.linalg.norm(curv, dim=1, keepdim=True).squeeze()
    return curv, curv_norm


def compute_curvature_learned(config, model, latent_vectors=None, labels=None, n_grid_points=2000):
    if config.verbose:
        print("Computing learned curvature...")
    if config.model_type == 'EuclideanVAE':
        z_grid = latent_vectors
    elif config.model_type == 'VonMisesVAE':
        z_grid = get_z_grid(config, n_grid_points)
    else:
        raise InvalidConfigError(f"Unknown model type: {config.model_type}")
    immersion = get_learned_immersion(model, config)
    manifold_dim = z_grid.shape[1] if z_grid.ndim > 1 else 1
    curv, curv_norm = _compute_curvature(z_grid, immersion, manifold_dim, config.embedding_dim)
    return z_grid, labels, curv, curv_norm


def _old_compute_curvature_true(config, n_grid_points=2000):
    if config.verbose:
        print("Computing true curvature of input dataset using on z-grid...")
    z_grid = get_z_grid(config, n_grid_points)
    immersion = get_true_immersion(config)
    if config.dataset_name in {"s1_synthetic", "interlocking_rings_synthetic", "scrunchy", "clelia_curve", "8_curve",
                               "flower_scrunchy"}:
        manifold_dim = 1
    elif config.dataset_name in {"s2_synthetic", "t2_synthetic", "torus", "interlocked_tori"}:
        manifold_dim = 2
    else:
        raise InvalidConfigError(f"Unknown dataset: {config.dataset_name}")
    result = _compute_curvature(z_grid, immersion, manifold_dim, config.embedding_dim)
    return z_grid, *result


def compute_curvature_true(config, labels=None, n_grid_points=2000):
    if labels is None:
        if config.verbose:
            print("Computing true curvature of input dataset using on z-grid...")
        angles = get_z_grid(config, n_grid_points)
    else:
        if config.verbose:
            print("Computing true curvature of input dataset on given angles...")

    if config.dataset_name in {"s1_synthetic", "interlocking_rings_synthetic", "scrunchy", "clelia_curve", "8_curve",
                               "flower_scrunchy"}:
        angles = labels
        manifold_dim = 1
        immersion = get_true_immersion(config)
        curv, curv_norm = _compute_curvature(angles, immersion, manifold_dim, config.embedding_dim)
    elif config.dataset_name in {"s2_synthetic", "t2_synthetic", "torus"}:
        angles = labels
        manifold_dim = 2
        immersion = get_true_immersion(config)
        curv, curv_norm = _compute_curvature(angles, immersion, manifold_dim, config.embedding_dim)
    elif config.dataset_name in {"nested_spheres", "interlocked_tori"}:
        manifold_dim = 2
        if config.dataset_name == "nested_spheres":
            immersion_inner, immersion_mid, immersion_outer = get_true_immersion(config)
            immersions = [immersion_inner, immersion_mid, immersion_outer]
        elif config.dataset_name == "interlocked_tori":
            immersion_torus1, immersion_torus2 = get_true_immersion(config)
            immersions = [immersion_torus1, immersion_torus2]
        else:
            raise InvalidConfigError(f"Unknown dataset: {config.dataset_name}")
        curv, curv_norm = [], []
        entity_indices = labels[:, 0]
        angles = labels[:, 1:]
        unique_entities = entity_indices.unique(sorted=True)
        for entity in unique_entities:
            entity_index = int(entity.item())
            immersion = immersions[entity_index]
            mask = (entity_indices == entity)
            angles_sub = angles[mask]
            print("angles_sub", angles_sub)
            curv_sub, curv_norm_sub = _compute_curvature(angles_sub, immersion, manifold_dim, 3)
            curv.append(curv_sub)
            curv_norm.append(curv_norm_sub)
        curv = torch.cat(curv)
        curv_norm = torch.cat(curv_norm)
    else:
        raise InvalidConfigError(f"Unknown dataset: {config.dataset_name}")

    return angles, curv, curv_norm


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


def compute_curvature_error_linf(curv1, curv2):
    curv1 = np.array(curv1)
    curv2 = np.array(curv2)
    return np.max(np.abs(curv1 - curv2))


def compute_curvature_error_mse(curv1, curv2, eps=1e-12):
    curv1 = np.array(curv1)
    curv2 = np.array(curv2)
    curv1 = np.clip(curv1, eps, None)
    curv2 = np.clip(curv2, eps, None)

    diff = (curv1 - curv2) ** 2
    mean = diff.sum() / len(curv1)
    return mean


def compute_curvature_error_male(true_curv, approx_curv, eps=1e-12):
    # Computes mean absolute log error

    true_curv = np.asarray(true_curv)
    approx_curv = np.asarray(approx_curv)
    true_curv = np.clip(true_curv, eps, None)
    approx_curv = np.clip(approx_curv, eps, None)

    log_error = np.log(approx_curv / true_curv)
    return np.mean(np.abs(log_error))


def compute_curvature_error_smape(true_curv, approx_curv, eps=1e-12):
    """
    Computes the symmetric mean absolute percentage error (SMAPE) between true and approximate curvatures.
    """
    true_curv = np.asarray(true_curv)
    approx_curv = np.asarray(approx_curv)

    # Ensure non-zero denominator
    denominator = np.clip(np.abs(true_curv) + np.abs(approx_curv), eps, None)
    smape = np.abs(true_curv - approx_curv) / denominator

    return 100 * np.mean(smape)


def compute_curvature_error(z_grid, learned, true, config):
    if config.dataset_name == "s1_synthetic":
        return _compute_curvature_error_s1(z_grid, learned, true)
    elif config.dataset_name in ("s2_synthetic", "t2_synthetic"):
        return _compute_curvature_error_s2(z_grid[:, 0], z_grid[:, 1], learned, true)
    else:
        raise InvalidConfigError(f"Unknown dataset: {config.dataset_name}")


def _get_pointcloud_diameters(points):
    max_dist = np.max(pdist(points, metric='euclidean'))
    min_dist = np.min(pdist(points, metric='euclidean'))
    return max_dist, min_dist


def normalize_curvature_to_input_radius(points, radius, curvatures):
    max_dist, min_dist = _get_pointcloud_diameters(points)
    norm_const = (max_dist + min_dist) / 2
    normed_curvatures = norm_const / radius * curvatures
    return normed_curvatures


# Empiric curvature estimate
def compute_empirical_curvature(config, labels, inputs, latents, recons, k=160):
    if config.dataset_name in {"8_curve", "clelia_curve", "flower_curve", "scrunchy", "flower_scrunchy"}:
        curv_in = estimate_curvature_1d_quadric(inputs, k)
        curv_lat = estimate_curvature_1d_quadric(latents, k)
        curv_rec = estimate_curvature_1d_quadric(recons, k)
    elif config.dataset_name in {"s2_synthetic", "t2_synthetic", "torus", "genus_3", "sphere"}:
        curv_in = estimate_curvature_2d_quadric(inputs, k)
        curv_lat = estimate_curvature_2d_quadric(latents, k)
        curv_rec = estimate_curvature_2d_quadric(recons, k)
    elif config.dataset_name in {"interlocked_tori", "nested_spheres"}:
        entity_indices = labels[:, 0]
        unique_entities = entity_indices.unique(sorted=True)
        curv_in, curv_lat, curv_rec = [], [], []
        for entity in unique_entities:
            mask = (entity_indices == entity)
            inputs_sub = inputs[mask]
            latents_sub = latents[mask]
            recons_sub = recons[mask]

            curv_in_sub = estimate_curvature_2d_quadric(inputs_sub, k)
            curv_lat_sub = estimate_curvature_2d_quadric(latents_sub, k)
            curv_rec_sub = estimate_curvature_2d_quadric(recons_sub, k)

            curv_in.append(curv_in_sub)
            curv_lat.append(curv_lat_sub)
            curv_rec.append(curv_rec_sub)

        curv_in = np.concatenate(curv_in)
        curv_lat = np.concatenate(curv_lat)
        curv_rec = np.concatenate(curv_rec)

    else:
        raise InvalidConfigError(f"Unknown dataset name: {config.dataset_name}")

    # Apply filter to smooth out curves
    if config.smoothing:
        curv_in = savgol_filter(curv_in, 20, 6)
        curv_lat = savgol_filter(curv_lat, 20, 6)
        curv_rec = savgol_filter(curv_rec, 20, 6)
    return curv_in, curv_lat, curv_rec, labels


def estimate_curvature_1d_quadric(points, k=160):
    n, d = points.shape
    nbrs = NearestNeighbors(n_neighbors=k).fit(points)
    _, indices = nbrs.kneighbors(points)
    curvatures = []
    for i in tqdm(range(n), desc="Estimating 1D curvature"):
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


def estimate_curvature_2d_quadric(points, k=200):
    n, d = points.shape
    nbrs = NearestNeighbors(n_neighbors=k).fit(points)
    _, indices = nbrs.kneighbors(points)
    curvatures = []
    for i in tqdm(range(n), desc="Estimating 2D curvature"):
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


def compute_all_curvatures(config, model, recons, latents, inputs, labels):
    # Compute empirical curvatures on full data
    curvature_inputs, curvature_latents, curvature_recons, labels = compute_empirical_curvature(config=config,
                                                                                                labels=labels,
                                                                                                inputs=inputs,
                                                                                                latents=latents,
                                                                                                recons=recons,
                                                                                                k=config.k
                                                                                                )

    # Compute pullback curvature on (ordered) subset of points to reduce computation time
    n_total = len(labels)
    n_points = min(config.n_curv_evaluation_points, n_total)
    sampled_indices = np.random.choice(n_total, size=n_points, replace=False)
    sampled_indices.sort()

    # Subsample data to match subset of curvatures for heat map plotting
    labels_subset = labels[sampled_indices]
    inputs_subset = inputs[sampled_indices]
    latents_subset = latents[sampled_indices]
    recons_subset = recons[sampled_indices]

    _, _, _, curvature_learned = compute_curvature_learned(config, model, latents_subset, labels_subset)

    r_norm = getattr(config, "radius", getattr(config, "major_radius", 1.0))
    curvature_latents_normalized = normalize_curvature_to_input_radius(
        latents_subset, r_norm, curvature_latents[sampled_indices]
    )

    _, _, curvature_true = compute_curvature_true(config, labels_subset)

    # Subsample empirical curvature to match subset for error computation
    curvature_inputs = curvature_inputs[sampled_indices]
    curvature_recons = curvature_recons[sampled_indices]
    curvature_latents = curvature_latents[sampled_indices]

    points = (inputs_subset, latents_subset, recons_subset)
    curvatures = (curvature_true, curvature_inputs, curvature_recons, curvature_latents, curvature_latents_normalized,
                  curvature_learned)

    return labels_subset, points, curvatures


class InvalidConfigError(Exception):
    pass
