import os
import numpy as np
import torch
from tqdm import tqdm

from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import PCA
from numpy.linalg import lstsq
from scipy.signal import savgol_filter
from scipy.spatial import cKDTree

from geomstats.geometry.base import ImmersedSet
from geomstats.geometry.euclidean import Euclidean
from geomstats.geometry.pullback_metric import PullbackMetric
from geomstats.geometry.special_orthogonal import SpecialOrthogonal

from ..datasets.synthetic_sphere_like import (
    get_s1_synthetic_immersion,
    get_s2_synthetic_immersion,
    get_t2_synthetic_immersion,
    get_scrunchy_immersion,
    get_flower_scrunchy_immersion,
)

from ..datasets.topo_datasets import (
    get_clelia_immersion,
    get_8_curve_immersion,
    get_torus_immersion,
    get_sphere_immersion,
    get_scrunchy_dim_n,
    get_sphere_high_dim_bump_immersion
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
        if config.dataset_name in {"s1_synthetic", "scrunchy_dim_n"}:
            z = gs.array([gs.cos(angle[0]), gs.sin(angle[0])])

        elif config.dataset_name in {"s2_synthetic", "sphere"}:
            theta, phi = angle
            z = gs.array([
                gs.sin(theta) * gs.cos(phi),
                gs.sin(theta) * gs.sin(phi),
                gs.cos(theta),
            ])

        elif config.dataset_name in {"t2_synthetic", "torus"}:
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
    if config.model_type in {'VonMisesVAE', 'VMFSphericalVAE', 'VMFToroidalVAE', 'VMToroidalVAE', 'SphericalAE',
                             'ToroidalAE'}:
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
    elif config.dataset_name == "sphere_high_dim":
        bump_dims = [config.embedding_dim - 3, config.embedding_dim - 2, config.embedding_dim - 1]
        bump_centers = [(torch.pi / 4, torch.pi / 2), (torch.pi / 4, 3 * torch.pi / 2), (torch.pi / 2, torch.pi / 2)]
        return get_sphere_high_dim_bump_immersion(radius=config.radius, deformation_amp=config.deformation_amp,
                                                  bump_dim=bump_dims, bump_center=bump_centers,
                                                  embedding_dim=config.embedding_dim, translation=trans,
                                                  rotation=rot)
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
    elif config.dataset_name == "scrunchy_dim_n":
        return get_scrunchy_dim_n(
            deformation_amp=config.geodesic_distortion_amp,
            embedding_dim=config.embedding_dim,
            translation=trans,
            rotation=rot
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
    elif config.dataset_name == "sphere":
        return get_sphere_immersion(radius=config.radius, embedding_dim=config.embedding_dim,
                                    deformation_amp=config.deformation_amp,
                                    translation=trans, rotation=rot)
    elif config.dataset_name == "nested_spheres":
        immersion_inner = get_sphere_immersion(radius=config.minor_radius, embedding_dim=3,
                                               deformation_amp=config.deformation_amp,
                                               translation=torch.zeros(3), rotation=torch.eye(n=3))
        immersion_mid = get_sphere_immersion(radius=config.mid_radius, embedding_dim=3,
                                             deformation_amp=config.deformation_amp,
                                             translation=torch.zeros(3), rotation=torch.eye(n=3))
        immersion_outer = get_sphere_immersion(radius=config.major_radius, embedding_dim=3,
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


def get_z_grid(config, n_grid_points):
    eps = 1e-4
    if config.dataset_name in {"s1_synthetic", "scrunchy", "scrunchy_dim_n"}:
        z_grid = torch.linspace(eps, 2 * gs.pi - eps, n_grid_points)
    elif config.dataset_name in {"s2_synthetic", "sphere"}:
        thetas = gs.arccos(np.linspace(0.99, -0.99, int(np.sqrt(n_grid_points))))
        phis = gs.linspace(0.01, 2 * gs.pi - 0.01, int(np.sqrt(n_grid_points)))
        z_grid = torch.cartesian_prod(thetas, phis)
    elif config.dataset_name in {"t2_synthetic", "torus"}:
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


def get_vectors(config, model, data_loader, n_samples, save_dir="./learned_vectors"):
    if config.verbose:
        print("Forwarding data through model to compute latents and recons...")

    model.eval()
    inputs, latents, recons, labels = [], [], [], []

    with torch.no_grad():
        for x, y in data_loader:
            x = x.to(config.device)
            y = y.to(config.device)

            if config.model_type in {"EuclideanVAE", "VMToroidalVAE", "VMFToroidalVAE", "VMFSphericalVAE"}:
                z, x_recon, _ = model(x)
            elif config.model_type in {"EuclideanAE", "ParamAE", "SphericalAE", "ToroidalAE"}:
                angles, z, x_recon = model(x)
            else:
                raise NotImplementedError

            inputs.append(x.cpu())
            latents.append(z.cpu())
            recons.append(x_recon.cpu())
            labels.append(y.cpu())

    inputs = torch.cat(inputs, dim=0)
    latents = torch.cat(latents, dim=0)
    recons = torch.cat(recons, dim=0)
    labels = torch.cat(labels, dim=0)

    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
        name = f"vectors_{config.experiment}_.pt"
        save_path = os.path.join(save_dir, name)
        torch.save({
            'inputs': inputs,
            'latents': latents,
            'recons': recons,
            'labels': labels
        }, save_path)

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
    torch.unsqueeze(z_grid[0], dim=0)
    if dim == 1:
        curv = gs.zeros(len(z_grid), embedding_dim)
        for i, z in tqdm(enumerate(z_grid), desc="Computing curvature from immersion (Manifold Dim = 1)",
                         total=len(z_grid), leave=False):
            z = torch.unsqueeze(z, dim=0)
            curv[i, :] = neural_manifold.metric.mean_curvature_vector(z)
    else:
        curv = torch.full((len(z_grid), embedding_dim), torch.nan)
        for i, z in tqdm(enumerate(z_grid), desc="Computing curvature from immersion  (Manifold Dim > 1)",
                         total=len(z_grid), leave=False):
            try:
                curv[i, :] = neural_manifold.metric.mean_curvature_vector(z)
            except Exception as e:
                print(f"An error occurred for i={i}: {e}")
    curv_norm = torch.linalg.norm(curv, dim=1, keepdim=True).squeeze()

    # Apply quantile-based clipping to suppress numerical spikes
    valid_mask = ~torch.isnan(curv_norm)
    if valid_mask.any():
        q_low, q_high = torch.quantile(curv_norm[valid_mask], torch.tensor((0.01, 0.99)))
        curv_norm = torch.clamp(curv_norm, min=q_low.item(), max=q_high.item())
    curv_norm = torch.clamp(curv_norm, min=0.0)

    return curv, curv_norm


def compute_curvature_learned(config, model, latents=None, labels=None, n_grid_points=2000):
    if config.verbose:
        print("Computing learned curvature...")
    if config.model_type == 'EuclideanVAE':
        z_grid = latents
    elif config.model_type in {'VMFSphericalVAE', "VMFToroidalVAE", 'SphericalAE', 'ToroidalAE'}:
        if config.latent_dim == 2:
            anchore = (latents[0], latents[10])
            z_grid = get_z_grid(config=config, n_grid_points=n_grid_points)
            #z_grid = shift_z_grid(z_grid, anchore, config)
        elif config.latent_dim == 3:
            z_grid = get_z_grid(config=config, n_grid_points=n_grid_points)
    elif config.model_type == 'VMToroidalVAE':
        z_grid = get_z_grid(config=config, n_grid_points=n_grid_points)
    else:
        raise InvalidConfigError(f"Unknown model type: {config.model_type}")
    immersion = get_learned_immersion(model=model, config=config)
    if config.dataset_name in {"s1_synthetic", "interlocking_rings_synthetic", "scrunchy", "clelia_curve", "8_curve",
                               "flower_scrunchy", "scrunchy_dim_n"}:
        manifold_dim = 1
    elif config.dataset_name in {"s2_synthetic", "t2_synthetic", "sphere", "sphere_high_dim", "torus", "wiggling_tube",
                                 "interlocked_tori", "nested_spheres"}:
        manifold_dim = 2
    else:
        raise NotImplementedError({"unknown dataset"})
    curv, curv_norm = _compute_curvature(z_grid=z_grid, immersion=immersion, dim=manifold_dim,
                                         embedding_dim=config.embedding_dim)
    return z_grid, labels, curv, curv_norm


def compute_curvature_true(config, labels=None, n_grid_points=2000, cache_dir="./curvature_cache"):
    os.makedirs(cache_dir, exist_ok=True)
    if config.dataset_name in {"s1_synthetic", "s2_synthetic", "t2_synthetic", "scrunchy", "scrunchy_dim_n"}:
        deformation = config.geodesic_distortion_amp
    else:
        deformation = config.deformation_amp
    name = f"{config.dataset_name}_{config.n_points_pullback_curv}_{deformation}.pt"
    cache_path = os.path.join(cache_dir, name)

    if os.path.exists(cache_path):
        if config.verbose:
            print(f"Loading cached curvature from {cache_path}")
        return torch.load(cache_path)

    if config.verbose:
        print("Computing true curvature...")

    if config.compute_emp_curv:
        angles = labels
    else:
        angles = get_z_grid(config, n_grid_points)

    if config.dataset_name in {"s1_synthetic", "interlocking_rings_synthetic", "scrunchy", "clelia_curve", "8_curve",
                               "flower_scrunchy", "scrunchy_dim_n"}:
        immersion = get_true_immersion(config)
        curv, curv_norm = _compute_curvature(angles, immersion, 1, config.embedding_dim)

    elif config.dataset_name in {"s2_synthetic", "t2_synthetic", "torus", "sphere_high_dim", "sphere"}:
        immersion = get_true_immersion(config)
        curv, curv_norm = _compute_curvature(angles, immersion, 2, config.embedding_dim)

    elif config.dataset_name in {"nested_spheres", "nested_spheres_high_dim", "interlocked_tori", "interlocked_tubes"}:
        immersions = get_true_immersion(config)
        entity_indices, angles = labels[:, 0], labels[:, 1:]
        curv, curv_norm = [], []
        for i in entity_indices.unique(sorted=True):
            mask = (entity_indices == i)
            curv_i, curv_norm_i = _compute_curvature(
                angles[mask], immersions[int(i.item())], 2, 3
            )
            curv.append(curv_i)
            curv_norm.append(curv_norm_i)
        curv = torch.cat(curv)
        curv_norm = torch.cat(curv_norm)

    else:
        raise InvalidConfigError(f"Unknown dataset: {config.dataset_name}")

    torch.save((angles, curv, curv_norm), cache_path)
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
        raise InvalidConfigError(f"Unknown XX dataset: {config.dataset_name}")


# Empiric curvature estimate
def compute_empirical_curvature(config, labels, inputs, latents, recons, k=160):
    if config.dataset_name in {"8_curve", "clelia_curve", "flower_curve", "scrunchy", "flower_scrunchy",
                               "scrunchy_dim_n",
                               "s1_synthetic"}:
        curv_in = estimate_curvature_1d_quadric(inputs, k)
        curv_lat = estimate_curvature_1d_quadric(latents, k)
        curv_rec = estimate_curvature_1d_quadric(recons, k)
    elif config.dataset_name in {"s2_synthetic", "t2_synthetic", "torus", "genus_3", "sphere", "sphere_high_dim",
                                 "wiggling_tube", "flat_torus_embedding"}:
        curv_in = estimate_curvature_2d_quadric(inputs, k)
        curv_lat = estimate_curvature_2d_quadric(latents, k)
        curv_rec = estimate_curvature_2d_quadric(recons, k)
    elif config.dataset_name in {"interlocked_tori", "nested_spheres", "nested_spheres_high_dim", "interlocked_tubes"}:
        entity_indices = labels[:, 0]
        unique_entities = entity_indices.unique(sorted=True)
        curv_in, curv_lat, curv_rec = [], [], []
        for entity in unique_entities:
            mask = (entity_indices == entity)
            inputs_sub = inputs[mask]
            latents_sub = latents[mask]
            recons_sub = recons[mask]

            if entity == 100:
                curv_in_sub = torch.full((inputs_sub.shape[0],), 0.0)
                curv_lat_sub = curv_in_sub
                curv_rec_sub = curv_in_sub
            else:
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
    for i in tqdm(range(n), desc="Estimating 2D curvature", leave=False):
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


def compute_all_curvatures(config, model, recons, latents, inputs, labels, save_dir="./curvatures"):
    # Compute pullback curvature on (ordered) subset of points to reduce computation time
    n_total = len(labels)
    n_points = int(np.sqrt(min(config.n_points_pullback_curv, n_total))) ** 2
    sampled_indices = np.random.choice(n_total, size=n_points, replace=False)
    sampled_indices.sort()

    # Subsample data to match subset of curvatures for heat map plotting
    labels_sub = labels[sampled_indices]
    inputs_sub = inputs[sampled_indices]
    latents_sub = latents[sampled_indices]
    recons_sub = recons[sampled_indices]

    # Compute empirical curvatures on full data
    if config.compute_emp_curv:
        curv_in, curv_lat, curv_rec, labels = compute_empirical_curvature(config=config, labels=labels, inputs=inputs,
                                                                          latents=latents, recons=recons, k=config.k
                                                                          )
        norm_factor = np.max(curv_in[sampled_indices]) / np.max(curv_lat[sampled_indices])
        curv_lat_norm = curv_lat * norm_factor
        curv_lat_norm_sub = curv_lat_norm[sampled_indices]
        # Subsample empirical curvature to match subset for error computation
        curv_in_sub = curv_in[sampled_indices]
        curv_rec_sub = curv_rec[sampled_indices]
        curv_lat_sub = curv_lat[sampled_indices]
    else:
        curv_in_sub = np.zeros(n_points)
        curv_rec_sub = np.zeros(n_points)
        curv_lat_sub = np.zeros(n_points)
        curv_lat_norm_sub = np.zeros(n_points)
        curv_lat_norm = np.zeros(len(labels))
        curv_in = np.zeros(len(labels))
        curv_lat = np.zeros(len(labels))
        curv_rec = np.zeros(len(labels))
    if config.compute_true_curv:
        z_grid, _, curv_true = compute_curvature_true(config=config, labels=labels_sub, n_grid_points=n_points)
    else:
        curv_true = np.zeros(n_points)
        z_grid = np.zeros(n_points)
    if config.compute_learned_curv:
        _, _, _, curv_learned = compute_curvature_learned(config=config, model=model, latents=latents_sub,
                                                          labels=labels_sub, n_grid_points=n_points)
        if config.model_type in {"VMFSphericalVAE", "SphericalAE"}:
            curv_learned_rotated = compute_curvature_rotated(learned_curvature=curv_learned, latents=latents_sub,
                                                             labels=labels_sub, z_grid=z_grid)
        else:
            curv_learned_rotated = curv_learned
    else:
        curv_learned = np.zeros(n_points)
        curv_learned_rotated = np.zeros(n_points)

    curvatures_sub = (labels_sub, curv_in_sub, curv_rec_sub, curv_lat_sub, curv_lat_norm_sub, curv_true,
                      curv_learned, curv_learned_rotated, z_grid)
    curvatures_emp_full = (labels, curv_in, curv_lat, curv_lat_norm, curv_rec)

    points_sub = (inputs_sub, latents_sub, recons_sub)

    points = (inputs, latents, recons)

    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
        filename = f"curvatures_{config.experiment}.pt"
        save_path = os.path.join(save_dir, filename)

        torch.save({
            'curvatures_sub': curvatures_sub,
            'curvatures_emp_full': curvatures_emp_full,
            'points_sub': points_sub,
            'points': points,
        }, save_path)

        if config.verbose:
            print(f"Saved curvatures to {save_path}")

    return points_sub, curvatures_sub, curvatures_emp_full, points


def compute_curvature_rotated(learned_curvature, latents, labels, z_grid):
    latents = latents.to(torch.float32)
    labels = labels.to(torch.float32)
    z_grid = z_grid.to(torch.float32)

    def angle_to_cartesian(theta):
        x = torch.cos(theta)
        y = torch.sin(theta)
        return torch.stack([x, y], dim=1).to(torch.float32)

    def spherical_to_cartesian(theta, phi):
        x = torch.sin(theta) * torch.cos(phi)
        y = torch.sin(theta) * torch.sin(phi)
        z = torch.cos(theta)
        return torch.stack([x, y, z], dim=1).to(torch.float32)

    def compute_rotation_matrix(source, target):
        A = source.T @ target
        U, _, Vt = torch.svd(A)
        R = Vt @ U.T
        # # Ensure det(R)=+1
        # if torch.det(R) < 0:
        #     Vt[-1, :] *= -1
        #     R = Vt @ U.T
        return R

    # Compute rotation
    if latents.shape[1] == 2:
        theta_grid = z_grid[:,0] if z_grid.ndim > 1 else z_grid
        vecs = angle_to_cartesian(theta_grid)

        label_angles = labels[:,0] if labels.ndim > 1 else labels
        label_cartesian = angle_to_cartesian(label_angles)
    elif latents.shape[1] == 3:
        vecs = spherical_to_cartesian(z_grid[:, 0], z_grid[:, 1])
        label_cartesian = spherical_to_cartesian(labels[:, 0], labels[:, 1])
    else:
        raise NotImplementedError

    R = compute_rotation_matrix(latents, label_cartesian)
    vecs_rotated = vecs @ R
    tree = cKDTree(vecs.numpy())
    _, nn_indices = tree.query(vecs_rotated.numpy(), k=1)
    rotated_curvature = learned_curvature[nn_indices]

    return rotated_curvature


class InvalidConfigError(Exception):
    pass
