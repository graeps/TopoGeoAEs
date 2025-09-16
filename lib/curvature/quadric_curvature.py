from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import PCA
from numpy.linalg import lstsq
from scipy.signal import savgol_filter
import numpy as np
from tqdm import tqdm
import torch

from .errors import InvalidConfigError


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


def compute_empirical_curvature(config, labels, inputs, latents, recons, k=160):
    if config.dataset_name in {"8_curve", "clelia_curve", "flower_curve", "scrunchy", "flower_scrunchy",
                               "s1_high",
                               "s1_low"}:
        curv_in = estimate_curvature_1d_quadric(inputs, k)
        curv_lat = estimate_curvature_1d_quadric(latents, k)
        curv_rec = estimate_curvature_1d_quadric(recons, k)
    elif config.dataset_name in {"s2_low", "t2_low", "t2_high", "genus_3", "s2_high", "sphere_high_dim",
                                 "wiggling_tube", "flat_t2_high_embedding"}:
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
