from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import PCA
from numpy.linalg import lstsq
from scipy.signal import savgol_filter
import numpy as np
from tqdm import tqdm

from lib.errors import InvalidConfigError


def estimate_curvature_1d_quadric(points, k=160):
    """
    Estimates the 1D curvature of a set of points using a quadratic approximation. The method
    fits a 1D quadratic curve to local neighborhoods of points and computes coefficients that
    represent the curvature.

    Args:
        points (ndarray): A NumPy array of shape (n, d), where `n` is the number of points
            and `d` is the dimensionality of each point. These represent the coordinates
            of the points for curvature estimation.
        k (int, optional): The number of nearest neighbors to consider for estimating
            local curvature. Defaults to 160.

    Returns:
        ndarray: A NumPy array of shape (n,) containing the curvature values for each point.
    """
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
    """
    Estimate the curvature of a point cloud in 2D space using a quadric surface
    fitting method. The function computes local neighborhoods for each point in
    the input, fits a quadratic surface to these neighborhoods, and estimates
    the curvature.

    Args:
        points (numpy.ndarray): A 2D numpy array of points with shape (n, d),
            where n is the number of points and d is the dimensionality of
            each point.
        k (int): The number of nearest neighbors to consider for local
            curvature estimation (default is 200).

    Returns:
        numpy.ndarray: A 1D numpy array of estimated curvatures for each point
            in the input point cloud.
    """
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


# Dataset category constants (centralized) and getters
from ..datasets.lookup import ONE_D_DATASETS, TWO_D_DATASETS, ENTITY_DATASETS, get_dataset_category


def _get_estimator_for_dataset(dataset_name: str):
    """Return the appropriate curvature estimator callable based on dataset type (or None for entity datasets)."""
    category = get_dataset_category(dataset_name)
    if category == "1d":
        return estimate_curvature_1d_quadric
    if category == "2d":
        return estimate_curvature_2d_quadric
    if category == "entity":
        return None
    # Safety: get_dataset_category raises for unknown names
    raise InvalidConfigError(f"Unknown dataset name: {dataset_name}")


def _apply_smoothing(do_smooth: bool, curv: np.ndarray, window_length: int = 20, polyorder: int = 6):
    """Optionally apply Savitzky–Golay smoothing to a single curvature array."""
    if not do_smooth:
        return curv
    return savgol_filter(curv, window_length, polyorder)

def compute_quadric_curvature(config, labels, points, k: int = 160):
    """
    Compute empirical curvature for a single set of points, depending on the dataset category
    specified in the configuration. Uses 1D or 2D quadric approximation as appropriate. For
    datasets with discrete entities, curvature is computed per entity and concatenated.

    Args:
        config: Configuration object with 'dataset_name' and 'smoothing' attributes.
        labels: Tensor of labels; for entity datasets, labels[:, 0] contains entity indices.
        points: Array-like (numpy or torch) of shape (n, d) representing the point cloud to analyze.
        k: Number of neighbors for local curvature estimation.

    Returns:
        Tuple:
        - curv: 1D numpy.ndarray with curvature values for the provided points.
    """
    estimator = _get_estimator_for_dataset(config.dataset_name)

    if estimator is not None:
        # Standard 1D/2D datasets
        curv = estimator(points, k)
    elif config.dataset_name in ENTITY_DATASETS:
        # Multi-entity datasets: compute per-entity then concatenate
        entity_indices = labels[:, 0]
        unique_entities = entity_indices.unique(sorted=True)

        curv_list = []
        for entity in unique_entities:
            mask = (entity_indices == entity)
            pts_sub = points[mask]

            # Special-case: flat entity labeled as 100, for datasets with manifold of dimensions > 2
            if int(entity) == 100:
                curv_sub = np.zeros(pts_sub.shape[0])
            else:
                curv_sub = estimate_curvature_2d_quadric(pts_sub, k)

            curv_list.append(curv_sub)

        curv = np.concatenate(curv_list)
    else:
        # Safety; _get_estimator_for_dataset already guards unknown names
        raise InvalidConfigError(f"Unknown dataset name: {config.dataset_name}")

    # Optional smoothing
    curv = _apply_smoothing(config.smoothing, curv)
    return curv
