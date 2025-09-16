import warnings

from gtda.homology import VietorisRipsPersistence
from gtda.diagrams import PairwiseDistance, BettiCurve, Scaler


def compute_persistence_diagrams(point_clouds, homology_dimensions, scale):
    """
    Compute persistence diagrams for a collection of point clouds.

    Uses Vietoris–Rips filtrations to compute homology in the specified dimensions.

    Args:
        point_clouds (array-like of shape [n_samples, n_points, n_features]):
            Collection of point clouds.
        homology_dimensions (list of int):
            Homology dimensions to compute (e.g., [0, 1] for connected components and loops).
        scale (bool):
            If True, apply diagram scaling to normalize birth–death values.

    Returns:
        ndarray of shape [n_samples, n_intervals, 3]:
            Persistence diagrams for each input point cloud.
    """
    print("Computing persistence diagrams")
    persistence = VietorisRipsPersistence(
        metric="euclidean",
        homology_dimensions=homology_dimensions,
        n_jobs=6,
        collapse_edges=True,
    )
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        diagrams = persistence.fit_transform(point_clouds)

        if scale:
            scaler = Scaler()
            diagrams = scaler.fit_transform(diagrams)

    return diagrams


def compute_bottleneck_dist(diagrams):
    """
    Compute pairwise bottleneck distances between persistence diagrams.

    Args:
        diagrams (ndarray of shape [n_samples, n_intervals, 3]):
            Persistence diagrams produced by Giotto-TDA.

    Returns:
        ndarray of shape [n_samples, n_samples]:
            Symmetric matrix of pairwise bottleneck distances.
    """
    print("Computing bottleneck distance")
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        bottleneck = PairwiseDistance(metric="bottleneck", n_jobs=6)
        distances = bottleneck.fit_transform(diagrams)

    return distances


def compute_betti_curve(diagrams):
    """
    Compute Betti curves from persistence diagrams.

    Betti curves capture the evolution of Betti numbers (topological features)
    across the filtration parameter.

    Args:
        diagrams (ndarray of shape [n_samples, n_intervals, 3]):
            Persistence diagrams.

    Returns:
        Tuple[ndarray, ndarray]:
            - betti_numbers: Array of shape [n_samples, n_bins] with Betti numbers.
            - samplings: Array of filtration parameter values corresponding to bins.
    """
    print("Computing betti curves")
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        BC = BettiCurve(n_bins=100, n_jobs=6)
        betti_numbers = BC.fit_transform(diagrams)
        samplings = BC.samplings_
        betti_curves = (betti_numbers, samplings)

    return betti_curves


def compare_persistent_homology(pointclouds, homology_dimensions, scale):
    """
    Compute and compare topological signatures of point clouds.

    Combines persistence diagrams, Betti curves, and pairwise bottleneck distances
    to provide a multi-faceted topological analysis.

    Args:
        pointclouds (array-like of shape [n_samples, n_points, n_features]):
            Collection of point clouds.
        homology_dimensions (list of int):
            Homology dimensions to compute (e.g., [0, 1]).
        scale (bool):
            Whether to scale persistence diagrams to unit ranges.

    Returns:
        Tuple[ndarray, Tuple[ndarray, ndarray], ndarray]:
            - diagrams: Persistence diagrams.
            - betti_curves: (betti_numbers, samplings) as returned by compute_betti_curve.
            - bottleneck_dist: Pairwise bottleneck distance matrix.
    """
    diagrams = compute_persistence_diagrams(pointclouds, homology_dimensions, scale)
    bottleneck_dist = compute_bottleneck_dist(diagrams)
    betti_curves = compute_betti_curve(diagrams)

    return diagrams, betti_curves, bottleneck_dist
