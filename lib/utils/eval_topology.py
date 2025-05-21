import warnings

from gtda.homology import VietorisRipsPersistence
from gtda.diagrams import PairwiseDistance
from gtda.diagrams import BettiCurve
from gtda.diagrams import Scaler


def compute_persistence_diagrams(point_clouds, homology_dimensions, scale):
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
    print("Computing bottleneck distance")
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        bottleneck = PairwiseDistance(metric="bottleneck", n_jobs=6)
        distances = bottleneck.fit_transform(diagrams)

    return distances


def compute_betti_curve(diagrams):
    print("Computing betti curves")
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        BC = BettiCurve(n_bins=100, n_jobs=6)
        betti_numbers = BC.fit_transform(diagrams)
        samplings = BC.samplings_
        betti_curves = (betti_numbers, samplings)

    return betti_curves


def compare_persistent_homology(pointclouds, homology_dimensions, scale):
    diagrams = compute_persistence_diagrams(pointclouds, homology_dimensions, scale)
    bottleneck_dist = compute_bottleneck_dist(diagrams)
    betti_curves = compute_betti_curve(diagrams)

    return diagrams, betti_curves, bottleneck_dist
