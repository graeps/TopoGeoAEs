import warnings

from gtda.homology import VietorisRipsPersistence
from gtda.diagrams import PairwiseDistance
from gtda.diagrams import Scaler



def compute_bottleneck_dist(points1, points2, homology_dimensions):
    print("computing persistence diagrams")
    point_clouds = [points1, points2]
    persistence = VietorisRipsPersistence(
        metric="euclidean",
        homology_dimensions=homology_dimensions,
        n_jobs=6,
        collapse_edges=True,
    )
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        diagrams = persistence.fit_transform(point_clouds)

        scaler = Scaler()
        diagrams_scaled = scaler.fit_transform(diagrams)

        bottleneck = PairwiseDistance(metric="bottleneck", n_jobs=6)
        distances = bottleneck.fit_transform(diagrams_scaled)

    print("Distance matrix:\n", distances)
    print("Distance between diagrams (0,1):", distances[0, 1])

    return diagrams, distances

