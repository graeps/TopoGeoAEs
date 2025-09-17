from typing import Literal

from ..errors import InvalidConfigError

# Dataset category constants (single source of truth)
ONE_D_DATASETS = {
    "s1_low",
    "interlocking_rings_synthetic",
    "scrunchy",
    "clelia_curve",
    "8_curve",
    "flower_scrunchy",
    "s1_high",
}

TWO_D_DATASETS = {
    "s2_low",
    "t2_low",
    "s2_high",
    "sphere_high_dim",
    "t2_high",
    "wiggling_tube",
    "interlocked_tori",
    "nested_spheres",
    "nested_spheres_high_dim",
    "interlocked_tubes",
}

# Datasets where curvature must be computed per entity and concatenated
ENTITY_DATASETS = {
    "nested_spheres",
    "nested_spheres_high_dim",
    "interlocked_tori",
    "interlocked_tubes",
}


def get_dataset_category(dataset_name: str) -> Literal["1d", "2d", "multi_entity"]:
    """
    Return the dataset category:
      - "1d" for intrinsically 1D manifolds
      - "2d" for intrinsically 2D manifolds
      - "multi_entity" for datasets composed of multiple entities/components
    """
    if dataset_name in ONE_D_DATASETS:
        return "1d"
    if dataset_name in TWO_D_DATASETS:
        return "2d"
    if dataset_name in ENTITY_DATASETS:
        return "multi_entity"
    raise InvalidConfigError(f"Unknown dataset name: {dataset_name}")


def get_manifold_dim(dataset_name: str) -> int:
    """
    Return the intrinsic manifold dimension for a dataset.
    Use get_dataset_category() when you need to distinguish entity datasets.

    Raises:
        InvalidConfigError if dataset is unknown or belongs to ENTITY_DATASETS.
    """
    category = get_dataset_category(dataset_name)
    if category == "1d":
        return 1
    if category == "2d":
        return 2
    raise InvalidConfigError(
        f"Dataset '{dataset_name}' is an entity dataset; "
        f"use ENTITY_DATASETS to handle per-entity curvature."
    )
