from typing import Set

# Explicit model type sets. Keep in sync with actual model classes.
EUCLIDEAN_MODELS: Set[str] = {
    "EuclideanVAE",
    # Add other euclidean-latent models here if applicable, e.g., "EuclideanAE"
}

NON_EUCLIDEAN_MODELS: Set[str] = {
    "VMFSphericalVAE",
    "VMFToroidalVAE",
    "SphericalAE",
    "ToroidalAE",
    "VMToroidalVAE",
}


def is_euclidean_model(model_type: str) -> bool:
    """
    Return True if the model has a Euclidean latent manifold.
    """
    return model_type in EUCLIDEAN_MODELS


def is_non_euclidean_model(model_type: str) -> bool:
    """
    Return True if the model has a non-Euclidean latent manifold (e.g., spherical, toroidal).
    """
    return model_type in NON_EUCLIDEAN_MODELS
