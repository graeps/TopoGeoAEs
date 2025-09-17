import os
from typing import Any, Callable, Tuple, Union

from lib.errors import InvalidConfigError

# Must be set before importing geomstats backend
os.environ["GEOMSTATS_BACKEND"] = "pytorch"

import numpy as np
import torch
import geomstats.backend as gs
from geomstats.geometry.base import ImmersedSet
from geomstats.geometry.euclidean import Euclidean
from geomstats.geometry.special_orthogonal import SpecialOrthogonal

from lib.datasets.datasets_main import (
    get_s1_low_immersion,
    get_s2_low_immersion,
    get_t2_low_immersion,
    get_t2_high_immersion,
    get_s2_high_immersion,
    get_s1_high,
)
from lib.datasets.datasets_other import (
    get_clelia_immersion,
    get_8_curve_immersion,
    get_scrunchy_immersion,
    get_sphere_high_dim_bump_immersion,
)
from lib.models.lookup import is_euclidean_model, is_non_euclidean_model

__all__ = [
    "get_z_grid",
    "NeuralManifoldIntrinsic",
    "get_true_immersion",
    "get_learned_immersion",
]


def get_z_grid(config: Any, n_grid_points: int) -> torch.Tensor:
    """
    Build a grid in the latent/z space for visualization or sampling.

    For 1D manifolds returns a 1D grid of angles.
    For spheres and tori returns a Cartesian product of (theta, phi).

    Raises:
        InvalidConfigError: if the dataset is unknown.
    """
    eps = 1e-4
    if config.dataset_name in {"s1_low", "scrunchy", "s1_high"}:
        z_grid = torch.linspace(eps, 2 * gs.pi - eps, n_grid_points)
    elif config.dataset_name in {"s2_low", "s2_high"}:
        side = int(np.sqrt(n_grid_points))
        thetas = gs.arccos(np.linspace(0.99, -0.99, side))
        phis = gs.linspace(0.01, 2 * gs.pi - 0.01, side)
        z_grid = torch.cartesian_prod(thetas, phis)
    elif config.dataset_name in {"t2_low", "t2_high"}:
        side = int(np.sqrt(n_grid_points))
        thetas = gs.linspace(eps, 2 * gs.pi - eps, side)
        phis = gs.linspace(eps, 2 * gs.pi - eps, side)
        z_grid = torch.cartesian_prod(thetas, phis)
    else:
        raise InvalidConfigError(f"Unknown dataset: {config.dataset_name}")
    return z_grid


class NeuralManifoldIntrinsic(ImmersedSet):
    """
    ImmersedSet backed by a neural immersion function.

    The immersion is provided externally (e.g., a decoder), and the embedding
    space is Euclidean with the specified dimension.
    """

    def __init__(
        self,
        dim: int,
        neural_embedding_dim: int,
        neural_immersion: Callable[[torch.Tensor], torch.Tensor],
        equip: bool = True,
    ) -> None:
        self.neural_embedding_dim = neural_embedding_dim
        super().__init__(dim=dim, equip=equip)
        self.neural_immersion = neural_immersion

    def immersion(self, point: torch.Tensor) -> torch.Tensor:
        return self.neural_immersion(point)

    def _define_embedding_space(self) -> Euclidean:
        return Euclidean(dim=self.neural_embedding_dim)


def get_true_immersion(config: Any) -> Union[
    Callable[[torch.Tensor], torch.Tensor],
    Tuple[Callable[[torch.Tensor], torch.Tensor], ...],
]:
    """
    Return the ground-truth immersion(s) for a dataset as callables.

    Some datasets return a single immersion, others return a tuple of immersions
    (e.g., interlocked objects or nested families).
    """
    rot = torch.eye(n=config.embedding_dim)
    trans = torch.zeros(config.embedding_dim)
    if getattr(config, "rotation", None) == "random":
        rot = SpecialOrthogonal(n=config.embedding_dim).random_point()

    if config.dataset_name == "s1_low":
        return get_s1_low_immersion(
            config.deformation_type,
            config.radius,
            config.n_wiggles,
            config.deformation_amp,
            config.embedding_dim,
            rot,
        )
    elif config.dataset_name == "sphere_high_dim":
        bump_dims = [
            config.embedding_dim - 3,
            config.embedding_dim - 2,
            config.embedding_dim - 1,
        ]
        bump_centers = [
            (torch.pi / 4, torch.pi / 2),
            (torch.pi / 4, 3 * torch.pi / 2),
            (torch.pi / 2, torch.pi / 2),
        ]
        return get_sphere_high_dim_bump_immersion(
            radius=config.radius,
            deformation_amp=config.deformation_amp,
            bump_dim=bump_dims,
            bump_center=bump_centers,
            embedding_dim=config.embedding_dim,
            rotation=rot,
        )
    elif config.dataset_name == "scrunchy":
        return get_scrunchy_immersion(
            config.radius,
            config.n_wiggles,
            config.deformation_amp,
            config.embedding_dim,
            rot,
        )
    elif config.dataset_name == "s1_high":
        return get_s1_high(
            deformation_amp=config.deformation_amp,
            embedding_dim=config.embedding_dim,
            translation=trans,
            rotation=rot,
        )
    elif config.dataset_name == "t2_high":
        return get_t2_high_immersion(
            major_radius=config.major_radius,
            minor_radius=config.minor_radius,
            embedding_dim=config.embedding_dim,
            deformation_amp=config.deformation_amp,
            translation=trans,
            rotation=rot,
        )
    elif config.dataset_name == "interlocked_tori":
        immersion1 = get_t2_high_immersion(
            major_radius=config.major_radius,
            minor_radius=config.minor_radius,
            embedding_dim=3,
            deformation_amp=config.deformation_amp,
            translation=torch.zeros(3),
            rotation=torch.eye(n=3),
        )
        immersion2 = get_t2_high_immersion(
            major_radius=config.major_radius,
            minor_radius=config.minor_radius,
            embedding_dim=3,
            deformation_amp=config.deformation_amp,
            translation=torch.zeros(3),
            rotation=torch.eye(n=3),
        )
        return immersion1, immersion2
    elif config.dataset_name == "s2_high":
        return get_s2_high_immersion(
            radius=config.radius,
            embedding_dim=config.embedding_dim,
            deformation_amp=config.deformation_amp,
            translation=trans,
            rotation=rot,
        )
    elif config.dataset_name == "nested_spheres":
        immersion_inner = get_s2_high_immersion(
            radius=config.minor_radius,
            embedding_dim=3,
            deformation_amp=config.deformation_amp,
            translation=torch.zeros(3),
            rotation=torch.eye(n=3),
        )
        immersion_mid = get_s2_high_immersion(
            radius=config.mid_radius,
            embedding_dim=3,
            deformation_amp=config.deformation_amp,
            translation=torch.zeros(3),
            rotation=torch.eye(n=3),
        )
        immersion_outer = get_s2_high_immersion(
            radius=config.major_radius,
            embedding_dim=3,
            deformation_amp=config.deformation_amp,
            translation=torch.zeros(3),
            rotation=torch.eye(n=3),
        )
        return immersion_inner, immersion_mid, immersion_outer
    elif config.dataset_name == "s2_low":
        return get_s2_low_immersion(
            radius=config.radius,
            deformation_amp=config.deformation_amp,
            embedding_dim=config.embedding_dim,
            rot=rot,
        )
    elif config.dataset_name == "t2_low":
        return get_t2_low_immersion(
            config.major_radius,
            config.minor_radius,
            config.deformation_amp,
            config.embedding_dim,
            rot,
        )
    elif config.dataset_name == "clelia_curve":
        return get_clelia_immersion(
            r=config.radius,
            c=config.clelia_c,
            embedding_dim=config.embedding_dim,
            rotation=rot,
        )
    elif config.dataset_name == "8_curve":
        return get_8_curve_immersion(
            embedding_dim=config.embedding_dim,
            rotation=rot,
        )
    else:
        raise InvalidConfigError(f"Unknown dataset: {config.dataset_name}")


def get_learned_immersion(
    model: Any, config: Any
) -> Callable[[torch.Tensor], torch.Tensor]:
    """
    Return an immersion function produced by a trained model.

    If the model is Euclidean, it decodes directly from z.
    If it is non-Euclidean, it maps angles (or torus angles) to z, then decodes.
    """

    def immersion_vm(angle: torch.Tensor) -> torch.Tensor:
        if config.dataset_name in {"s1_low", "s1_high"}:
            z = gs.array([gs.cos(angle[0]), gs.sin(angle[0])])

        elif config.dataset_name in {"s2_low", "s2_high"}:
            theta, phi = angle
            z = gs.array(
                [
                    gs.sin(theta) * gs.cos(phi),
                    gs.sin(theta) * gs.sin(phi),
                    gs.cos(theta),
                ]
            )

        elif config.dataset_name in {"t2_low", "t2_high"}:
            theta, phi = angle
            z = gs.array(
                [
                    (config.major_radius - config.minor_radius * gs.cos(theta)) * gs.cos(
                        phi
                    ),
                    (config.major_radius - config.minor_radius * gs.cos(theta)) * gs.sin(
                        phi
                    ),
                    config.minor_radius * gs.sin(theta),
                ]
            )
        else:
            raise InvalidConfigError(f"Unknown dataset: {config.dataset_name}")

        z = z.to(config.device)
        return model.decode(z)

    def immersion_euclidean(z: torch.Tensor) -> torch.Tensor:
        z = z.to(config.device)
        return model.decode(z)

    if is_euclidean_model(config.model_type):
        return immersion_euclidean
    if is_non_euclidean_model(config.model_type):
        return immersion_vm
    else:
        raise InvalidConfigError(f"Unknown model type: {config.model_type}")
