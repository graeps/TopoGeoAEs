import os

from .errors import InvalidConfigError

os.environ["GEOMSTATS_BACKEND"] = "pytorch"
import geomstats.backend as gs
import torch
from geomstats.geometry.special_orthogonal import SpecialOrthogonal

from geomstats.geometry.base import ImmersedSet
from geomstats.geometry.euclidean import Euclidean

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
    get_sphere_high_dim_bump_immersion
)


class NeuralManifoldIntrinsic(ImmersedSet):
    def __init__(self, dim, neural_embedding_dim, neural_immersion, equip=True):
        self.neural_embedding_dim = neural_embedding_dim
        super().__init__(dim=dim, equip=equip)
        self.neural_immersion = neural_immersion

    def immersion(self, point):
        return self.neural_immersion(point)

    def _define_embedding_space(self):
        return Euclidean(dim=self.neural_embedding_dim)


def get_true_immersion(config):
    rot = torch.eye(n=config.embedding_dim)
    trans = torch.zeros(config.embedding_dim)
    if config.rotation == "random":
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
        bump_dims = [config.embedding_dim - 3, config.embedding_dim - 2, config.embedding_dim - 1]
        bump_centers = [(torch.pi / 4, torch.pi / 2), (torch.pi / 4, 3 * torch.pi / 2), (torch.pi / 2, torch.pi / 2)]
        return get_sphere_high_dim_bump_immersion(radius=config.radius, deformation_amp=config.deformation_amp,
                                                  bump_dim=bump_dims, bump_center=bump_centers,
                                                  embedding_dim=config.embedding_dim,
                                                  rotation=rot)
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
            rotation=rot
        )
    elif config.dataset_name == "t2_high":
        return get_t2_high_immersion(
            major_radius=config.major_radius,
            minor_radius=config.minor_radius,
            embedding_dim=config.embedding_dim,
            deformation_amp=config.deformation_amp,
            translation=trans,
            rotation=rot
        )
    elif config.dataset_name == "interlocked_tori":
        immersion1 = get_t2_high_immersion(major_radius=config.major_radius,
                                           minor_radius=config.minor_radius,
                                           embedding_dim=3,
                                           deformation_amp=config.deformation_amp,
                                           translation=torch.zeros(3),
                                           rotation=torch.eye(n=3),
                                           )
        immersion2 = get_t2_high_immersion(major_radius=config.major_radius,
                                           minor_radius=config.minor_radius,
                                           embedding_dim=3,
                                           deformation_amp=config.deformation_amp,
                                           translation=torch.zeros(3),
                                           rotation=torch.eye(n=3),
                                           )
        return immersion1, immersion2
    elif config.dataset_name == "s2_high":
        return get_s2_high_immersion(radius=config.radius, embedding_dim=config.embedding_dim,
                                     deformation_amp=config.deformation_amp,
                                     translation=trans, rotation=rot)
    elif config.dataset_name == "nested_spheres":
        immersion_inner = get_s2_high_immersion(radius=config.minor_radius, embedding_dim=3,
                                                deformation_amp=config.deformation_amp,
                                                translation=torch.zeros(3), rotation=torch.eye(n=3))
        immersion_mid = get_s2_high_immersion(radius=config.mid_radius, embedding_dim=3,
                                              deformation_amp=config.deformation_amp,
                                              translation=torch.zeros(3), rotation=torch.eye(n=3))
        immersion_outer = get_s2_high_immersion(radius=config.major_radius, embedding_dim=3,
                                                deformation_amp=config.deformation_amp,
                                                translation=torch.zeros(3), rotation=torch.eye(n=3))
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
            rotation=rot
        )
    elif config.dataset_name == "8_curve":
        return get_8_curve_immersion(
            embedding_dim=config.embedding_dim,
            rotation=rot
        )
    else:
        raise InvalidConfigError(f"Unknown dataset: {config.dataset_name}")


def get_learned_immersion(model, config):
    def immersion_vm(angle):
        if config.dataset_name in {"s1_low", "s1_high"}:
            z = gs.array([gs.cos(angle[0]), gs.sin(angle[0])])

        elif config.dataset_name in {"s2_low", "s2_high"}:
            theta, phi = angle
            z = gs.array([
                gs.sin(theta) * gs.cos(phi),
                gs.sin(theta) * gs.sin(phi),
                gs.cos(theta),
            ])

        elif config.dataset_name in {"t2_low", "t2_high"}:
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
