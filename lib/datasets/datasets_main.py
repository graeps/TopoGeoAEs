import os
import panda as pd

# Set Geomstats backend before importing it
os.environ["GEOMSTATS_BACKEND"] = "pytorch"

import torch  # noqa: E402
from torch.distributions.multivariate_normal import MultivariateNormal  # noqa: E402

import geomstats.backend as gs  # noqa: E402
from geomstats.geometry.special_orthogonal import SpecialOrthogonal  # noqa: E402
import numpy as np

from utils import embedd, rotate_translate, embedd_rotate_translate


def get_s1_low_immersion(deformation_type, radius, n_wiggles, deformation_amp, embedding_dim, rot):
    def immersion(angle):
        if deformation_type == "wiggles":
            amp = radius * (1 + deformation_amp * gs.cos(n_wiggles * angle))
        elif deformation_type == "bump":
            amp = radius * (
                    1
                    + deformation_amp * gs.exp(-5 * (angle - gs.pi / 2) ** 2)
                    + deformation_amp * gs.exp(-5 * (angle - 3 * gs.pi / 2) ** 2)
            )
        else:
            raise NotImplementedError(f"Unknown distortion: {deformation_type}")

        base_point = amp * gs.array([gs.cos(angle), gs.sin(angle)])
        base_point = gs.squeeze(base_point, axis=-1)
        if embedding_dim > 2:
            base_point = gs.concatenate([base_point, gs.zeros(embedding_dim - 2)])

        return gs.einsum("ij,j->i", rot, base_point)

    return immersion


def load_s1_low(rotation, n_points, radius, n_wiggles, deformation_amp, embedding_dim,
                noise_var, deformation_type="wiggles", random_seed=42,
                ):
    gs.random.seed(random_seed)
    rot = torch.eye(embedding_dim)
    if rotation == "random":
        rot = SpecialOrthogonal(n=embedding_dim).random_point()

    immersion = get_s1_low_immersion(deformation_type=deformation_type, radius=radius,
                                     n_wiggles=n_wiggles, deformation_amp=deformation_amp,
                                     embedding_dim=embedding_dim, rot=rot, )

    angles = gs.linspace(0, 2 * gs.pi, n_points)
    data = torch.stack([immersion(angle) for angle in angles])

    if noise_var != 0:
        noise = MultivariateNormal(
            loc=torch.zeros(embedding_dim),
            covariance_matrix=noise_var * torch.eye(embedding_dim),
        ).sample((n_points,))
        data = data + radius * noise

    return data, angles


def get_s2_low_immersion(radius, deformation_amp, embedding_dim, rot):
    """Returns a function mapping S² angles to R^embedding_dim with distortion."""

    def spherical(theta, phi):
        x = radius * gs.sin(theta) * gs.cos(phi)
        y = radius * gs.sin(theta) * gs.sin(phi)
        z = radius * gs.cos(theta)
        return gs.array([x, y, z])

    def immersion(angle_pair):
        theta, phi = angle_pair
        amplitude = (
                1 + deformation_amp * gs.exp(-5 * theta ** 2)
                + deformation_amp * gs.exp(-5 * (theta - gs.pi) ** 2)
        )

        point = amplitude * spherical(theta, phi)
        point = gs.squeeze(point, axis=-1)
        if embedding_dim > 3:
            point = gs.concatenate([point, gs.zeros(embedding_dim - 3)])
        return gs.einsum("ij,j->i", rot, point)

    return immersion


def load_s2_low(rotation, n_points, radius, deformation_amp, embedding_dim, noise_var, random_seed=42, ):
    """Generate noisy S²-immersed data in R^embedding_dim."""
    gs.random.seed(random_seed)
    rot = torch.eye(embedding_dim)
    if rotation == "random":
        rot = SpecialOrthogonal(n=embedding_dim).random_point()

    immersion = get_s2_low_immersion(radius, deformation_amp, embedding_dim, rot)

    sqrt_n_points = int(gs.sqrt(n_points))
    eps = 1e-4
    thetas = gs.linspace(0.01, gs.pi - eps, sqrt_n_points)
    phis = gs.linspace(eps, 2 * gs.pi - eps, sqrt_n_points)

    angle_grid = torch.cartesian_prod(thetas, phis)
    data = torch.stack([immersion(pair) for pair in angle_grid])

    if noise_var != 0:
        noise = MultivariateNormal(
            loc=torch.zeros(embedding_dim),
            covariance_matrix=radius * noise_var * torch.eye(embedding_dim),
        ).sample((sqrt_n_points ** 2,))
        data = data + noise

    return data, angle_grid


def get_t2_low_immersion(major_radius, minor_radius, deformation_amp, embedding_dim, rot):
    """Returns a function mapping T² angles to R^embedding_dim with distortion."""

    def immersion(angle_pair):
        theta, phi = angle_pair
        amplitude = (
                1
                + deformation_amp * gs.exp(-2 * (phi - gs.pi / 2) ** 2) * gs.exp(-2 * (theta - gs.pi) ** 2)
                + deformation_amp * gs.exp(-2 * (phi - 3 * gs.pi / 2) ** 2) * gs.exp(-2 * (theta - gs.pi) ** 2)
        )

        x = (major_radius - minor_radius * gs.cos(theta)) * gs.cos(phi)
        y = (major_radius - minor_radius * gs.cos(theta)) * gs.sin(phi)
        z = minor_radius * gs.sin(theta)

        point = amplitude * gs.array([x, y, z])
        point = gs.squeeze(point, axis=-1)
        if embedding_dim > 3:
            point = gs.concatenate([point, gs.zeros(embedding_dim - 3)])
        return gs.einsum("ij,j->i", rot, point)

    return immersion


def load_t2_low(
        rotation,
        n_points,
        major_radius,
        minor_radius,
        deformation_amp,
        embedding_dim,
        noise_var,
        random_seed=42,
):
    """Generate noisy T²-immersed data in R^embedding_dim."""
    gs.random.seed(random_seed)
    rot = torch.eye(embedding_dim)
    if rotation == "random":
        rot = SpecialOrthogonal(n=embedding_dim).random_point()

    immersion = get_t2_low_immersion(
        major_radius, minor_radius, deformation_amp, embedding_dim, rot
    )

    sqrt_n_points = int(gs.sqrt(n_points))
    thetas = gs.linspace(0, 2 * gs.pi, sqrt_n_points)
    phis = gs.linspace(0, 2 * gs.pi, sqrt_n_points)

    angle_grid = torch.cartesian_prod(thetas, phis)
    data = torch.stack([immersion(pair) for pair in angle_grid])

    if noise_var != 0:
        noise = MultivariateNormal(
            loc=torch.zeros(embedding_dim),
            covariance_matrix=major_radius * noise_var * torch.eye(embedding_dim),
        ).sample((sqrt_n_points ** 2,))
        data = data + noise

    return data, angle_grid


def get_s2_high_immersion(radius, deformation_amp, embedding_dim, translation, rotation):
    def immersion(angle_pair):
        theta, phi = angle_pair

        amplitude = (
                1 + 0.5 * deformation_amp * (gs.exp(-5 * theta ** 2)
                                             + gs.exp(-5 * (theta - gs.pi) ** 2))
        )

        # Base S^2 coordinates in R^3
        x = radius * gs.sin(theta) * gs.cos(phi)
        y = radius * gs.sin(theta) * gs.sin(phi)
        z = radius * gs.cos(theta)

        point = amplitude * gs.array([x, y, z])
        point = gs.squeeze(point, axis=-1)

        if embedding_dim > 3:
            point = gs.concatenate([point, gs.zeros(embedding_dim - 3)])

        if embedding_dim > 5:
            for i in range(2):
                point[3 + i] = gs.sin((i + 1) * theta) * gs.cos((i + 2) * phi)

        point = rotate_translate(point, translation, rotation)
        return point

    return immersion

    # def immersion(angle_pair):
    #     theta, phi = angle_pair
    #     amplitude1 = (
    #             1 + deformation_amp * (gs.exp(-5 * theta ** 2)
    #                                          + gs.exp(-5 * (theta - gs.pi) ** 2))
    #     )
    #     amplitude2 = (
    #             1 + deformation_amp * (gs.exp(-5 * (theta - gs.pi / 2) ** 2)
    #                                    + gs.exp(-5 * (theta - 3 * gs.pi / 2) ** 2))
    #     )
    #     amplitude3 = 1 + deformation_amp * (torch.cos(theta) + torch.cos(2 * theta))
    #     amplitude4 = 1 + deformation_amp * (torch.cos(phi) + torch.cos(2 * phi))
    #
    #     x = radius * gs.sin(theta) * gs.cos(phi)
    #     y = radius * gs.sin(theta) * gs.sin(phi)
    #     z = radius * gs.cos(theta)
    #
    #     point = gs.array([x, y, z])
    #     point = gs.squeeze(point, axis=-1)
    #     if embedding_dim > 6:
    #         point = gs.concatenate([point, gs.zeros(embedding_dim - 3)])
    #         point[3] = amplitude1
    #         point[4] = amplitude2
    #         point[5] = amplitude3
    #         point[6] = amplitude4
    #
    #     point = rotate_translate(point, translation, rotation)
    #     return point


def load_s2_high(n_points, radius, noise_var, embedding_dim, deformation_amp, translation, rotation, random_seed=42):
    gs.random.seed(random_seed)
    torch.manual_seed(random_seed)

    if rotation == "random":
        rot = SpecialOrthogonal(n=embedding_dim).random_point()
    else:
        rot = torch.eye(embedding_dim)
    if translation is None:
        trans = torch.zeros(embedding_dim)
    else:
        trans = translation

    immersion = get_s2_high_immersion(radius=radius, embedding_dim=embedding_dim, deformation_amp=deformation_amp,
                                      translation=trans, rotation=rot)
    sqrt_n_points = int(gs.sqrt(n_points))
    thetas = gs.arccos(np.linspace(0.99, -0.99, sqrt_n_points))  # For more uniform distribution of sample points
    # thetas = gs.linspace(0.01, np.pi - 0.01, sqrt_n_points)
    phis = gs.linspace(0.01, 2 * np.pi - 0.01, sqrt_n_points)
    angle_grid = torch.cartesian_prod(thetas, phis)

    data = torch.stack([immersion(pair) for pair in angle_grid])

    if noise_var != 0:
        noise = MultivariateNormal(
            loc=torch.zeros(embedding_dim),
            covariance_matrix=noise_var * torch.eye(embedding_dim),
        ).sample((sqrt_n_points ** 2,))
        data = data + noise

    return data, angle_grid


def get_t2_high_immersion(major_radius, minor_radius, embedding_dim, deformation_amp, translation, rotation):
    def immersion(angle_pair):
        theta, phi = angle_pair

        amplitude1 = (1 + 0.5 * (
                deformation_amp * gs.exp(-2 * (phi - gs.pi / 2) ** 2) * gs.exp(-2 * (theta - gs.pi) ** 2)
                + deformation_amp * gs.exp(-2 * (phi - 3 * gs.pi / 2) ** 2) * gs.exp(-2 * (theta - gs.pi) ** 2))
                      )
        amplitude2 = (
                deformation_amp * gs.exp(-2 * (phi - gs.pi / 2) ** 2) * gs.exp(-2 * (theta - gs.pi / 2) ** 2)
                + deformation_amp * gs.exp(-2 * (phi - 3 * gs.pi / 2) ** 2) * gs.exp(-2 * (theta - 3 * gs.pi / 2) ** 2)
        )

        # Standard 3D torus coordinates
        x = (major_radius - minor_radius * gs.cos(theta)) * gs.cos(phi)
        y = (major_radius - minor_radius * gs.cos(theta)) * gs.sin(phi)
        z = minor_radius * gs.sin(theta)

        point = amplitude1 * gs.array([x, y, z])
        point = gs.squeeze(point, axis=-1)

        if embedding_dim > 3:
            point = gs.concatenate([point, gs.zeros(embedding_dim - 3)])

        if embedding_dim > 7:
            point[3] = amplitude2
        for i in range(2):
            point[4 + i] = gs.sin((i + 1) * theta) * gs.cos((i + 2) * phi)

        point = rotate_translate(point, translation, rotation)
        return point

    return immersion


def load_t2_high(n_points, major_radius, minor_radius, noise_var, embedding_dim, deformation_amp,
                 translation="random",
                 rotation="random", random_seed=42, ):
    gs.random.seed(random_seed)
    torch.manual_seed(random_seed)

    rot = torch.eye(embedding_dim)
    if rotation == "random":
        rot = SpecialOrthogonal(n=embedding_dim).random_point()
    trans = torch.zeros(embedding_dim)
    if translation == "random":
        trans = 100 * torch.rand(embedding_dim)

    immersion = get_t2_high_immersion(major_radius=major_radius, minor_radius=minor_radius, embedding_dim=embedding_dim,
                                      deformation_amp=deformation_amp,
                                      translation=trans, rotation=rot)

    sqrt_n_points = int(gs.sqrt(n_points))
    eps = 1e-4
    thetas = gs.linspace(eps, 2 * gs.pi - eps, sqrt_n_points)
    phis = gs.linspace(eps, 2 * gs.pi - eps, sqrt_n_points)
    angle_grid = torch.cartesian_prod(thetas, phis)

    data = torch.stack([immersion(pair) for pair in angle_grid])

    if noise_var != 0:
        noise = MultivariateNormal(
            loc=torch.zeros(embedding_dim),
            covariance_matrix=major_radius * noise_var * torch.eye(embedding_dim),
        ).sample((sqrt_n_points ** 2,))
        data = data + noise

    return data, angle_grid


def get_s1_high(deformation_amp, embedding_dim, translation, rotation, deformation="peaks"):
    def immersion(angle):
        if deformation == "peaks":
            x1 = gs.sin(angle)
            x2 = gs.cos(angle)
            x3 = gs.sin(0 * angle)
            x4 = deformation_amp * gs.cos(2 * angle)
            x5 = deformation_amp * gs.sin(3 * angle)
            x6 = deformation_amp * gs.cos(4 * angle)
            x7 = deformation_amp * gs.sin(5 * angle)
            x8 = deformation_amp * gs.cos(6 * angle)
            x9 = deformation_amp * gs.sin(7 * angle)
            x10 = deformation_amp * gs.cos(7 * angle)
        else:
            x1 = gs.sin(angle)
            x2 = gs.cos(angle)
            x3 = gs.sin(2 * angle)
            x4 = deformation_amp * gs.cos(2 * angle)
            x5 = deformation_amp * gs.sin(3 * angle)
            x6 = deformation_amp * gs.cos(3 * angle)
            x7 = deformation_amp * gs.sin(4 * angle)
            x8 = deformation_amp * gs.cos(4 * angle)
            x9 = deformation_amp * gs.sin(5 * angle)
            x10 = deformation_amp * gs.cos(5 * angle)

        point = gs.array([x1, x2, x3, x4, x5, x6, x7, x8, x9, x10])[:embedding_dim]
        point = gs.squeeze(point, axis=-1)
        point = rotate_translate(point, translation, rotation)
        return point

    return immersion


def load_s1_high(n_points, noise_var, deformation_amp, embedding_dim, translation, rotation, random_seed=42):
    rot = torch.eye(embedding_dim)
    if rotation == "random":
        rot = SpecialOrthogonal(n=embedding_dim).random_point()
    trans = torch.zeros(embedding_dim)
    if translation == "random":
        trans = 100 * torch.rand(embedding_dim)
    immersion = get_s1_high(deformation_amp=deformation_amp, embedding_dim=embedding_dim, translation=trans,
                            rotation=rot)
    angles = torch.linspace(0, 2 * torch.pi, n_points)
    data = torch.stack([immersion(angle) for angle in angles])

    if noise_var != 0:
        noise = MultivariateNormal(loc=torch.zeros(embedding_dim),
                                   covariance_matrix=noise_var * torch.eye(embedding_dim), ).sample((n_points,))
        data = data + noise

    return data, angles
